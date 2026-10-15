"""Reviewable similarity candidates and explicit approval, scoped to the active filters."""

import hashlib
import sqlite3
import uuid

import streamlit as st

from bnivoice360.domain import IssueFilters
from bnivoice360.services.agent.workflow import BatchAgent, ReviewRequired, context_key
from bnivoice360.ui.credentials import groq_key


def _api_key() -> str:
    return groq_key()


def _cancel(agent: BatchAgent, session_id: str):
    proposal = st.session_state.get("batch_proposal")
    if proposal:
        agent.audit.cancel(proposal.operation_id, session_id)
    for key in ("batch_proposal", "batch_approval", "batch_draft_signature"):
        st.session_state.pop(key, None)


def _render_draft(proposal):
    if proposal.mode == "manual":
        if proposal.reason == "manual_template":
            st.info("Template manual dipilih. Aktifkan 'Buat draft dengan AI Groq' untuk membuat ringkasan AI.")
        else:
            reasons = {
                "remote_disabled": "fitur draft AI dinonaktifkan dalam konfigurasi",
                "missing_api_key": "API key Groq belum tersedia",
                "masked_content_unavailable": "masked_content keluhan terpilih belum tersedia",
                "context_budget": "keluhan terpilih melebihi batas konteks AI",
                "unverified_number": "ringkasan AI memuat angka yang belum dapat diverifikasi",
                "invalid_json": "respons AI belum berupa JSON yang valid setelah percobaan ulang",
                "invalid_schema": "struktur respons AI belum sesuai format draft setelah percobaan ulang",
                "agent_deadline": "pembuatan draft AI melewati batas waktu",
            }
            detail = reasons.get(proposal.reason)
            if detail is None:
                detail = (
                    "layanan Groq belum memberikan respons yang dapat digunakan"
                    if proposal.reason.startswith("provider_")
                    else "ringkasan AI belum lolos pemeriksaan bukti dan isi"
                )
            st.info(f"Template manual digunakan karena {detail}. Anda tetap dapat meninjau keluhan.")
    else:
        st.caption(
            "Interpretasi AI perlu diperiksa. Kutipan harus mendukung setiap temuan; dugaan belum terbukti."
        )
    for kind, title in (
        ("findings", "Temuan dari keluhan"),
        ("hypotheses", "Dugaan yang perlu diverifikasi"),
        ("actions", "Usulan tindak lanjut"),
    ):
        if proposal.draft[kind]:
            st.markdown("**" + title + "**")
        for item in proposal.draft[kind]:
            st.text(item["text"])
            if item.get("team"):
                st.caption("Usulan unit tujuan: " + item["team"])
            for reference in item["evidence"]:
                st.text(
                    "Sumber " + proposal.aliases[reference["ticket"]] + ": " + reference["quote"]
                )


def render_batch_agent(agent: BatchAgent, filters: IssueFilters, topic_id: int | None):
    anchor_id = st.session_state.get("batch_anchor_id")
    if not anchor_id:
        return
    session_id = st.session_state.setdefault("batch_session_id", str(uuid.uuid4()))
    role = st.session_state["selected_role"]
    context = context_key(filters, topic_id)
    with st.container(border=True):
        st.subheader("Keluhan Serupa & Eskalasi Batch")
        st.caption(
            "Pilih kandidat berdasarkan isi keluhan. Kemiripan vector tidak membuktikan penyebab yang sama."
        )
        try:
            signature = repr((anchor_id, context, role))
            if signature != st.session_state.get("batch_context_signature"):
                _cancel(agent, session_id)
                st.session_state.pop("batch_search", None)
                st.session_state["batch_context_signature"] = signature
            st.caption("Keluhan awal: " + anchor_id)
            left, right = st.columns(2)
            threshold = left.slider(
                "Minimum similarity",
                0.0,
                1.0,
                min(1.0, max(0.0, agent.settings.agent_similarity_threshold)),
                step=0.01,
                key="batch_threshold",
            )
            limit = right.number_input(
                "Maksimal kandidat tambahan",
                min_value=1,
                max_value=50,
                value=min(50, max(1, agent.settings.agent_candidate_limit)),
                key="batch_limit",
            )
            cross_topic = st.checkbox(
                "Cari lintas topik dalam filter aktif", key="batch_cross_topic"
            )
            search_options = repr((signature, threshold, limit, cross_topic))
            if st.session_state.get("batch_search_options") != search_options:
                _cancel(agent, session_id)
                st.session_state.pop("batch_search", None)
                st.session_state["batch_search_options"] = search_options
            if st.button("Cari kandidat", key="batch_search_button"):
                _cancel(agent, session_id)
                st.session_state.pop("batch_search", None)
                with st.spinner("Mencari vector keluhan serupa..."):
                    st.session_state["batch_search"] = agent.search(
                        anchor_id,
                        filters,
                        topic_id,
                        cross_topic=cross_topic,
                        threshold=threshold,
                        limit=int(limit),
                        session_id=session_id,
                    )
                st.session_state["batch_search_token"] = uuid.uuid4().hex
            search = st.session_state.get("batch_search")
            if search is None:
                return
            table = search.candidates.copy()
            table["date_posted"] = table["date_posted"].astype(str)
            table = table[
                [
                    "suara_nasabah_id",
                    "similarity",
                    "content",
                    "date_posted",
                    "topic_cluster",
                    "escalation_status",
                ]
            ]
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption(
                f"{search.missing_embeddings} keluhan dalam cakupan tidak memiliki embedding dan dilewati."
            )
            options = search.candidates["suara_nasabah_id"].astype(str).tolist()
            token = st.session_state["batch_search_token"]
            selected = tuple(
                st.multiselect(
                    "Pilih keluhan untuk eskalasi",
                    options,
                    default=[anchor_id],
                    key="batch_selection_" + token,
                )
            )
            instruction = st.text_area(
                "Arahan investigasi",
                value="Ringkas keluhan terpilih dan usulkan tindak lanjut eskalasi.",
                max_chars=1000,
                key="batch_instruction",
            )
            api_key = _api_key() if agent.settings.agent_remote_enabled else ""
            ai_available = agent.settings.agent_remote_enabled and bool(api_key)
            use_ai = st.checkbox(
                "Buat draft dengan AI Groq",
                value=ai_available,
                disabled=not ai_available,
                key="batch_use_ai_masked",
            )
            use_ai = use_ai and ai_available
            st.caption(
                "Draft AI menggunakan cuplikan masked_content dari data sumber. "
                "Isi keluhan asli dan author tidak dikirim ke Groq. "
                "Jika masked_content tidak tersedia, aplikasi menggunakan template manual."
            )
            if not agent.settings.agent_remote_enabled:
                st.caption(
                    "Draft AI dinonaktifkan oleh BNIVOICE360_AGENT_REMOTE_ENABLED=0. "
                    "Ubah menjadi 1 untuk mengaktifkannya."
                )
            elif not api_key:
                st.caption("Isi API key Groq di .streamlit/secrets.toml atau environment GROQ_API_KEY untuk menggunakan draft AI.")
            draft_signature = hashlib.sha256(
                repr(("masked-draft-v1", token, selected, instruction, use_ai)).encode()
            ).hexdigest()
            if st.session_state.get("batch_draft_signature") != draft_signature:
                _cancel(agent, session_id)
                st.session_state["batch_draft_signature"] = draft_signature
            if st.button("Siapkan preview eskalasi", key="batch_prepare", disabled=not selected):
                _cancel(agent, session_id)
                with st.spinner("Menyiapkan draft dan memeriksa guardrails..."):
                    proposal = agent.prepare(
                        search,
                        selected,
                        context,
                        session_id,
                        role,
                        instruction=instruction,
                        use_ai=use_ai,
                        api_key=api_key if use_ai else "",
                    )
                st.session_state["batch_proposal"] = proposal
                st.session_state["batch_draft_signature"] = draft_signature
            proposal = st.session_state.get("batch_proposal")
            if proposal:
                st.write(f"Preview: {len(proposal.ids)} keluhan Open akan menjadi Escalated.")
                _render_draft(proposal)
                approved = st.checkbox(
                    "Saya sudah memeriksa tiket dan menyetujui eskalasi batch ini.",
                    key="batch_approval",
                )
                if st.button(
                    f"Eskalasi {len(proposal.ids)} keluhan",
                    key="batch_commit",
                    disabled=not approved,
                ):
                    count = agent.commit(
                        proposal, session_id, role, context, selected, approved=approved
                    )
                    st.session_state.pop("batch_proposal", None)
                    st.session_state.pop("batch_search", None)
                    st.session_state.pop("batch_anchor_id", None)
                    st.session_state["escalation_notice"] = (
                        f"{count} keluhan berhasil dieskalasi dan dicatat di audit."
                    )
                    st.rerun()
            if st.button("Tutup dan batalkan preview", key="batch_cancel"):
                _cancel(agent, session_id)
                st.session_state.pop("batch_anchor_id", None)
                st.session_state.pop("batch_search", None)
                st.rerun()
        except ReviewRequired as exc:
            _cancel(agent, session_id)
            st.session_state.pop("batch_search", None)
            st.warning(str(exc))
        except (OSError, sqlite3.Error, ValueError, ImportError):
            # No raw provider exceptions, secret values or customer text in error messages.
            st.error(
                "Fitur batch belum tersedia. Periksa file embedding atau database audit; "
                "tidak ada eskalasi batch yang disimpan."
            )
