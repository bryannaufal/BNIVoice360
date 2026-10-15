"""Conversational complaint search, explicit context, draft review and human approval."""

import copy
import hashlib
import sqlite3
import uuid

import streamlit as st

from bnivoice360.container import AppServices
from bnivoice360.services.agent.guardrails import GuardrailError
from bnivoice360.services.agent.workflow import ReviewRequired
from bnivoice360.services.companion_state import CompanionState, Stage
from bnivoice360.ui.components import header
from bnivoice360.ui.credentials import groq_key


def _reply(text: str, state: CompanionState):
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": text,
            "tools": copy.deepcopy(state.tool_trace),
            "sources": copy.deepcopy(state.analysis_sources),
        }
    )
    if len(st.session_state.chat_history) > 41:
        st.session_state.chat_history = [
            st.session_state.chat_history[0],
            *st.session_state.chat_history[-40:],
        ]


def render(services: AppServices) -> None:
    header(
        "BNIVoice360 Companion",
        "Temukan issue, tinjau keluhan asli, dan siapkan eskalasi dengan konteks yang jelas.",
        "Complaint Investigation Companion",
    )
    state = st.session_state.setdefault("companion_context", CompanionState())
    # Streamlit can retain a session created before new context fields were added.
    for name, value in vars(CompanionState()).items():
        if not hasattr(state, name):
            setattr(state, name, copy.deepcopy(value))
    session_id = st.session_state.setdefault("companion_session_id", str(uuid.uuid4()))
    role = st.session_state["selected_role"]
    conversation = services.conversation
    api_key = groq_key() if services.batch_agent.settings.companion_remote_enabled else ""

    def respond(text):
        _reply(text, state)

    with st.expander("Contoh pertanyaan"):
        st.markdown(
            "- Apakah ada issue terkait QRIS akhir-akhir ini?\n"
            "- Apakah ada keluhan mendesak terkait produk tertentu?\n"
            "- Tampilkan issue IT yang masih Escalated.\n"
            "- Yang masih Open saja.\n"
            "- Tampilkan keluhan transfer gagal 30 hari terakhir dari PlayStore.\n"
            "- Keluhan apa yang meningkat minggu ini dibanding periode sebelumnya?\n"
            "- Apakah keluhan QRIS lebih banyak muncul di X atau PlayStore?\n"
            "- Topik mana yang punya backlog Open terbanyak dan paling lama?\n"
            "- Masalah spesifik apa yang dilaporkan dalam keluhan transfer?\n"
            "- Buat briefing keluhan minggu ini untuk tim IT.\n"
            "- Pilih nomor 2.\n- Apakah ada keluhan serupa?\n"
            "- Saya ingin escalate keluhan tersebut."
            "\n- Revisi draft agar lebih singkat dan formal."
        )
    st.caption(
        "Cuplikan keluhan yang dikirim ke Groq menggunakan masked_content dari sumber data. "
        "Arahan petugas tetap diperiksa sebelum dikirim."
    )
    if not api_key:
        st.info(
            "Chat Companion memerlukan koneksi Groq. Atur API key untuk mulai percakapan. "
            "Tombol pencarian, pilihan tiket, dan form eskalasi tetap tersedia."
        )
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.text(message["content"])
            scopes = set()
            for source in message.get("sources", []):
                scope = source["data"].get("scope")
                if scope:
                    label = (
                        f"{scope.get('start_date') or 'Awal data'} sampai {scope['end_date']}"
                        f" · Sentimen: {scope['sentiment']} · Kanal: {scope['source'] or 'Semua'}"
                        f" · Status: {scope['status']} · Kategori: {scope['category'] or 'Semua'}"
                        f" · Kata kunci: {scope['keyword'] or 'Semua'}"
                        f" · Topik: {scope['topic'] or 'Semua'} · Urgent saja: {'Ya' if scope['urgent_only'] else 'Tidak'}"
                    )
                    if label not in scopes:
                        st.caption(label)
                        scopes.add(label)
            if message.get("tools"):
                with st.expander("Tools yang digunakan", expanded=True):
                    for trace in message["tools"]:
                        st.text(f"{trace.get('source', '')} {trace['tool']} — {trace['status']}")
                        st.caption(f"Durasi: {trace.get('duration_ms', 0)} ms")
                        if "arguments" in trace:
                            st.json(trace["arguments"])
            if message.get("sources"):
                with st.expander("Sumber dan hasil analisis"):
                    for source in message["sources"]:
                        st.text(f"[{source['id']}] {source['tool']}")
                        st.json(source["data"], expanded=False)
    try:
        conversation.set_role(state, role, session_id)
        if prompt := st.chat_input(
            "Tanyakan keluhan atau tulis langkah berikutnya...",
            max_chars=1000,
            disabled=not api_key,
        ):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("Memeriksa permintaan dan konteks keluhan..."):
                reply = conversation.handle(state, prompt, session_id, role, api_key=api_key)
            respond(reply)
            st.rerun()
        if state.query:
            st.markdown("**Cakupan pencarian aktif**")
            st.caption(conversation.scope(state))
        else:
            st.caption(
                "Mulai dengan topik bebas atau pilih pencarian berikut. Periode default: 30 hari terakhir."
            )
            for column, label, arguments in zip(
                st.columns(3),
                ["Issue urgent", "IT · Escalated", "Keluhan Open"],
                [
                    {"action": "display", "urgent_only": True},
                    {"action": "display", "categories": ["IT"], "status": "Escalated"},
                    {"action": "display", "status": "Open"},
                ],
                strict=True,
            ):
                if column.button(label, key="companion_start_" + label, use_container_width=True):
                    respond(conversation.execute_workflow(state, arguments, session_id, role))
                    st.rerun()
        if state.stage == Stage.OFFER:
            with st.container(border=True):
                st.write(
                    f"{state.total:,} keluhan ditemukan. Ingin melihat detail dan memilih tiket?"
                )
                show, dismiss = st.columns(2)
                if show.button(
                    "Tampilkan issue",
                    key="companion_show_results",
                    type="primary",
                    use_container_width=True,
                ):
                    respond(
                        conversation.execute_workflow(
                            state, {"action": "display", "refinement": True}, session_id, role
                        )
                    )
                    st.rerun()
                if dismiss.button("Nanti dulu", key="companion_dismiss", use_container_width=True):
                    respond(
                        conversation.execute_workflow(
                            state, {"action": "dismiss"}, session_id, role
                        )
                    )
                    st.rerun()
        if state.analysis_evidence:
            with st.expander("Pilih keluhan dari analisis terbaru"):
                for alias in state.analysis_evidence:
                    if st.button(f"Pilih {alias}", key="companion_evidence_" + alias):
                        state.tool_trace, state.analysis_sources = [], []
                        respond(conversation.select_evidence(state, alias, session_id))
                        st.rerun()
        if state.stage == Stage.PERIOD:
            buttons = st.columns(3)
            for column, label, period in zip(
                buttons,
                ["7 hari terakhir", "30 hari terakhir", "Semua periode"],
                ["7d", "30d", "all"],
                strict=True,
            ):
                if column.button(label, key="companion_period_" + period):
                    respond(
                        conversation.execute_workflow(
                            state, {"action": "period", "period": period}, session_id, role
                        )
                    )
                    st.rerun()
        if state.result_ids:
            st.markdown(f"**Hasil pencarian · {state.total:,} keluhan**")
            if state.total > len(state.result_ids):
                st.caption(
                    f"Menampilkan {len(state.result_ids)} hasil terbaru. Persempit filter untuk melihat hasil lain."
                )
            if state.result_kind == "similar":
                st.caption(
                    "Hasil kemiripan: kandidat Negative dan Open; keyword tidak membatasi vector search."
                )
            rows = conversation.visible_rows(state)
            for _, row in rows.iterrows():
                ticket = str(row["suara_nasabah_id"])
                number = state.result_ids.index(ticket) + 1
                digest = hashlib.sha256(ticket.encode()).hexdigest()[:16]
                with st.container(border=True):
                    st.write(f"{number}. ID: {ticket}")
                    st.caption(
                        f"{row['date_posted']} · {row['source']} · {row['topic_cluster']} · {row['escalation_status']}"
                    )
                    categories = [
                        name
                        for name in ("IT", "Bisnis", "Operasional")
                        if row.get("is_" + name) == 1
                    ]
                    st.caption(
                        " · ".join(
                            [*categories, "Urgent" if row.get("is_Urgent") == 1 else "Non-urgent"]
                        )
                    )
                    if ticket in state.scores:
                        st.caption(f"Similarity: {state.scores[ticket]:.3f}")
                    st.text(str(row["content"])[:600])
                    if len(str(row["content"])) > 600:
                        with st.expander("Baca keluhan lengkap"):
                            st.text(str(row["content"]))
                    label = "Batalkan pilihan" if ticket in state.selected_ids else "Pilih keluhan"
                    if st.button(label, key="companion_pick_" + digest):
                        ids = (
                            tuple(i for i in state.selected_ids if i != ticket)
                            if ticket in state.selected_ids
                            else (*state.selected_ids, ticket)
                        )
                        state.tool_trace, state.analysis_sources = [], []
                        respond(conversation.select(state, ids, session_id))
                        st.rerun()
            previous, summary, next_page = st.columns([1, 2, 1])
            if previous.button("Sebelumnya", key="companion_prev", disabled=state.page == 0):
                state.page -= 1
                st.rerun()
            summary.caption(
                f"Halaman {state.page + 1} dari {max(1, (len(state.result_ids) + 9) // 10)} · {state.total} hasil"
            )
            if next_page.button(
                "Berikutnya",
                key="companion_next",
                disabled=(state.page + 1) * 10 >= len(state.result_ids),
            ):
                state.page += 1
                st.rerun()
        if state.selected_ids:
            st.write("Keluhan terpilih: " + ", ".join(state.selected_ids))
            if st.button(
                "Cari keluhan serupa",
                key="companion_similar",
                disabled=len(state.selected_ids) != 1,
            ):
                respond(
                    conversation.execute_workflow(state, {"action": "similar"}, session_id, role)
                )
                st.rerun()
            if state.stage == Stage.CROSS_TOPIC:
                if st.button("Cari lintas topik", key="companion_cross_topic"):
                    respond(
                        conversation.execute_workflow(
                            state, {"action": "similar", "cross_topic": True}, session_id, role
                        )
                    )
                    st.rerun()
            st.markdown("**Informasi eskalasi dari petugas**")
            form_key = hashlib.sha256(
                repr((state.selected_ids, state.team, state.request, state.examination)).encode()
            ).hexdigest()[:16]
            with st.form("companion_details_" + form_key):
                teams = ["", "IT", "Operasional", "Bisnis"]
                team = st.selectbox(
                    "Unit tujuan",
                    teams,
                    index=teams.index(state.team),
                    format_func=lambda v: v or "Pilih unit",
                    key="companion_team_" + form_key,
                )
                request = st.text_area(
                    "Permintaan kepada unit tujuan",
                    value=state.request,
                    max_chars=1000,
                    key="companion_request_" + form_key,
                )
                examination = st.text_area(
                    "Pemeriksaan yang sudah dilakukan (opsional)",
                    value=state.examination,
                    max_chars=1000,
                    key="companion_examination_" + form_key,
                )
                submitted = st.form_submit_button(
                    "Buat draft eskalasi", key="companion_draft_submit"
                )
            if submitted:
                conversation.update_details(state, session_id, team, request, examination)
                state.tool_trace, state.analysis_sources = [], []
                with st.spinner("Menyusun draft dari bukti dan informasi petugas..."):
                    respond(conversation.build_draft(state, session_id, role, api_key))
                st.rerun()
        if state.draft_text:
            st.markdown("**Draft untuk ditinjau**")
            st.caption(
                "Keluhan asli dicantumkan utuh. Ringkasan menjelaskan laporan nasabah; "
                "saran AI perlu ditinjau. Revisi draft memerlukan persetujuan baru."
            )
            if state.draft_mode != "groq":
                reason = state.draft_reason
                if reason == "analysis_unverified_number":
                    detail = "angka dalam ringkasan AI belum dapat diverifikasi terhadap bukti"
                elif reason == "model_unavailable" or reason.startswith("provider_"):
                    detail = "layanan AI tidak tersedia atau gagal menghasilkan respons yang dapat digunakan"
                elif reason == "agent_deadline":
                    detail = "pembuatan ringkasan AI melewati batas waktu"
                elif reason == "masked_content_unavailable":
                    detail = "masked_content keluhan belum tersedia untuk dikirim ke AI"
                elif reason == "context_budget":
                    detail = "isi keluhan terpilih melebihi batas konteks AI"
                else:
                    detail = "ringkasan AI belum memenuhi pemeriksaan bukti dan isi"
                st.info(
                    f"Draft dasar: {detail}. "
                    "Keluhan asli tetap lengkap dan dapat ditinjau."
                )
            st.code(state.draft_text, language=None, wrap_lines=True)
            st.download_button(
                "Unduh draft (.txt)",
                state.draft_text.encode("utf-8"),
                file_name="draft_eskalasi.txt",
                mime="text/plain",
                key="companion_download",
            )
            with st.form("companion_revision_form"):
                revision = st.text_area(
                    "Arahan revisi draft",
                    placeholder="Contoh: lebih singkat dan formal; tekankan kendala login.",
                    max_chars=1000,
                    key="companion_revision_instruction",
                )
                revise = st.form_submit_button("Revisi draft", key="companion_revise_submit")
            if revise:
                if revision.strip():
                    with st.spinner("Merevisi draft berdasarkan tiket dan arahan Anda..."):
                        respond(
                            conversation.revise_draft(state, session_id, role, revision, api_key)
                        )
                    st.rerun()
                else:
                    st.info("Isi arahan revisi terlebih dahulu.")
            if state.proposal:
                approved = st.checkbox(
                    "Saya telah memeriksa draft dan menyetujui perubahan tiket terpilih menjadi Escalated.",
                    key="companion_approve_" + state.proposal.operation_id,
                )
                if st.button(
                    f"Konfirmasi eskalasi {len(state.selected_ids)} tiket",
                    key="companion_commit",
                    disabled=not approved,
                ):
                    count = conversation.commit(state, session_id, role, approved=approved)
                    state.tool_trace, state.analysis_sources = [], []
                    respond(
                        f"{count} tiket berubah menjadi Escalated dan dicatat dalam audit. Draft tidak dikirim ke unit lain."
                    )
                    st.rerun()
            else:
                st.caption(
                    "Preview perubahan status tidak tersedia: tiket sudah ditindaklanjuti atau konteks perlu diperbarui."
                )
        if state.query or state.selected_ids:
            if st.button("Batalkan konteks dan preview", key="companion_reset"):
                respond(
                    conversation.execute_workflow(state, {"action": "cancel"}, session_id, role)
                )
                st.rerun()
    except ReviewRequired as exc:
        conversation.invalidate(state, session_id)
        respond(str(exc))
        st.rerun()
    except GuardrailError:
        conversation.invalidate(state, session_id)
        st.warning(
            "Draft ditolak oleh pemeriksaan keamanan. Ubah informasi dan hindari data pribadi."
        )
    except (OSError, sqlite3.Error, ValueError, ImportError):
        st.error(
            "Pencarian atau tindakan gagal. Tidak ada konfirmasi keberhasilan. "
            "Periksa ketersediaan database, audit, dan embedding sebelum mencoba kembali."
        )
