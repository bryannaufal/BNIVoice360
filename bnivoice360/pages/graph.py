"""Complaint relationship exploration with explicit reviewed handoff to Companion."""

import json
import sqlite3
import uuid
from datetime import timedelta

import streamlit as st

from bnivoice360.config import CATEGORY_COLUMNS
from bnivoice360.container import AppServices
from bnivoice360.domain import IssueFilters
from bnivoice360.services.agent.audit import fingerprint
from bnivoice360.services.agent.workflow import ReviewRequired
from bnivoice360.services.analytics import category_labels
from bnivoice360.services.companion_state import CompanionState, ComplaintQuery
from bnivoice360.services.graph_analytics import bind_snapshots, build_network, explain_group
from bnivoice360.ui.components import header
from bnivoice360.ui.credentials import groq_key
from bnivoice360.ui.graph import network_figure, selected_ticket
from bnivoice360.ui.voice_feed import render_voice_feed


def _handoff(services, network, ids, filters):
    """Navigation callbacks run before the sidebar widget is instantiated."""
    state = st.session_state.setdefault("companion_context", CompanionState())
    session_id = st.session_state.setdefault("companion_session_id", str(uuid.uuid4()))
    query = ComplaintQuery(
        "",
        start=filters.start_date,
        end=filters.end_date,
        sources=filters.sources,
        status="Open",
        categories=filters.categories,
        urgent_only=filters.urgent_only,
    )
    try:
        reply = services.conversation.select_from_graph(
            state,
            ids,
            network.snapshots,
            query,
            session_id,
            st.session_state.selected_role,
        )
    except (ReviewRequired, OSError, sqlite3.Error) as exc:
        st.session_state["graph_action_error"] = (
            str(exc)
            if isinstance(exc, ReviewRequired)
            else "Pemilihan gagal. Periksa database dan audit."
        )
        return
    history = st.session_state.setdefault("chat_history", [])
    history.append({"role": "assistant", "content": reply})
    if len(history) > 41:
        st.session_state["chat_history"] = [history[0], *history[-40:]]
    st.session_state["current_page"] = "BNIVoice360 Companion"


def _filters(frame, settings, today):
    negative = frame.loc[frame["sentiment"].eq("Negative")]
    catalog = negative.loc[negative["topic_id"].notna() & negative["topic_id"].ne(-1)]
    counts = catalog.groupby("topic_id").size().sort_values(ascending=False)
    labels = catalog.drop_duplicates("topic_id").set_index("topic_id")["topic_cluster"].to_dict()
    topic_options = [*counts.index.astype(int), "all"]
    # Clamp obsolete widget choices after a database replacement or removal of a topic.
    if st.session_state.get("graph_topic") not in topic_options:
        st.session_state["graph_topic"] = topic_options[0]
    dates = frame["date_posted"].dropna()
    latest = dates.max().date() if not dates.empty else today
    minimum = dates.min().date() if not dates.empty else today
    channels = sorted(frame["source"].unique())
    if "graph_sources" in st.session_state:
        retained = [v for v in st.session_state["graph_sources"] if v in channels]
        if retained != st.session_state["graph_sources"]:
            st.session_state["graph_sources"] = retained
    if "graph_dates" in st.session_state:
        clamped = tuple(
            max(minimum, min(v, max(today, latest))) for v in st.session_state["graph_dates"]
        )
        if clamped != st.session_state["graph_dates"]:
            st.session_state["graph_dates"] = clamped
    with st.form("graph_filters"):
        left, right = st.columns(2)
        topic_id = left.selectbox(
            "Topik",
            topic_options,
            format_func=lambda value: (
                "Semua topik (lintas topik)" if value == "all" else str(labels[value])
            ),
            key="graph_topic",
        )
        selected_dates = right.date_input(
            "Periode",
            (max(minimum, latest - timedelta(days=29)), latest),
            min_value=minimum,
            max_value=max(today, latest),
            key="graph_dates",
        )
        left, right = st.columns(2)
        all_time = left.checkbox(
            "Semua periode (termasuk tanggal kosong)", value=dates.empty, key="graph_all_time"
        )
        cross_topic = right.checkbox(
            "Izinkan lintas topik, termasuk Outlier/Unlabeled",
            key="graph_cross_topic",
            help="Pilih Semua topik untuk mencari hubungan lintas topik. Pilihan satu topik tetap membatasi jaringan pada topik tersebut.",
        )
        left, middle, right = st.columns(3)
        sources = left.multiselect("Kanal", channels, default=channels, key="graph_sources")
        statuses = middle.multiselect(
            "Status",
            ["Open", "Escalated", "Resolved"],
            default=["Open", "Escalated"],
            key="graph_statuses",
        )
        categories = right.multiselect(
            "Unit terkait",
            list(CATEGORY_COLUMNS),
            default=list(CATEGORY_COLUMNS),
            key="graph_categories",
        )
        urgent = st.checkbox("Urgent saja (label database)", key="graph_urgent")
        left, middle, right = st.columns(3)
        threshold = left.slider(
            "Ambang kemiripan",
            0.50,
            1.0,
            min(1.0, max(0.5, settings.agent_similarity_threshold)),
            0.01,
            key="graph_threshold",
        )
        max_nodes = middle.slider("Maksimal keluhan", 20, 200, 120, 20, key="graph_limit")
        neighbors = right.slider("Tetangga terdekat per keluhan", 1, 10, 5, key="graph_neighbors")
        st.caption(
            "Jika dibatasi: urgent Open didahulukan, lalu Open, Escalated, Resolved; terbaru pada setiap kelompok. Hubungan adalah gabungan tetangga terdekat, bukan semua pasangan."
        )
        submitted = st.form_submit_button(
            "Bangun / perbarui jaringan", type="primary", key="graph_build"
        )
    filters = IssueFilters(
        start_date=None if all_time or not selected_dates else selected_dates[0],
        end_date=None if all_time or not selected_dates else selected_dates[-1],
        sources=tuple(sources),
        sentiments=("Negative",),
        categories=tuple(categories),
        urgent_only=urgent,
    )
    return (
        submitted,
        all_time or len(selected_dates) == 2,
        labels,
        filters,
        dict(
            topic_id=None if topic_id == "all" else topic_id,
            cross_topic=cross_topic,
            statuses=tuple(statuses),
            threshold=threshold,
            max_nodes=max_nodes,
            neighbors=neighbors,
            as_of=today,
        ),
    )


def _evidence(network, visible, edges, revision, chosen):
    left, right = st.columns([3, 2])
    evidence_key = "graph_evidence_" + revision + chosen
    with left:
        event = st.plotly_chart(
            network_figure(visible, edges),
            width="stretch",
            theme=None,
            key="graph_chart_" + revision + chosen,
            on_select="rerun",
            selection_mode="points",
            config={"displaylogo": False, "scrollZoom": False},
        )
        st.caption(
            "Warna isi: kanal · Garis tepi: Open oranye, Escalated biru, Resolved hijau · Node besar: urgent. Klik node untuk membaca bukti; klik tidak memilih tiket untuk eskalasi."
        )
    with right:
        signature = json.dumps(event.get("selection", {}), sort_keys=True, default=str)
        if signature != st.session_state.get(evidence_key + "_event"):
            st.session_state[evidence_key + "_event"] = signature
            clicked = selected_ticket(event, set(visible["suara_nasabah_id"]))
            if clicked:
                st.session_state[evidence_key] = clicked
        ticket = st.selectbox(
            "Keluhan untuk dibaca", list(visible["suara_nasabah_id"]), key=evidence_key
        )
        row = visible.set_index("suara_nasabah_id").loc[ticket]
        st.caption(
            f"{row['source']} · {row['topic_cluster']} · {row['escalation_status']} · {row['date_posted']}"
        )
        st.text(str(row["content"]))
        st.caption("Unit terkait: " + (", ".join(category_labels(row)) or "Belum dilabeli"))
        related = edges.loc[edges["source_id"].eq(ticket) | edges["target_id"].eq(ticket)].copy()
        if not related.empty:
            st.dataframe(
                related,
                hide_index=True,
                width="stretch",
                column_config={
                    "similarity": st.column_config.NumberColumn("Cosine similarity", format="%.4f")
                },
            )
        else:
            st.caption("Tidak ada hubungan langsung pada ambang dan batas tetangga ini.")


def _explanation(services, network, revision, chosen):
    if chosen == "all":
        return
    if st.button("Jelaskan kelompok", key="graph_explain"):
        st.session_state.pop("graph_explanation", None)
        try:
            session_id = st.session_state.setdefault("companion_session_id", str(uuid.uuid4()))
            with st.spinner("Menjelaskan bukti kelompok..."):
                answer = explain_group(
                    services.batch_agent,
                    network,
                    chosen,
                    session_id,
                    st.session_state.selected_role,
                    groq_key() if services.batch_agent.settings.companion_remote_enabled else "",
                )
            st.session_state["graph_explanation"] = (revision, chosen, answer)
        except (ReviewRequired, OSError, ValueError, sqlite3.Error) as exc:
            st.warning(
                str(exc)
                if isinstance(exc, ReviewRequired)
                else "Penjelasan gagal. Periksa database dan audit."
            )
    explained = st.session_state.get("graph_explanation")
    if explained and explained[:2] == (revision, chosen):
        answer, reports, mode = explained[2]
        st.caption(
            "Penjelasan AI tervalidasi"
            if mode == "groq"
            else "Ringkasan lokal; penjelasan AI tidak aktif atau tidak lolos validasi."
        )
        st.text(answer)
        with st.expander("Sumber penjelasan"):
            for report in reports:
                st.write(f"[{report['id']}] {report['tool']}")
                st.json(report["data"], expanded=False)


def _selection(services, network, visible, filters, revision, chosen):
    st.markdown("**Pilih tiket Open untuk Companion**")
    open_rows = visible.loc[visible["escalation_status"].eq("Open")]
    if open_rows.empty:
        st.info(
            "Tidak ada tiket Open dalam pilihan ini. Keluhan lain tetap tersedia sebagai bukti."
        )
        return
    selection = open_rows[
        ["suara_nasabah_id", "source", "topic_cluster", "date_posted", "is_Urgent", "content"]
    ].copy()
    selection.insert(0, "Pilih", False)
    selected = st.data_editor(
        selection,
        hide_index=True,
        width="stretch",
        key="graph_pick_" + revision + chosen,
        disabled=list(selection.columns[1:]),
        column_config={
            "Pilih": st.column_config.CheckboxColumn("Pilih", default=False),
            "content": st.column_config.TextColumn("Keluhan", width="large"),
        },
    )
    # Use server-owned IDs, regardless of any client edits to disabled columns.
    ids = tuple(selection.loc[selected["Pilih"].to_numpy(dtype=bool), "suara_nasabah_id"])
    st.caption(
        f"{len(ids)} tiket dipilih · Maksimal 51. Membuka Companion membatalkan draft/approval Companion sebelumnya; belum mengubah status."
    )
    st.button(
        "Lanjutkan ke Companion",
        key="graph_to_companion",
        type="primary",
        disabled=not 1 <= len(ids) <= 51,
        on_click=_handoff,
        args=(services, network, ids, filters),
    )


def render(services: AppServices) -> None:
    header(
        "Graph Analytics",
        "Temukan pola keluhan terkait lintas kanal dan unit untuk investigasi bersama.",
        "Related Complaint Network",
    )
    st.caption(
        "Hubungan berarti kemiripan teks, bukan akar penyebab, duplikasi, atau insiden terkonfirmasi. Kelompok dapat terhubung melalui rantai; tidak semua pasangan memiliki hubungan langsung."
    )
    frame = services.repository.load()
    if frame.empty:
        st.info("Belum ada keluhan untuk dianalisis.")
        return
    submitted, valid_dates, labels, filters, options = _filters(
        frame, services.batch_agent.settings, services.conversation.today()
    )
    if submitted:
        st.session_state.pop("graph_result", None)
        st.session_state.pop("graph_explanation", None)
        if not valid_dates or not options["statuses"]:
            st.warning("Pilih dua tanggal dan sedikitnya satu status.")
        else:
            try:
                with st.spinner("Membangun hubungan dari embedding keluhan..."):
                    network = build_network(
                        frame,
                        services.batch_agent.settings.embedding_directory
                        / "data_id_embedding_mapping.parquet",
                        filters,
                        **options,
                    )
                    bind_snapshots(services.repository, network)
                st.session_state["graph_result"] = (
                    network,
                    filters,
                    options["topic_id"],
                    options["statuses"],
                )
            except ReviewRequired as exc:
                st.warning(str(exc))
            except FileNotFoundError:
                st.warning(
                    "Embedding keluhan belum tersedia. Jaringan tidak dapat dihitung; ini bukan hasil nol hubungan. Periksa unduhan embedding lalu coba kembali."
                )
            except (OSError, ValueError, ImportError, sqlite3.Error) as exc:
                st.warning(
                    str(exc)
                    if isinstance(exc, ValueError)
                    else "Jaringan gagal dibangun. Periksa database dan file embedding."
                )
    if error := st.session_state.pop("graph_action_error", None):
        st.warning(error)
    result = st.session_state.get("graph_result")
    if result is None:
        st.info(
            "Pilih cakupan lalu klik Bangun / perbarui jaringan. Tidak ada tiket yang dipilih otomatis."
        )
        return
    network, filters, built_topic, built_statuses = result
    period = (
        f"{filters.start_date} sampai {filters.end_date}"
        if filters.start_date
        else "Semua periode, termasuk tanggal kosong"
    )
    st.caption(
        f"Snapshot: {network.as_of} WIB · {period} · Topik: {labels.get(built_topic, 'Semua topik')} "
        f"· Status: {', '.join(built_statuses)} · Kanal: {', '.join(filters.sources) or 'Tidak ada'} "
        f"· Unit: {', '.join(filters.categories) or 'Tidak ada'} · Urgent saja: {'Ya' if filters.urgent_only else 'Tidak'} "
        f"· Ambang: {network.threshold:.2f} · Tetangga: {network.neighbors}. Perubahan filter berlaku setelah membangun ulang."
    )
    metrics = st.columns(4)
    metrics[0].metric("Keluhan dalam cakupan", network.scope_count)
    metrics[1].metric("Keluhan dalam jaringan", len(network.nodes))
    metrics[2].metric("Kelompok terkait", int(network.groups["count"].gt(1).sum()))
    metrics[3].metric("Hubungan langsung", len(network.edges))
    st.caption(
        f"{network.missing_embeddings} keluhan tanpa embedding · {network.omitted_by_limit} keluhan berembedding di luar batas · {int(network.nodes['degree'].eq(0).sum())} keluhan tanpa hubungan dalam jaringan ini. Angka kelompok hanya berlaku untuk keluhan yang ditampilkan."
    )
    if network.nodes.empty:
        st.info(
            "Tidak ada keluhan berembedding dalam cakupan ini. Ubah filter atau periksa cakupan embedding."
        )
        return
    revision = fingerprint(
        {
            "snapshots": network.snapshots,
            "threshold": network.threshold,
            "neighbors": network.neighbors,
        }
    )[:16]
    groups = network.groups
    group_labels = {
        row.group_id: f"{'Kelompok' if row.count > 1 else 'Tanpa hubungan'} {i + 1} · {row.count} keluhan · {row.urgent_open} urgent Open · {row.channels} kanal"
        for i, row in enumerate(groups.itertuples(index=False))
    }
    group_table = groups.drop(columns="group_id").copy()
    group_table.insert(0, "Kelompok", [i + 1 for i in range(len(groups))])
    with st.expander("Peringkat kelompok dan cakupan unit", expanded=True):
        st.caption(
            "Urutan: urgent Open terbanyak, Open terbanyak, lalu tanggal Open tertua. Umur dihitung sejak posting, bukan waktu dalam status atau SLA."
        )
        st.dataframe(
            group_table.rename(
                columns={
                    "count": "Keluhan",
                    "open": "Open",
                    "escalated": "Escalated",
                    "resolved": "Resolved",
                    "urgent_open": "Urgent Open",
                    "channels": "Kanal",
                    "units": "Unit terkait",
                    "oldest_open_date": "Posting Open tertua",
                    "oldest_open_days": "Umur posting (hari)",
                    "edges": "Hubungan",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    chosen = st.selectbox(
        "Kelompok untuk investigasi",
        ["all", *group_labels],
        format_func=lambda value: "Seluruh jaringan" if value == "all" else group_labels[value],
        key="graph_group_" + revision,
    )
    visible = network.nodes if chosen == "all" else network.members(chosen)
    edges = network.edges.loc[
        network.edges["source_id"].isin(visible["suara_nasabah_id"])
        & network.edges["target_id"].isin(visible["suara_nasabah_id"])
    ]
    _evidence(network, visible, edges, revision, chosen)
    _explanation(services, network, revision, chosen)
    _selection(services, network, visible, filters, revision, chosen)
    with st.expander("Voice Feed kelompok — status terkini"):
        live = services.repository.load()
        live = live.loc[live["suara_nasabah_id"].isin(visible["suara_nasabah_id"])]
        render_voice_feed(live, services.repository, filter_context="graph:" + revision + chosen)
        st.caption(
            "Perubahan status di feed memerlukan jaringan baru sebelum tiket dapat diteruskan ke Companion."
        )
