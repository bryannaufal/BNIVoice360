"""Negative issue monitoring, topic diagnostics, and customer action center."""

import json
from html import escape

import streamlit as st

from bnivoice360.container import AppServices
from bnivoice360.services import analytics
from bnivoice360.services import issues as diagnostics
from bnivoice360.ui import charts
from bnivoice360.ui.batch_agent import render_batch_agent
from bnivoice360.ui.components import header, section_title, urgent_alert
from bnivoice360.ui.filters import render_issue_filters
from bnivoice360.ui.state import show_notice
from bnivoice360.ui.voice_feed import render_voice_feed

ALL_TOPICS = "all"


def _remember_topic() -> None:
    st.session_state["issue_topic_id"] = st.session_state["issues_deep_dive_topic"]


def _text_metric(column, label: str, value: str, key: str) -> None:
    with column, st.container(key=key):
        st.metric(label, value, help=value)


def _select_topic(catalog, event):
    valid = set(catalog["topic_id"])
    selected = st.session_state.get("issue_topic_id", int(catalog.iloc[0]["topic_id"]))
    if selected is None:
        selected = ALL_TOPICS
    if selected != ALL_TOPICS and selected not in valid:
        selected = int(catalog.iloc[0]["topic_id"])
    signature = json.dumps(event.get("selection", {}), sort_keys=True, default=str)
    if signature != st.session_state.get("issue_treemap_selection"):
        st.session_state["issue_treemap_selection"] = signature
        clicked = diagnostics.selected_topic(event, catalog)
        if clicked is not None:
            selected = clicked
    st.session_state["issue_topic_id"] = selected
    # Update before instantiating the selectbox; chart selection never overwrites it later.
    st.session_state["issues_deep_dive_topic"] = selected
    labels = catalog.set_index("topic_id")["Label"].to_dict()
    chosen = st.selectbox(
        "🔍 Pilih Topik untuk Deep-Dive",
        [*catalog["topic_id"].astype(int), ALL_TOPICS],
        format_func=lambda topic_id: (
            labels[topic_id] if topic_id != ALL_TOPICS else "Semua isu (tanpa deep-dive)"
        ),
        key="issues_deep_dive_topic",
        on_change=_remember_topic,
    )
    return None if chosen == ALL_TOPICS else chosen


def render(services: AppServices) -> None:
    header(
        "Issue Analytics & Voice Feed",
        "Pantau isu negatif, telusuri pemicu utama, dan tindak lanjuti suara nasabah.",
        "VoC Intelligence",
    )
    show_notice("escalation_notice")
    frame = services.repository.load()
    if frame.empty or frame["date_posted"].dropna().empty:
        st.info("Belum ada data bertanggal valid untuk dianalisis.")
        return
    filters = render_issue_filters(frame)
    filtered = diagnostics.negative_issues(analytics.filter_issues(frame, filters))
    urgent_alert(analytics.open_urgent_count(filtered))
    st.caption(
        "Halaman ini berfokus pada sentimen Negative. Topik Outlier dan Unlabeled tidak masuk peringkat atau grafik topik."
    )
    metrics = diagnostics.issue_metrics(filtered)
    columns = st.columns(4)
    columns[0].metric(
        "Total Issue Voices",
        f"{metrics['total']:,}",
        help="Suara negatif dengan topik teridentifikasi.",
    )
    columns[1].metric(
        "Pending Triage / Escalations",
        metrics["pending"],
        help="Isu negatif urgent berstatus Open.",
    )
    columns[2].metric(
        "Active Escalations", metrics["active"], help="Isu negatif urgent berstatus Escalated."
    )
    _text_metric(columns[3], "Top Issue Driver", metrics["driver"], "issue_driver_metric")
    counts = analytics.topic_assignment_counts(filtered)
    st.caption(
        f"{counts['outlier']:,} Outlier · {counts['unlabeled']:,} Unlabeled dalam filter aktif. Pilih Semua isu untuk melihatnya di action center."
    )
    if filtered.empty:
        st.info("Tidak ada isu negatif yang sesuai dengan kombinasi filter.")
        render_voice_feed(filtered, services.repository, filter_context=repr((filters, None)))
        return

    catalog = diagnostics.topic_catalog(filtered)
    binning = diagnostics.time_binning(filters.start_date, filters.end_date)
    binned = diagnostics.binned_topic_volumes(
        filtered, filters.start_date, filters.end_date, binning
    )
    tiles = diagnostics.category_topic_tiles(filtered)
    section_title("Global issue analytics")
    st.caption(
        f"{binning} bins · 4 topik berdasarkan total volume + 2 berdasarkan puncak volume per interval. Grafik tren tidak menggabungkan topik lainnya."
    )
    left, right = st.columns(2)
    event = {}
    with left:
        if tiles.empty:
            st.info("Belum ada kategori untuk topik teridentifikasi dalam filter ini.")
        else:
            event = st.plotly_chart(
                charts.issue_topic_treemap(tiles),
                on_select="rerun",
                selection_mode="points",
                key="treemap_issues",
                width="stretch",
                theme=None,
                config={"responsive": True, "displaylogo": False},
            )
        st.caption(
            "Top 5 topik per unit. Topik lainnya tersedia di dropdown deep-dive dan Voice Feed. "
            "Satu voice dapat terkait beberapa unit. Klik topik untuk deep-dive."
        )
    with right:
        trend = diagnostics.spike_aware_topics(binned)
        if trend.empty:
            st.info("Belum ada topik teridentifikasi untuk grafik tren.")
        else:
            charts.render_chart(
                charts.topic_trend(trend, "Global topic trend", binning), "issues_global_trend"
            )

    topic_id = None
    topic_filtered = filtered
    if catalog.empty:
        st.session_state.pop("issue_topic_id", None)
        st.info(
            "Belum ada topik teridentifikasi untuk deep-dive. Semua isu negatif tersedia di action center."
        )
    else:
        topic_id = _select_topic(catalog, event)
        if topic_id is not None:
            topic_filtered = filtered.loc[filtered["topic_id"].eq(topic_id)]
            label = catalog.set_index("topic_id").loc[topic_id, "Label"]
            st.markdown(f"### 🔎 Deep-Dive Analisis: {escape(label)}")
            summary = diagnostics.topic_summary(topic_filtered)
            columns = st.columns(4)
            columns[0].metric("Volume Suara Isu", summary["volume"])
            _text_metric(columns[1], "Status Eskalasi", summary["statuses"], "issue_status_metric")
            _text_metric(columns[2], "Kanal Utama", summary["channel"], "issue_channel_metric")
            _text_metric(columns[3], "Cakupan Unit", summary["coverage"], "issue_coverage_metric")
            left, right = st.columns(2)
            with left:
                charts.render_chart(
                    charts.topic_trend(
                        binned.loc[binned["topic_id"].eq(topic_id)], "Isolated topic trend", binning
                    ),
                    "issues_topic_trend",
                )
            with right:
                charts.render_chart(
                    charts.issue_channel_breakdown(analytics.channel_volume(topic_filtered)),
                    "issues_topic_channels",
                )
                charts.render_chart(
                    charts.department_overlap(diagnostics.department_involvement(topic_filtered)),
                    "issues_topic_units",
                )
                st.caption("Unit dapat tumpang tindih; total persentase dapat melebihi 100%.")
    render_voice_feed(topic_filtered, services.repository, filter_context=repr((filters, topic_id)))
    render_batch_agent(services.batch_agent, filters, topic_id)
