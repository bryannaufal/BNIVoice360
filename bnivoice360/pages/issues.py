"""Issue discovery and escalation workspace."""

import streamlit as st

from bnivoice360.container import AppServices
from bnivoice360.services import analytics
from bnivoice360.ui import charts
from bnivoice360.ui.components import header, section_title, urgent_alert
from bnivoice360.ui.filters import render_issue_filters
from bnivoice360.ui.state import show_notice
from bnivoice360.ui.voice_feed import render_voice_feed


def render(services: AppServices) -> None:
    header(
        "Issue Analytics & Voice Feed",
        "Temukan isu, pahami akar sentimen, dan tindak lanjuti prioritas nasabah.",
        "VoC Intelligence",
    )
    show_notice("escalation_notice")
    frame = services.repository.load()
    if frame.empty or frame["date_posted"].dropna().empty:
        st.info("Belum ada data bertanggal valid untuk dianalisis.")
        return
    filters = render_issue_filters(frame)
    filtered = analytics.filter_issues(frame, filters)
    urgent_alert(analytics.pending_urgent_count(filtered))
    section_title("Advanced issue analytics")
    if filtered.empty:
        st.info("Tidak ada data yang sesuai dengan kombinasi filter.")
        render_voice_feed(filtered, services.repository, filter_context=repr(filters))
        return

    left, right = st.columns(2)
    sentiment_figure = charts.topic_sentiment_distribution(analytics.topic_sentiments(filtered))
    hierarchy = analytics.category_topics(filtered)
    if not hierarchy.empty:
        with left:
            category_figure = charts.category_topic_map(hierarchy)
            category_figure.update_layout(height=sentiment_figure.layout.height)
            charts.render_chart(category_figure, "chart_issues_categories")
    with right:
        charts.render_chart(
            sentiment_figure,
            "chart_issues_sentiments",
        )
    render_voice_feed(filtered, services.repository, filter_context=repr(filters))
