"""Executive overview page."""

import streamlit as st

from bnivoice360.container import AppServices
from bnivoice360.services import analytics
from bnivoice360.ui import charts
from bnivoice360.ui.components import header


def render(services: AppServices) -> None:
    header(
        "Executive Overview",
        "Satu pandangan terpadu atas suara nasabah, sentimen, dan prioritas eskalasi.",
        "BNIVoice360 · Command Center",
    )
    frame = services.repository.load()
    metrics = analytics.overview_metrics(frame)
    columns = st.columns(4)
    columns[0].metric(
        "Total VoC Volume",
        f"{metrics.total_volume:,}",
        f"{metrics.growth_percent:+.1f}% · 30d change",
    )
    columns[1].metric("Net Sentiment Score", f"{metrics.net_sentiment:+.1f}", "Positive − Negative")
    columns[2].metric("Urgent Open", metrics.urgent_open, "Perlu intervensi", delta_color="inverse")
    columns[3].metric("Topic Clusters", metrics.topic_count, "Tema terdeteksi")
    if frame.empty:
        st.info("Belum ada data VoC. Data Admin dapat mengimpor data di Data Ingestion.")
        return

    left, right = st.columns([1, 1.3])
    with left:
        charts.render_chart(charts.channel_mix(analytics.channel_volume(frame)), "home_channels")
    trend = analytics.sentiment_trend(frame)
    if trend.empty:
        right.info("Jalankan pipeline NLP untuk melihat tren sentimen.")
    else:
        with right:
            charts.render_chart(charts.sentiment_signal(trend), "home_sentiment")
    topics = analytics.top_topics(frame)
    if topics.empty:
        st.info("Belum ada topic cluster yang terdeteksi.")
    else:
        charts.render_chart(charts.top_issue_clusters(topics), "home_topics")
