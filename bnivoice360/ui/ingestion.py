"""Source-specific ingestion panels without persistence or prediction logic."""

from datetime import date, timedelta
from io import BytesIO

import pandas as pd
import streamlit as st

from bnivoice360.domain import CrawlRequest
from bnivoice360.services.crawler import Crawler
from bnivoice360.services.ingestion import IngestionService
from bnivoice360.services.prediction import PredictionService
from bnivoice360.ui.state import notify_and_rerun


def render_data_panel(source: str, frame: pd.DataFrame, prediction: PredictionService) -> None:
    records, labeled, unlabeled = st.columns(3)
    records.metric("Records", len(frame))
    percentage = frame["sentiment"].notna().mean() * 100 if len(frame) else 0
    labeled.metric("Labeled", f"{percentage:.0f}%")
    unlabeled.metric("Unlabeled", int(frame["sentiment"].isna().sum()))
    st.dataframe(frame, use_container_width=True, hide_index=True, height=310)
    if st.button("Run NLP Model & Topic Modelling", type="primary", key=f"predict_{source}"):
        count = prediction.run(source)
        notify_and_rerun("ingestion_notice", f"Pipeline selesai: {count} record diprediksi.")


def _date_range(source: str) -> tuple[date, date] | None:
    selected = st.date_input(
        "Date range",
        (date.today() - timedelta(days=7), date.today()),
        key=f"crawl_dates_{source}",
    )
    if len(selected) != 2:
        st.caption("Pilih tanggal awal dan tanggal akhir.")
        return None
    return selected[0], selected[1]


def _crawl_request(source: str) -> CrawlRequest | None:
    if source in ("PlayStore", "AppStore"):
        app_id = st.text_input(
            "App package / App ID",
            "id.bni.wondr" if source == "PlayStore" else "com.bni.mobile",
            key=f"app_{source}",
        )
        rating = st.slider("Minimum star rating", 1, 5, 1, key=f"rating_{source}")
        limit = st.number_input("Max review limit", 1, 100, 10, key=f"limit_{source}")
        return CrawlRequest(source, int(limit), app_id=app_id, minimum_rating=rating)

    if source == "Laporan Nasabah":
        branch = st.text_input("Branch / service code", "KCU-JKT-001", key=f"branch_{source}")
        category = st.selectbox(
            "Ticket category",
            ["All", "Complaint", "Inquiry", "Request"],
            key=f"ticket_{source}",
        )
        dates = _date_range(source)
        limit = st.number_input("Max ticket limit", 1, 100, 10, key=f"limit_{source}")
        if dates:
            return CrawlRequest(
                source,
                int(limit),
                start_date=dates[0],
                end_date=dates[1],
                branch_code=branch,
                ticket_category=category,
            )
        return None

    keywords = st.text_input("Keywords / hashtags", "BNI, wondr, #BNI", key=f"kw_{source}")
    dates = _date_range(source)
    limit = st.number_input("Max post limit", 1, 100, 10, key=f"limit_{source}")
    if dates:
        return CrawlRequest(
            source, int(limit), keywords=keywords, start_date=dates[0], end_date=dates[1]
        )
    return None


def _store_batch(
    frame: pd.DataFrame,
    source: str,
    ingestion: IngestionService,
    clear_state_key: str | None = None,
) -> None:
    try:
        result = ingestion.ingest(frame, forced_source=source)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state["ingestion_errors"] = result.errors
    if clear_state_key:
        st.session_state.pop(clear_state_key, None)
    notify_and_rerun(
        "ingestion_notice",
        f"{result.inserted} record tersimpan. {result.skipped} dilewati/duplikat.",
    )


def render_crawl_panel(source: str, crawler: Crawler, ingestion: IngestionService) -> None:
    st.caption(
        "Simulator pengumpulan data untuk prototipe lokal — "
        "hubungkan adaptor API resmi untuk produksi."
    )
    request = _crawl_request(source)
    state_key = f"crawl_data_{source}"
    if st.button("Start Crawling", type="primary", key=f"crawl_{source}", disabled=request is None):
        try:
            st.session_state[state_key] = crawler.crawl(request)
        except ValueError as exc:
            st.error(str(exc))
    preview = st.session_state.get(state_key)
    if preview is not None:
        st.success(f"{len(preview)} record berhasil dikumpulkan dari simulator.")
        st.dataframe(preview, use_container_width=True, hide_index=True)
        if st.button("Save Crawled Data to SQLite", key=f"save_crawl_{source}"):
            _store_batch(preview, source, ingestion, clear_state_key=state_key)


def render_upload_panel(source: str, ingestion: IngestionService) -> None:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"upload_{source}")
    st.caption(
        "Kolom wajib: content dan date_posted. "
        "Kolom source opsional karena akan dipetakan ke tab aktif."
    )
    if uploaded is None:
        return
    try:
        # getvalue avoids depending on the stream cursor across reruns.
        frame = pd.read_csv(BytesIO(uploaded.getvalue()))
        ingestion.validate_columns(frame, forced_source=source)
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        st.error(f"CSV tidak dapat diproses: {exc}")
        return
    st.success("Validasi struktur berhasil.")
    st.dataframe(frame.head(100), use_container_width=True, hide_index=True)
    if st.button("Ingest Batch to SQLite", type="primary", key=f"ingest_{source}"):
        _store_batch(frame, source, ingestion)
