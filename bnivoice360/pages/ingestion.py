"""Admin source workspace. Source widgets live in reusable panels."""

import streamlit as st

from bnivoice360.config import SOURCES
from bnivoice360.container import AppServices
from bnivoice360.domain import Role
from bnivoice360.ui.components import header
from bnivoice360.ui.ingestion import render_crawl_panel, render_data_panel, render_upload_panel
from bnivoice360.ui.state import show_notice


def render(services: AppServices) -> None:
    if st.session_state.get("selected_role") != Role.DATA_ADMIN.value:
        st.error("Data Ingestion tersedia untuk Data Admin.")
        return
    header(
        "Data Ingestion",
        "Kelola akuisisi, kualitas, dan enrichment data lintas kanal.",
        "Admin Workspace",
    )
    show_notice("ingestion_notice")
    errors = st.session_state.pop("ingestion_errors", ())
    if errors:
        st.warning(f"{len(errors)} record gagal validasi.")
        with st.expander("Validation details"):
            for error in errors:
                st.write(error)
    # A single snapshot per render, rather than one query for every tab.
    frame = services.repository.load()
    for source, source_tab in zip(SOURCES, st.tabs(SOURCES), strict=True):
        with source_tab:
            see, crawl, upload = st.tabs(["See Data", "Crawl Data", "Upload Data by .csv"])
            with see:
                render_data_panel(source, frame[frame["source"] == source], services.prediction)
            with crawl:
                render_crawl_panel(source, services.crawler, services.ingestion)
            with upload:
                render_upload_panel(source, services.ingestion)
