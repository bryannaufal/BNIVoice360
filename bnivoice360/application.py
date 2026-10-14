"""Streamlit bootstrap and explicit page registration."""

import sqlite3

import streamlit as st

from bnivoice360.config import Settings
from bnivoice360.container import create_services
from bnivoice360.data.embeddings import ensure_embeddings
from bnivoice360.pages import (
    companion,
    companion_evaluation,
    home,
    ingestion,
    issues,
    source_routing_graph,
)
from bnivoice360.ui.sidebar import render_sidebar
from bnivoice360.ui.state import initialize_state
from bnivoice360.ui.theme import inject_css

PAGES = {
    "Home": home.render,
    "Data Ingestion": ingestion.render,
    "Issue Analytics & Voice Feed": issues.render,
    "Graph Analytics": source_routing_graph.render,
    "BNIVoice360 Companion": companion.render,
    "Companion Evaluation": companion_evaluation.render,
}


@st.cache_resource(show_spinner="Downloading embedding files for first use...")
def initialize_embeddings(settings: Settings) -> tuple:
    return ensure_embeddings(settings)


def main() -> None:
    st.set_page_config(
        page_title="BNIVoice360",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="auto",
    )
    initialize_state()
    inject_css()
    settings = Settings()
    try:
        initialize_embeddings(settings)
    except (OSError, ValueError) as exc:
        st.warning(f"Embedding files could not be downloaded; startup will retry: {exc}")
    try:
        services = create_services(settings)
    except (sqlite3.Error, OSError, ValueError) as exc:
        st.error(f"Database tidak dapat diinisialisasi: {exc}")
        st.stop()
    page = render_sidebar(services.repository, list(PAGES))
    PAGES[page](services)
