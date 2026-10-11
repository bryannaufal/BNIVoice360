"""Streamlit bootstrap and explicit page registration."""

import sqlite3

import streamlit as st

from bnivoice360.container import create_services
from bnivoice360.pages import companion, graph, home, ingestion, issues
from bnivoice360.ui.sidebar import render_sidebar
from bnivoice360.ui.state import initialize_state
from bnivoice360.ui.theme import inject_css

PAGES = {
    "Home": home.render,
    "Data Ingestion": ingestion.render,
    "Issue Analytics & Voice Feed": issues.render,
    "Graph Analytics": graph.render,
    "BNIVoice360 Companion": companion.render,
}


def main() -> None:
    st.set_page_config(
        page_title="BNIVoice360",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="auto",
    )
    initialize_state()
    inject_css()
    try:
        services = create_services()
    except (sqlite3.Error, OSError, ValueError) as exc:
        st.error(f"Database tidak dapat diinisialisasi: {exc}")
        st.stop()
    page = render_sidebar(services.repository, list(PAGES))
    PAGES[page](services)
