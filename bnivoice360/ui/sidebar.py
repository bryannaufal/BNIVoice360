"""Sidebar navigation and a lightweight database health display."""

import base64
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st

from bnivoice360.data.repository import VoiceRepository
from bnivoice360.domain import Role

LOGO_PATHS = (
    Path(r"D:\BNI\Bootcamp\CAPSTPONE\BNIVoice360.png"),
    Path(r"D:\BNI\Bootcamp\CAPSTONE\BNIVoice360.png"),
    Path(__file__).resolve().parent / "assets" / "BNIVoice360.png",
)


def _get_logo_path() -> Path:
    for candidate in LOGO_PATHS:
        if candidate.is_file():
            return candidate
    return LOGO_PATHS[-1]


def render_sidebar(repository: VoiceRepository, page_titles: list[str]) -> str:
    with st.sidebar:
        logo = _get_logo_path()
        if logo.is_file():
            encoded = base64.b64encode(logo.read_bytes()).decode()
            st.markdown(
                f'<div style="display: flex; justify-content: center; align-items: center; margin: 0.25rem 0 1.25rem 0;">'
                f'<img src="data:image/png;base64,{encoded}" alt="BNIVoice360" style="width: 175px; max-width: 80%; height: auto; display: block;" />'
                f"</div>",
                unsafe_allow_html=True,
            )
        st.selectbox("Active role", [role.value for role in Role], key="selected_role")
        st.markdown("---")
        # Role changes rebuild the page list in the same run.
        titles = [
            title
            for title in page_titles
            if title not in {"Data Ingestion", "Companion Evaluation"}
            or st.session_state.selected_role == Role.DATA_ADMIN.value
        ]
        if st.session_state.get("current_page") not in titles:
            st.session_state.current_page = "Home"
        page = st.radio("Navigation", titles, key="current_page", label_visibility="collapsed")
        st.markdown("---")
        try:
            health = repository.health()
            st.markdown("**Database status**")
            st.success("● SQLite connected")
            st.caption(
                f"{health['records']:,} records · {health['size_kb']:.1f} KB\n\n"
                f"Last checked: {datetime.now():%H:%M:%S}"
            )
        except (sqlite3.Error, OSError) as exc:
            st.error(f"Database error: {exc}")
        st.caption("Internal Use Only · CX Division")
    return page
