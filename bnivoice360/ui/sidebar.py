"""Sidebar navigation and a lightweight database health display."""

import sqlite3
from datetime import datetime

import streamlit as st

from bnivoice360.data.repository import VoiceRepository
from bnivoice360.domain import Role


def render_sidebar(repository: VoiceRepository, page_titles: list[str]) -> str:
    with st.sidebar:
        st.markdown("## BNIVoice360")
        st.caption("SOCIAL LISTENING · COMMAND CENTER")
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
