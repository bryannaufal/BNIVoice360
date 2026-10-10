"""Load BNI styling from a dedicated asset."""

from pathlib import Path

import streamlit as st

STYLE_PATH = Path(__file__).resolve().parent / "assets" / "styles.css"


def inject_css() -> None:
    styles = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
