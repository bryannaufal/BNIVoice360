"""Small shared display components, with escaped dynamic HTML."""

from html import escape

import streamlit as st


def header(title: str, subtitle: str, eyebrow: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{escape(eyebrow)}</div>'
        f"<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def urgent_alert(count: int) -> None:
    if count:
        st.markdown(
            f'<div class="alert"><b>⚠ {count} critical voice(s) berstatus Open.</b>'
            "<br>Prioritaskan pemeriksaan pada voice feed di bawah.</div>",
            unsafe_allow_html=True,
        )
