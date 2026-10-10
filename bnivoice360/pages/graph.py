"""Reserved graph analytics page."""

import streamlit as st

from bnivoice360.container import AppServices
from bnivoice360.ui.components import header


def render(services: AppServices) -> None:
    header(
        "Graph Analytics",
        "Relationship intelligence untuk memahami keterkaitan kanal, isu, dan journey nasabah.",
        "Network Knowledge Mapping",
    )
    st.markdown(
        '<div class="coming"><div class="icon">◉—◉</div>'
        "<h2>Graph Analytics & Network Knowledge Mapping</h2>"
        '<p class="muted">Module is currently under development (Coming Soon).</p></div>',
        unsafe_allow_html=True,
    )
