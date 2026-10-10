"""Session initialization and notifications that survive a Streamlit rerun."""

import streamlit as st

from bnivoice360.domain import Role


def initialize_state() -> None:
    st.session_state.setdefault("selected_role", Role.MANAGER.value)
    st.session_state.setdefault("current_page", "Home")
    st.session_state.setdefault("voice_feed_page", 1)
    st.session_state.setdefault("voice_feed_search", "")
    st.session_state.setdefault("issue_filter_values", {})
    st.session_state.setdefault(
        "chat_history",
        [
            {
                "role": "assistant",
                "content": (
                    "Halo! Saya siap membantu membaca data VoC BNI. "
                    "Coba tanyakan top issue, sentimen kanal, atau urgent escalation."
                ),
            }
        ],
    )


def show_notice(key: str) -> None:
    message = st.session_state.pop(key, None)
    if message:
        st.success(message)


def notify_and_rerun(key: str, message: str) -> None:
    st.session_state[key] = message
    st.rerun()
