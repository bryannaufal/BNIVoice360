"""Translate filter widgets into a domain value; selection logic is in analytics."""

import pandas as pd
import streamlit as st

from bnivoice360.config import CATEGORY_COLUMNS, SENTIMENTS, SOURCES
from bnivoice360.domain import IssueFilters
from bnivoice360.ui.components import section_title


def _remember_filter(key: str) -> None:
    st.session_state["issue_filter_values"][key] = st.session_state[key]


def render_issue_filters(frame: pd.DataFrame) -> IssueFilters:
    section_title("Filter issues")
    saved = st.session_state.setdefault("issue_filter_values", {})
    dates = frame["date_posted"].dropna()
    minimum, maximum = dates.min().date(), dates.max().date()
    saved_dates = saved.get("issues_dates", (minimum, maximum))
    saved_dates = tuple(max(minimum, min(value, maximum)) for value in saved_dates)
    left, middle, right = st.columns(3)
    selected = left.date_input(
        "Date range",
        saved_dates,
        min_value=minimum,
        max_value=maximum,
        key="issues_dates",
        on_change=_remember_filter,
        args=("issues_dates",),
    )
    sources = middle.multiselect(
        "Sources",
        SOURCES,
        default=saved.get("issues_sources", list(SOURCES)),
        key="issues_sources",
        on_change=_remember_filter,
        args=("issues_sources",),
    )
    topic_options = sorted(frame["topic_cluster"].dropna().astype(str).unique())
    topics = right.multiselect(
        "Topic clusters",
        topic_options,
        default=[
            topic for topic in saved.get("issues_topics", topic_options) if topic in topic_options
        ],
        key="issues_topics",
        on_change=_remember_filter,
        args=("issues_topics",),
    )
    left, middle, right = st.columns(3)
    sentiments = left.multiselect(
        "Sentiment",
        SENTIMENTS,
        default=saved.get("issues_sentiments", list(SENTIMENTS)),
        key="issues_sentiments",
        on_change=_remember_filter,
        args=("issues_sentiments",),
    )
    categories = middle.multiselect(
        "Category",
        list(CATEGORY_COLUMNS),
        default=saved.get("issues_categories", list(CATEGORY_COLUMNS)),
        key="issues_categories",
        on_change=_remember_filter,
        args=("issues_categories",),
    )
    escalation = right.selectbox(
        "Escalation status",
        ["All", "Pending Escalation", "Escalated"],
        index=["All", "Pending Escalation", "Escalated"].index(
            saved.get("issues_escalation", "All")
        ),
        key="issues_escalation",
        on_change=_remember_filter,
        args=("issues_escalation",),
    )
    urgent_only = st.toggle(
        "Urgent only",
        value=saved.get("issues_urgent", False),
        key="issues_urgent",
        on_change=_remember_filter,
        args=("issues_urgent",),
    )
    start = selected[0] if selected else minimum
    end = selected[-1] if selected else maximum
    return IssueFilters(
        start,
        end,
        tuple(sources),
        tuple(topics),
        tuple(sentiments),
        tuple(categories),
        escalation,
        urgent_only,
    )
