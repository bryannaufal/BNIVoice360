"""Compact paginated voice cards with durable state and committed escalation."""

import hashlib
from html import escape
from math import ceil

import pandas as pd
import streamlit as st

from bnivoice360.config import ESCALATION_STATUSES, FEED_PAGE_SIZE
from bnivoice360.data.repository import VoiceRepository
from bnivoice360.services.analytics import category_labels, voice_feed
from bnivoice360.ui.components import section_title


def _remember_search() -> None:
    st.session_state["voice_feed_search"] = st.session_state["issues_search"]


def _update_status(record_id: str, repository: VoiceRepository, key: str) -> None:
    status = st.session_state[key]
    changed = repository.set_escalation_status(record_id, status)
    st.session_state["escalation_notice"] = (
        f"Status issue diperbarui menjadi {status}."
        if changed
        else "Status issue sudah sesuai atau tidak lagi tersedia."
    )


def _change_page(delta: int, page_count: int) -> None:
    current = st.session_state.get("voice_feed_page", 1)
    st.session_state["voice_feed_page"] = min(max(1, current + delta), page_count)


def _pagination(total: int, context: str) -> tuple[int, int]:
    """Reset on changed filters/search, preserve on data writes, and clamp shrinking results."""
    page_count = max(1, ceil(total / FEED_PAGE_SIZE))
    if st.session_state.get("voice_feed_context") != context:
        st.session_state["voice_feed_page"] = 1
        st.session_state["voice_feed_context"] = context
    page = min(max(1, st.session_state.get("voice_feed_page", 1)), page_count)
    st.session_state["voice_feed_page"] = page
    previous, summary, following = st.columns([1, 3, 1])
    previous.button(
        "← Previous",
        key="feed_previous",
        disabled=page == 1,
        on_click=_change_page,
        args=(-1, page_count),
        width="stretch",
    )
    with summary:
        start = (page - 1) * FEED_PAGE_SIZE + 1 if total else 0
        end = min(page * FEED_PAGE_SIZE, total)
        st.markdown(
            f'<div class="feed-pagination-summary">Page <b>{page}</b> of <b>{page_count}</b>'
            f"<span>{start}–{end} of {total} voices · urgent first · Escalated → Open → Resolved</span></div>",
            unsafe_allow_html=True,
        )
    following.button(
        "Next →",
        key="feed_next",
        disabled=page >= page_count,
        on_click=_change_page,
        args=(1, page_count),
        width="stretch",
    )
    return page, page_count


def _badge(label: object, style: str) -> str:
    return f'<span class="voice-badge voice-badge--{style}">{escape(str(label))}</span>'


def _metadata(row: pd.Series) -> str:
    posted = row["date_posted"]
    date_label = posted.strftime("%d %b %Y, %H:%M:%S WIB") if pd.notna(posted) else "Unknown date"
    sentiment = row["sentiment"] if pd.notna(row["sentiment"]) else "Unlabeled"
    topic = row["topic_cluster"] if pd.notna(row["topic_cluster"]) else "Unlabeled"
    badges = [
        _badge(row["source"], "topic"),
        _badge(sentiment, str(sentiment).lower()),
        _badge(topic, "topic"),
        *[_badge(category, category.lower()) for category in category_labels(row)],
    ]
    if row["is_Urgent"] == 1:
        badges.append(_badge("🚨 URGENT", "urgent"))
    return (
        f'<div class="voice-meta"><strong>{escape(str(row["source"]))}</strong>'
        f"<span>{date_label}</span></div>"
        f'<div class="voice-badges">{"".join(badges)}</div>'
    )


def _render_card(row: pd.Series, repository: VoiceRepository) -> None:
    record_id = str(row["suara_nasabah_id"])
    # Hashed keys remain stable even when uploaded IDs contain CSS-unsafe characters.
    card_key = hashlib.sha256(record_id.encode()).hexdigest()[:20]
    with st.container(border=True, key=f"voice_card_{card_key}"):
        metadata, action = st.columns([4, 1], gap="small")
        with metadata:
            st.markdown(_metadata(row), unsafe_allow_html=True)
        with action:
            status = row["escalation_status"]
            status_key = f"status_{record_id}"
            st.session_state[status_key] = status
            st.selectbox(
                "Status",
                ESCALATION_STATUSES,
                key=status_key,
                label_visibility="collapsed",
                on_change=_update_status,
                args=(record_id, repository, status_key),
            )
        content = str(row["content"])
        st.markdown(
            f'<div class="voice-content">{escape(content)}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Full text & details"):
            st.write(row["content"])
            st.caption(
                f"ID: {record_id} · Topic: {row['topic_cluster']} · "
                f"Sentiment: {row['sentiment']} · "
                f"Status: {row['escalation_status']} · "
                f"Category: {', '.join(category_labels(row)) or 'Uncategorized'}"
            )
        if row["escalation_status"] == "Open":
            if st.button("Cari keluhan serupa", key=f"similar_{card_key}"):
                st.session_state["batch_anchor_id"] = record_id
                st.rerun()


def render_voice_feed(
    frame: pd.DataFrame, repository: VoiceRepository, filter_context: str = ""
) -> None:
    section_title("Suara Nasabah & Action Center")
    search = st.text_input(
        "Search customer voice",
        value=st.session_state.get("voice_feed_search", ""),
        placeholder="Cari kata seperti transfer, QRIS, login…",
        key="issues_search",
        on_change=_remember_search,
    )
    feed = voice_feed(frame, search)
    context = repr((filter_context, search, FEED_PAGE_SIZE))
    page, _ = _pagination(len(feed), context)
    if feed.empty:
        st.info("Tidak ada voice yang cocok dengan pencarian ini.")
        return
    start = (page - 1) * FEED_PAGE_SIZE
    for _, row in feed.iloc[start : start + FEED_PAGE_SIZE].iterrows():
        _render_card(row, repository)
