"""Pagination, filter persistence, and committed card actions through Streamlit."""

import pandas as pd
from streamlit.testing.v1 import AppTest

from bnivoice360.config import FEED_PAGE_SIZE, PROJECT_ROOT, Settings
from bnivoice360.container import create_services


def _navigate(app, title):
    app.sidebar.radio(key="current_page").set_value(title).run()
    assert not app.exception
    if title == "Issue Analytics & Voice Feed":
        app.selectbox(key="issues_deep_dive_topic").set_value("all").run()


def _actions(app):
    return [
        widget
        for widget in app.selectbox
        if widget.key.startswith("status_") and widget.value == "Open"
    ]


def test_feed_is_paginated_and_escalation_preserves_page(streamlit_app, tmp_path):
    app = streamlit_app
    _navigate(app, "Issue Analytics & Voice Feed")
    assert len(app.expander) == FEED_PAGE_SIZE
    assert app.button(key="feed_previous").disabled
    page_one_ids = {button.key for button in _actions(app)}
    app.button(key="feed_next").click().run()
    assert not app.exception
    assert app.session_state["voice_feed_page"] == 2
    assert len(app.expander) == FEED_PAGE_SIZE
    assert not page_one_ids.intersection({button.key for button in _actions(app)})
    action = _actions(app)[0]
    record_id = action.key.removeprefix("status_")
    action.set_value("Escalated").run()
    assert not app.exception
    assert app.session_state["voice_feed_page"] == 2
    data = (
        create_services(Settings(tmp_path / "ui.db"))
        .repository.load()
        .set_index("suara_nasabah_id")
    )
    assert data.loc[record_id, "escalation_status"] == "Escalated"
    assert action.key not in {button.key for button in _actions(app)}


def test_page_filters_and_search_survive_navigation_and_reset_when_changed(streamlit_app):
    app = streamlit_app
    _navigate(app, "Issue Analytics & Voice Feed")
    sources = ["X", "Facebook", "Instagram"]
    app.multiselect(key="issues_sources").set_value(sources).run()
    assert not app.exception
    app.button(key="feed_next").click().run()
    assert app.session_state["voice_feed_page"] == 2
    _navigate(app, "Graph Analytics")
    _navigate(app, "Issue Analytics & Voice Feed")
    assert app.multiselect(key="issues_sources").value == sources
    assert app.session_state["voice_feed_page"] == 2
    app.text_input(key="issues_search").set_value("transfer").run()
    assert not app.exception
    assert app.session_state["voice_feed_page"] == 1
    _navigate(app, "Home")
    _navigate(app, "Issue Analytics & Voice Feed")
    assert app.text_input(key="issues_search").value == "transfer"
    app.multiselect(key="issues_sources").set_value([]).run()
    assert not app.exception
    assert app.session_state["voice_feed_page"] == 1
    assert app.button(key="feed_next").disabled
    assert len(app.expander) == 0


def test_pending_only_escalation_clamps_the_last_page(tmp_path, monkeypatch):
    database_path = tmp_path / "voices.db"
    monkeypatch.setenv("BNIVOICE360_DB_PATH", str(database_path))
    services = create_services(Settings(database_path, seed_initial_data=False))
    services.ingestion.ingest(
        pd.DataFrame(
            [
                {
                    "suara_nasabah_id": f"pending-{index}",
                    "content": f"Voice {index}",
                    "date_posted": "2026-10-10",
                    "source": "X",
                    "sentiment": "Negative",
                    "is_IT": 1,
                    "is_Bisnis": 0,
                    "is_Operasional": 0,
                    "is_Urgent": 0,
                    "topic_cluster": "Test topic",
                }
                for index in range(FEED_PAGE_SIZE + 1)
            ]
        )
    )
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    _navigate(app, "Issue Analytics & Voice Feed")
    app.selectbox(key="issues_escalation").set_value("Open").run()
    app.button(key="feed_next").click().run()
    assert not app.exception
    assert len(app.expander) == 1
    _actions(app)[0].set_value("Escalated").run()
    assert not app.exception
    assert app.session_state["voice_feed_page"] == 1
    assert len(app.expander) == FEED_PAGE_SIZE
    assert services.repository.load()["escalation_status"].eq("Escalated").sum() == 1


def test_search_empty_state_and_html_content_are_safe(streamlit_app, tmp_path):
    services = create_services(Settings(tmp_path / "ui.db"))
    services.ingestion.ingest(
        pd.DataFrame(
            [
                {
                    "content": "<script>alert('x')</script>",
                    "date_posted": "2026-10-10",
                    "source": "X",
                    "sentiment": "Negative",
                    "is_IT": 1,
                    "is_Urgent": 1,
                    "topic_cluster": "Gagal Transfer",
                }
            ]
        )
    )
    app = streamlit_app
    _navigate(app, "Issue Analytics & Voice Feed")
    app.text_input(key="issues_search").set_value("<script>").run()
    assert not app.exception
    assert len(app.expander) == 1
    assert any("&lt;script&gt;" in element.value for element in app.markdown)
    app.text_input(key="issues_search").set_value("no matching voices at all").run()
    assert not app.exception
    assert len(app.expander) == 0
    assert app.button(key="feed_previous").disabled
    assert app.button(key="feed_next").disabled
