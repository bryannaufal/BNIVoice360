"""Three-state lifecycle, binary migration, analytics and card actions."""

import sqlite3

import pandas as pd
import pytest

from bnivoice360.config import Settings
from bnivoice360.container import create_services
from bnivoice360.data.database import Database
from bnivoice360.domain import IssueFilters
from bnivoice360.services.analytics import filter_issues, open_urgent_count, voice_feed
from bnivoice360.services.companion import answer_question


def status_records():
    return pd.DataFrame(
        [
            {
                "suara_nasabah_id": status,
                "content": "feedback",
                "source": "X",
                "date_posted": "2026-10-10",
                "sentiment": "Negative",
                "is_IT": 1,
                "is_Bisnis": 0,
                "is_Operasional": 0,
                "is_Urgent": 1,
                "escalation_status": status,
            }
            for status in ("Open", "Escalated", "Resolved")
        ]
    )


def test_status_lifecycle_is_persistent_and_idempotent(services):
    services.ingestion.ingest(status_records())
    repository = services.repository
    assert repository.escalate("Open")
    assert not repository.escalate("Open")
    assert repository.resolve("Open")
    assert not repository.resolve("Open")
    assert not repository.escalate("Open")  # Resolved must be reopened explicitly.
    assert repository.reopen("Open")
    assert not repository.reopen("Open")
    assert repository.resolve("Open")  # Direct resolution of an Open voice is allowed.
    assert not repository.set_escalation_status("missing", "Resolved")
    with pytest.raises(ValueError):
        repository.set_escalation_status("Open", "Closed")
    second = create_services(Settings(repository.database.path, seed_initial_data=False))
    assert (
        second.repository.load().set_index("suara_nasabah_id").loc["Open", "escalation_status"]
        == "Resolved"
    )


def test_ingestion_validates_status_and_defaults_blank_or_missing_to_open(services):
    frame = status_records()
    frame.loc[0, "escalation_status"] = "Closed"
    result = services.ingestion.ingest(frame)
    assert result.inserted == 2
    assert result.skipped == 1
    assert "escalation_status" in result.errors[0]
    services.ingestion.ingest(
        pd.DataFrame(
            [
                {
                    "suara_nasabah_id": "missing",
                    "content": "hello",
                    "source": "X",
                    "date_posted": "2026-10-10",
                },
                {
                    "suara_nasabah_id": "blank",
                    "content": "hello",
                    "source": "X",
                    "date_posted": "2026-10-10",
                    "escalation_status": " ",
                },
            ]
        )
    )
    rows = services.repository.load().set_index("suara_nasabah_id")
    assert rows.loc["missing", "escalation_status"] == "Open"
    assert rows.loc["blank", "escalation_status"] == "Open"
    services.prediction.run("X")
    assert (
        services.repository.load()
        .set_index("suara_nasabah_id")
        .loc["Resolved", "escalation_status"]
        == "Resolved"
    )
    with services.repository.database.transaction() as connection:
        for status in ("Closed", None, 1):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE suara_nasabah SET escalation_status=? WHERE suara_nasabah_id='Resolved'",
                    (status,),
                )


def test_status_filters_urgent_metrics_and_companion(services):
    services.ingestion.ingest(status_records())
    frame = services.repository.load()
    assert open_urgent_count(frame) == 1
    assert voice_feed(frame).iloc[0]["escalation_status"] == "Escalated"
    for status in ("Open", "Escalated", "Resolved"):
        assert filter_issues(frame, IssueFilters(escalation_status=status))[
            "escalation_status"
        ].tolist() == [status]
        assert "**1 voice**" in answer_question(f"How many {status} voices?", frame)
    assert "**2 voice**" in answer_question("How many unresolved voices?", frame)
    assert "**1 voice**" in answer_question("Berapa urgent belum dieskalasi?", frame)
    assert "**2 voice**" in answer_question("Berapa belum selesai?", frame)
    assert open_urgent_count(filter_issues(frame, IssueFilters(escalation_status="Resolved"))) == 0


def test_two_table_binary_migration_preserves_topics_authors_and_backup(tmp_path):
    path = tmp_path / "binary.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE topic_detail (topic_id INTEGER PRIMARY KEY, topic_count INTEGER, topic_share_pct REAL, bertopic_name TEXT, keywords TEXT, keyword_scores TEXT, representative_docs TEXT, topic_title TEXT, topic_summary TEXT, customer_issue TEXT, labeling_status TEXT)"
        )
        connection.execute(
            "INSERT INTO topic_detail (topic_id, topic_title, topic_summary) VALUES (7, 'Topic seven', 'Original description')"
        )
        connection.execute("""CREATE TABLE suara_nasabah (
            suara_nasabah_id TEXT PRIMARY KEY, content TEXT NOT NULL,
            date_posted DATETIME, source TEXT NOT NULL, sentiment TEXT,
            is_IT INTEGER, is_Bisnis INTEGER, is_Operasional INTEGER, is_Urgent INTEGER,
            is_escalated INTEGER NOT NULL DEFAULT 0, author_id_or_name TEXT,
            topic_id INTEGER REFERENCES topic_detail(topic_id),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        connection.execute("CREATE INDEX idx_voice_topic ON suara_nasabah (topic_id)")
        connection.executemany(
            """INSERT INTO suara_nasabah
            (suara_nasabah_id, content, source, is_escalated, topic_id, author_id_or_name, created_at)
            VALUES (?, 'feedback', 'X', ?, ?, 'author', '2026-01-01')""",
            [("open", 0, 7), ("escalated", 1, None)],
        )
    database = Database(path)
    database.initialize()
    database.initialize()
    services = create_services(Settings(path, seed_initial_data=False))
    rows = services.repository.load().set_index("suara_nasabah_id")
    assert len(rows) == 2
    assert "is_escalated" not in rows.columns
    assert rows.loc["open", "escalation_status"] == "Open"
    assert rows.loc["escalated", "escalation_status"] == "Escalated"
    assert rows.loc["open", "topic_id"] == 7
    assert rows.loc["open", "topic_summary"] == "Original description"
    assert rows["author_id_or_name"].eq("author").all()
    assert rows["created_at"].eq("2026-01-01").all()
    with database.transaction() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    backups = list(tmp_path.glob("binary.backup-*.db"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert (
            backup.execute(
                "SELECT is_escalated FROM suara_nasabah WHERE suara_nasabah_id='escalated'"
            ).fetchone()[0]
            == 1
        )


def test_voice_card_resolve_and_reopen_persist_and_filters_include_resolved(
    streamlit_app, tmp_path
):
    app = streamlit_app
    app.sidebar.radio(key="current_page").set_value("Issue Analytics & Voice Feed").run()
    # Keep the whole channel on one page when escalation changes its sort position.
    app.multiselect(key="issues_sources").set_value(["X"]).run()
    app.selectbox(key="issues_deep_dive_topic").set_value("all").run()
    assert app.selectbox(key="issues_escalation").options == [
        "All",
        "Open",
        "Escalated",
        "Resolved",
    ]
    action = next(
        widget
        for widget in app.selectbox
        if widget.key.startswith("status_") and widget.value == "Open"
    )
    record_id = action.key.removeprefix("status_")
    action.set_value("Escalated").run()
    assert not app.exception
    app.selectbox(key=f"status_{record_id}").set_value("Resolved").run()
    assert not app.exception
    services = create_services(Settings(tmp_path / "ui.db", seed_initial_data=False))
    assert (
        services.repository.load().set_index("suara_nasabah_id").loc[record_id, "escalation_status"]
        == "Resolved"
    )
    assert app.selectbox(key=f"status_{record_id}").value == "Resolved"
    app.selectbox(key="issues_escalation").set_value("Resolved").run()
    assert not app.exception
    assert len(app.expander) == 1
    app.selectbox(key=f"status_{record_id}").set_value("Open").run()
    assert not app.exception
    assert len(app.expander) == 0
    assert (
        services.repository.load().set_index("suara_nasabah_id").loc[record_id, "escalation_status"]
        == "Open"
    )
