"""Source initialization, overwrite safety, migration and outlier regressions."""

import sqlite3
from dataclasses import replace

import pandas as pd
import pytest

from bnivoice360.config import Settings
from bnivoice360.container import create_services
from bnivoice360.data.database import Database
from bnivoice360.data.seed import TOPIC_COLUMNS, initialize_data, main, prepare_seed
from bnivoice360.domain import IssueFilters
from bnivoice360.services import analytics
from bnivoice360.services.companion import answer_question


def source_frames():
    voices = pd.DataFrame(
        [
            {
                "data_id": identifier,
                "source_platform": "Internal_CRM",
                "raw_content": "feedback",
                "author_id_or_name": "author",
                "created_at": posted,
                "sentiment": "Netral",
                "is_IT": 1,
                "is_Operasional": 0,
                "is_Bisnis": 0,
                "is_Urgent": 1,
            }
            for identifier, posted in (
                ("old", "2026-09-10T23:59:59"),
                ("boundary", "2026-09-11T00:00:00"),
                ("outlier", "2026-10-01"),
                ("unmapped", "2026-10-10"),
                ("undated", ""),
            )
        ]
    )
    topics = pd.DataFrame(
        [
            {
                **dict.fromkeys(TOPIC_COLUMNS, ""),
                "topic_id": 0,
                "topic_title": "Transfer",
                "topic_count": 2,
                "topic_share_pct": 40,
            }
        ]
    )
    mapping = pd.DataFrame({"data_id": ["old", "boundary", "outlier"], "topic_id": [0, 0, -1]})
    return voices, topics, mapping


def sample_seed():
    return prepare_seed(*source_frames(), now=pd.Timestamp("2026-10-11"))


def test_initialization_retains_unmapped_outliers_and_calendar_boundary(services):
    report = initialize_data(services.repository.database, sample_seed())
    assert report["voices"] == 5
    assert report["outliers"] == 1
    assert report["unlabeled"] == 2
    assert report["undated"] == 1
    rows = services.repository.load().set_index("suara_nasabah_id")
    assert rows.loc["old", "escalation_status"] == "Escalated"
    assert rows.loc["boundary", "escalation_status"] == "Open"
    assert rows.loc["undated", "escalation_status"] == "Escalated"
    assert pd.isna(rows.loc["undated", "date_posted"])
    assert rows.loc["outlier", "topic_cluster"] == "Outlier"
    assert rows.loc["outlier", "topic_id"] == -1
    assert pd.isna(rows.loc["unmapped", "topic_id"])
    assert rows.loc["unmapped", "topic_cluster"] == "Unlabeled"
    assert set(rows["source"]) == {"Laporan Nasabah"}
    assert set(rows["sentiment"]) == {"Neutral"}
    assert set(rows["author_id_or_name"]) == {"author"}
    assert services.prediction.run("Laporan Nasabah") == 0
    with services.repository.database.transaction() as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert tables == {"suara_nasabah", "topic_detail"}
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE suara_nasabah SET topic_id=999 WHERE suara_nasabah_id='old'")


def test_preserve_keeps_escalation_and_overwrite_has_restorable_backup(services):
    database = services.repository.database
    data = sample_seed()
    initialize_data(database, data)
    services.repository.escalate("boundary")
    assert initialize_data(database, data)["status"] == "preserved"
    assert (
        services.repository.load()
        .set_index("suara_nasabah_id")
        .loc["boundary", "escalation_status"]
        == "Escalated"
    )
    report = initialize_data(database, data, overwrite=True)
    assert (
        services.repository.load()
        .set_index("suara_nasabah_id")
        .loc["boundary", "escalation_status"]
        == "Open"
    )
    with sqlite3.connect(report["backup"]) as backup:
        assert (
            backup.execute(
                "SELECT escalation_status FROM suara_nasabah WHERE suara_nasabah_id='boundary'"
            ).fetchone()[0]
            == "Escalated"
        )


def test_failed_overwrite_rolls_back_existing_data(services):
    database = services.repository.database
    data = sample_seed()
    initialize_data(database, data)
    invalid = replace(data, voices=(replace(data.voices[0], topic_id=999),))
    with pytest.raises(sqlite3.IntegrityError):
        initialize_data(database, invalid, overwrite=True)
    assert len(services.repository.load()) == 5
    with database.transaction() as connection:
        assert (
            connection.execute("SELECT topic_title FROM topic_detail WHERE topic_id=0").fetchone()[
                0
            ]
            == "Transfer"
        )


@pytest.mark.parametrize("problem", ["duplicate", "missing_topic", "unknown_doc", "invalid_date"])
def test_validation_rejects_invalid_sources(problem):
    voices, topics, mapping = source_frames()
    if problem == "duplicate":
        voices.loc[1, "data_id"] = "old"
    elif problem == "missing_topic":
        mapping.loc[0, "topic_id"] = 99
    elif problem == "unknown_doc":
        mapping.loc[0, "data_id"] = "unknown"
    else:
        voices.loc[0, "created_at"] = "bad-date"
    with pytest.raises(ValueError):
        prepare_seed(voices, topics, mapping)


def test_source_outlier_description_is_preserved_and_labels_use_id(services):
    voices, topics, mapping = source_frames()
    topics.loc[1] = {
        **dict.fromkeys(TOPIC_COLUMNS, ""),
        "topic_id": -1,
        "topic_title": "Noise from model",
        "topic_summary": "original",
    }
    data = prepare_seed(voices, topics, mapping)
    initialize_data(services.repository.database, data)
    row = services.repository.load().set_index("suara_nasabah_id").loc["outlier"]
    assert row["topic_cluster"] == "Outlier"
    assert row["topic_summary"] == "original"


def test_outlier_majority_never_becomes_top_topic(services):
    initialize_data(services.repository.database, sample_seed())
    frame = services.repository.load()
    outliers = frame[frame["topic_id"].eq(-1)]
    frame = pd.concat([frame, *([outliers] * 10)], ignore_index=True)
    assert analytics.top_topics(frame)["topic_cluster"].tolist() == ["Transfer"]
    assert analytics.category_topics(frame)["Topic"].tolist() == ["Transfer"]
    assert analytics.topic_sentiments(frame)["topic_cluster"].tolist() == ["Transfer"]
    assert analytics.overview_metrics(frame).total_volume == 15
    assert analytics.overview_metrics(frame).topic_count == 1
    assert "Outlier" not in answer_question("Top 3 topics", frame)
    special_only = frame[frame["topic_id"].isna() | frame["topic_id"].eq(-1)]
    assert analytics.top_topics(special_only).empty
    assert analytics.category_topics(special_only).empty
    assert analytics.topic_sentiments(special_only).empty
    assert analytics.overview_metrics(special_only).topic_count == 0


def test_startup_ignores_overwrite_configuration_and_does_not_download(services, monkeypatch):
    initialize_data(services.repository.database, sample_seed())
    services.repository.escalate("boundary")
    monkeypatch.setattr("bnivoice360.container.load_seed", lambda _: pytest.fail("download"))
    settings = Settings(
        services.repository.database.path, seed_initial_data=True, overwrite_initialization=True
    )
    second = create_services(settings)
    assert (
        second.repository.load().set_index("suara_nasabah_id").loc["boundary", "escalation_status"]
        == "Escalated"
    )


def test_cli_preserve_overrides_overwrite_environment_without_downloading(
    services, monkeypatch, capsys
):
    initialize_data(services.repository.database, sample_seed())
    monkeypatch.setenv("BNIVOICE360_DB_PATH", str(services.repository.database.path))
    monkeypatch.setenv("BNIVOICE360_INIT_OVERWRITE", "1")
    monkeypatch.setattr("sys.argv", ["seed", "--preserve"])
    monkeypatch.setattr("bnivoice360.data.seed.load_seed", lambda *args: pytest.fail("download"))
    main()
    assert '"status": "preserved"' in capsys.readouterr().out
    assert len(services.repository.load()) == 5


def test_empty_startup_seeds_once(services, monkeypatch):
    calls = []

    def load(settings):
        calls.append(settings)
        return sample_seed()

    monkeypatch.setattr("bnivoice360.container.load_seed", load)
    settings = Settings(services.repository.database.path, seed_initial_data=True)
    create_services(settings)
    create_services(settings)
    assert len(calls) == 1
    assert len(services.repository.load()) == 5


def test_special_topics_remain_filterable_and_all_categories_include_uncategorized(services):
    initialize_data(services.repository.database, sample_seed())
    frame = services.repository.load()
    frame.loc[:, ["is_IT", "is_Bisnis", "is_Operasional"]] = 0
    selected = analytics.filter_issues(frame, IssueFilters(topics=("Outlier",)))
    assert selected["suara_nasabah_id"].tolist() == ["outlier"]
    assert analytics.filter_issues(frame, IssueFilters(topics=("Unlabeled",))).shape[0] == 2
    assert analytics.filter_issues(frame, IssueFilters(categories=("IT",))).empty


def test_legacy_migration_preserves_rows_and_creates_backup(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE suara_nasabah (
            suara_nasabah_id TEXT PRIMARY KEY, content TEXT NOT NULL,
            date_posted DATETIME NOT NULL, source TEXT NOT NULL, sentiment TEXT,
            is_IT INTEGER, is_Bisnis INTEGER, is_Operasional INTEGER, is_Urgent INTEGER,
            is_escalated INTEGER NOT NULL DEFAULT 0, topic_cluster TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        connection.executemany(
            """INSERT INTO suara_nasabah
            (suara_nasabah_id, content, date_posted, source, is_escalated, topic_cluster)
            VALUES (?, 'feedback', '2026-10-10', 'X', 1, ?)""",
            [("assigned", "Reviewed"), ("outlier", "Outlier"), ("unmapped", None)],
        )
    database = Database(path)
    database.initialize()
    database.initialize()
    rows = (
        create_services(Settings(path, seed_initial_data=False))
        .repository.load()
        .set_index("suara_nasabah_id")
    )
    assert len(rows) == 3
    assert rows["escalation_status"].eq("Escalated").all()
    assert rows.loc["assigned", "topic_cluster"] == "Reviewed"
    assert rows.loc["outlier", "topic_id"] == -1
    assert pd.isna(rows.loc["unmapped", "topic_id"])
    assert len(list(tmp_path.glob("legacy.backup-*.db"))) == 1
