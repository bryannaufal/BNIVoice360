"""Database lifecycle and short-lived, safely closed transactions."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS topic_detail (
    topic_id INTEGER PRIMARY KEY,
    topic_count INTEGER,
    topic_share_pct REAL,
    bertopic_name TEXT,
    keywords TEXT,
    keyword_scores TEXT,
    representative_docs TEXT,
    topic_title TEXT,
    topic_summary TEXT,
    customer_issue TEXT,
    labeling_status TEXT
);
CREATE TABLE IF NOT EXISTS suara_nasabah (
    suara_nasabah_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    date_posted DATETIME,
    source TEXT NOT NULL,
    sentiment TEXT,
    is_IT INTEGER,
    is_Bisnis INTEGER,
    is_Operasional INTEGER,
    is_Urgent INTEGER,
    escalation_status TEXT NOT NULL DEFAULT 'Open'
        CHECK (escalation_status IN ('Open', 'Escalated', 'Resolved')),
    author_id_or_name TEXT,
    topic_id INTEGER REFERENCES topic_detail(topic_id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_voice_source_date
    ON suara_nasabah (source, date_posted);
CREATE INDEX IF NOT EXISTS idx_voice_urgency
    ON suara_nasabah (is_Urgent, escalation_status);
CREATE INDEX IF NOT EXISTS idx_voice_topic ON suara_nasabah (topic_id);
"""


class Database:
    """Owns connection lifetime; no global or shared SQLite connections."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as connection:
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(suara_nasabah)")
            }
            if columns and ("topic_id" not in columns or "escalation_status" not in columns):
                self.backup()
            connection.execute("BEGIN IMMEDIATE")
            # Recheck after acquiring the write lock, for concurrent startup.
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(suara_nasabah)")
            }
            legacy = bool(
                columns and ("topic_id" not in columns or "escalation_status" not in columns)
            )
            if legacy:
                connection.execute("ALTER TABLE suara_nasabah RENAME TO legacy_voices")
                connection.execute("DROP INDEX IF EXISTS idx_voice_source_date")
                connection.execute("DROP INDEX IF EXISTS idx_voice_urgency")
                connection.execute("DROP INDEX IF EXISTS idx_voice_topic")
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    connection.execute(statement)
            connection.execute(
                "INSERT OR IGNORE INTO topic_detail (topic_id, topic_title) VALUES (-1, 'Outlier')"
            )
            if legacy:
                rows = connection.execute("SELECT * FROM legacy_voices").fetchall()
                for row in rows:
                    values = dict(row)
                    topic_id = (
                        values.get("topic_id")
                        if "topic_id" in columns
                        else resolve_topic(connection, values.get("topic_cluster"))
                    )
                    status = values.get("escalation_status")
                    if status is None:
                        status = "Escalated" if values.get("is_escalated") == 1 else "Open"
                    connection.execute(
                        """INSERT INTO suara_nasabah
                        (suara_nasabah_id, content, date_posted, source, sentiment, is_IT,
                         is_Bisnis, is_Operasional, is_Urgent, escalation_status, topic_id,
                         author_id_or_name, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        tuple(
                            row[key]
                            for key in (
                                "suara_nasabah_id",
                                "content",
                                "date_posted",
                                "source",
                                "sentiment",
                                "is_IT",
                                "is_Bisnis",
                                "is_Operasional",
                                "is_Urgent",
                            )
                        )
                        + (status, topic_id, values.get("author_id_or_name"), row["created_at"]),
                    )
                connection.execute("DROP TABLE legacy_voices")
            connection.execute("PRAGMA user_version=3")

    def backup(self) -> Path:
        """SQLite's backup API includes committed WAL data as well."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        destination = self.path.with_name(f"{self.path.stem}.backup-{stamp}{self.path.suffix}")
        with sqlite3.connect(self.path) as source, sqlite3.connect(destination) as target:
            source.backup(target)
        return destination


def resolve_topic(connection: sqlite3.Connection, title: str | None) -> int | None:
    """Resolve legacy/imported classifier titles into actual topic records."""
    if not title or title == "Unlabeled":
        return None
    if title.lower() in ("outlier", "out;ier"):
        connection.execute(
            "INSERT OR IGNORE INTO topic_detail (topic_id, topic_title) VALUES (-1, 'Outlier')"
        )
        return -1
    row = connection.execute(
        "SELECT topic_id FROM topic_detail WHERE topic_title=? AND topic_id != -1", (title,)
    ).fetchone()
    if row:
        return row[0]
    topic_id = connection.execute(
        "SELECT MAX(COALESCE((SELECT MAX(topic_id) FROM topic_detail), -1), -1) + 1"
    ).fetchone()[0]
    connection.execute(
        "INSERT INTO topic_detail (topic_id, topic_title, labeling_status) VALUES (?, ?, ?)",
        (topic_id, title, "local"),
    )
    return topic_id
