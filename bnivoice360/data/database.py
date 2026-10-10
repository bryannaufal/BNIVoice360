"""Database lifecycle and short-lived, safely closed transactions."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS suara_nasabah (
    suara_nasabah_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    date_posted DATETIME NOT NULL,
    source TEXT NOT NULL,
    sentiment TEXT,
    is_IT INTEGER,
    is_Bisnis INTEGER,
    is_Operasional INTEGER,
    is_Urgent INTEGER,
    is_escalated INTEGER NOT NULL DEFAULT 0,
    topic_cluster TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_voice_source_date
    ON suara_nasabah (source, date_posted);
CREATE INDEX IF NOT EXISTS idx_voice_urgency
    ON suara_nasabah (is_Urgent, is_escalated);
"""


class Database:
    """Owns connection lifetime; no global or shared SQLite connections."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=20)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as connection:
            connection.executescript(SCHEMA)
