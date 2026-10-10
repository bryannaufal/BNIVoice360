"""The only application layer that executes voice queries."""

from collections.abc import Iterable
from dataclasses import asdict

import pandas as pd

from bnivoice360.config import PREDICTION_COLUMNS
from bnivoice360.data.database import Database
from bnivoice360.domain import Prediction, VoiceRecord

INSERT_VOICE = """
INSERT INTO suara_nasabah
(suara_nasabah_id, content, date_posted, source, sentiment, is_IT,
 is_Bisnis, is_Operasional, is_Urgent, is_escalated, topic_cluster)
VALUES (:suara_nasabah_id, :content, :date_posted, :source, :sentiment,
        :is_IT, :is_Bisnis, :is_Operasional, :is_Urgent,
        :is_escalated, :topic_cluster)
ON CONFLICT(suara_nasabah_id) DO NOTHING
"""


class VoiceRepository:
    """SQLite repository; services receive an instance by constructor injection."""

    def __init__(self, database: Database):
        self.database = database

    def seed_if_empty(self, records: Iterable[VoiceRecord]) -> None:
        with self.database.transaction() as connection:
            # Serialize the empty check and insert across simultaneous sessions.
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT COUNT(*) FROM suara_nasabah").fetchone()[0] == 0:
                connection.executemany(INSERT_VOICE, [asdict(record) for record in records])

    def load(self) -> pd.DataFrame:
        with self.database.transaction() as connection:
            frame = pd.read_sql_query(
                "SELECT * FROM suara_nasabah ORDER BY date_posted DESC", connection
            )
        frame["date_posted"] = pd.to_datetime(frame["date_posted"], errors="coerce")
        for column in ("is_IT", "is_Bisnis", "is_Operasional", "is_Urgent", "is_escalated"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def insert(self, records: Iterable[VoiceRecord]) -> tuple[int, int]:
        inserted = skipped = 0
        with self.database.transaction() as connection:
            for record in records:
                cursor = connection.execute(INSERT_VOICE, asdict(record))
                inserted += cursor.rowcount
                skipped += int(cursor.rowcount == 0)
        return inserted, skipped

    def find_unpredicted(self, source: str) -> list[dict]:
        missing = " OR ".join(f"{column} IS NULL" for column in PREDICTION_COLUMNS)
        with self.database.transaction() as connection:
            rows = connection.execute(
                f"SELECT suara_nasabah_id, content FROM suara_nasabah "
                f"WHERE source=? AND ({missing})",
                (source,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fill_predictions(self, predictions: Iterable[tuple[str, Prediction]]) -> None:
        # Only missing values are filled; reviewed labels and escalation stay intact.
        assignments = ", ".join(
            f"{column}=COALESCE({column}, :{column})" for column in PREDICTION_COLUMNS
        )
        with self.database.transaction() as connection:
            for record_id, prediction in predictions:
                connection.execute(
                    f"UPDATE suara_nasabah SET {assignments} WHERE suara_nasabah_id=:record_id",
                    {**asdict(prediction), "record_id": record_id},
                )

    def escalate(self, record_id: str) -> bool:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE suara_nasabah SET is_escalated=1 "
                "WHERE suara_nasabah_id=? AND is_escalated=0",
                (record_id,),
            )
        return cursor.rowcount == 1

    def health(self) -> dict[str, int | float]:
        with self.database.transaction() as connection:
            count = connection.execute("SELECT COUNT(*) FROM suara_nasabah").fetchone()[0]
        return {"records": count, "size_kb": self.database.path.stat().st_size / 1024}
