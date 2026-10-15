"""The only application layer that executes voice queries."""

from collections.abc import Iterable
from dataclasses import asdict

import pandas as pd

from bnivoice360.config import ESCALATION_STATUSES, PREDICTION_COLUMNS
from bnivoice360.data.database import Database, resolve_topic
from bnivoice360.domain import Prediction, VoiceRecord

INSERT_VOICE = """
INSERT INTO suara_nasabah
(suara_nasabah_id, content, date_posted, source, sentiment, is_IT,
 is_Bisnis, is_Operasional, is_Urgent, escalation_status, topic_id, author_id_or_name, masked_content)
VALUES (:suara_nasabah_id, :content, :date_posted, :source, :sentiment,
        :is_IT, :is_Bisnis, :is_Operasional, :is_Urgent,
        :escalation_status, :topic_id, :author_id_or_name, :masked_content)
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
                for record in records:
                    self._insert_record(connection, record)

    def is_empty(self) -> bool:
        with self.database.transaction() as connection:
            return connection.execute("SELECT COUNT(*) FROM suara_nasabah").fetchone()[0] == 0

    def topic_ids(self) -> set[int]:
        with self.database.transaction() as connection:
            return {row[0] for row in connection.execute("SELECT topic_id FROM topic_detail")}

    @staticmethod
    def _insert_record(connection, record: VoiceRecord):
        values = asdict(record)
        if values["topic_id"] is None:
            values["topic_id"] = resolve_topic(connection, record.topic_cluster)
        return connection.execute(INSERT_VOICE, values)

    def load(self) -> pd.DataFrame:
        with self.database.transaction() as connection:
            frame = pd.read_sql_query(
                """SELECT v.*, CASE
                    WHEN v.topic_id IS NULL THEN 'Unlabeled'
                    WHEN v.topic_id = -1 THEN 'Outlier'
                    ELSE COALESCE(NULLIF(t.topic_title, ''), t.bertopic_name,
                                  'Topic ' || v.topic_id)
                    END AS topic_cluster,
                    t.topic_summary, t.customer_issue
                    FROM suara_nasabah v LEFT JOIN topic_detail t ON v.topic_id=t.topic_id
                    ORDER BY v.date_posted DESC""",
                connection,
            )
        frame["date_posted"] = pd.to_datetime(frame["date_posted"], errors="coerce")
        for column in ("is_IT", "is_Bisnis", "is_Operasional", "is_Urgent"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def insert(self, records: Iterable[VoiceRecord]) -> tuple[int, int]:
        inserted = skipped = 0
        with self.database.transaction() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for record in records:
                if connection.execute(
                    "SELECT 1 FROM suara_nasabah WHERE suara_nasabah_id=?",
                    (record.suara_nasabah_id,),
                ).fetchone():
                    skipped += 1
                    continue
                cursor = self._insert_record(connection, record)
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
            connection.execute("BEGIN IMMEDIATE")
            for record_id, prediction in predictions:
                connection.execute(
                    f"UPDATE suara_nasabah SET {assignments} WHERE suara_nasabah_id=:record_id",
                    {**asdict(prediction), "record_id": record_id},
                )
                row = connection.execute(
                    "SELECT topic_id FROM suara_nasabah WHERE suara_nasabah_id=?", (record_id,)
                ).fetchone()
                if row is not None and row[0] is None:
                    topic_id = resolve_topic(connection, prediction.topic_cluster)
                    connection.execute(
                        "UPDATE suara_nasabah SET topic_id=? WHERE suara_nasabah_id=?",
                        (topic_id, record_id),
                    )

    def escalate(self, record_id: str) -> bool:
        return self.set_escalation_status(record_id, "Escalated", from_status="Open")

    def resolve(self, record_id: str) -> bool:
        return self.set_escalation_status(record_id, "Resolved")

    def reopen(self, record_id: str) -> bool:
        return self.set_escalation_status(record_id, "Open")

    def set_escalation_status(
        self, record_id: str, status: str, from_status: str | None = None
    ) -> bool:
        if status not in ESCALATION_STATUSES or (
            from_status is not None and from_status not in ESCALATION_STATUSES
        ):
            raise ValueError("escalation_status must be Open, Escalated, or Resolved.")
        condition = " AND escalation_status=?" if from_status is not None else ""
        parameters = (status, record_id, status)
        if from_status is not None:
            parameters += (from_status,)
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE suara_nasabah SET escalation_status=? "
                "WHERE suara_nasabah_id=? AND escalation_status!=?" + condition,
                parameters,
            )
        return cursor.rowcount == 1

    def health(self) -> dict[str, int | float]:
        with self.database.transaction() as connection:
            count = connection.execute("SELECT COUNT(*) FROM suara_nasabah").fetchone()[0]
        return {"records": count, "size_kb": self.database.path.stat().st_size / 1024}
