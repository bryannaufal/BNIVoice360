"""Validate and normalize incoming data before it reaches persistence."""

import uuid

import pandas as pd

from bnivoice360.config import ESCALATION_STATUSES, SENTIMENTS, SOURCES
from bnivoice360.data.repository import VoiceRepository
from bnivoice360.domain import IngestionResult, VoiceRecord


def nullable_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def binary_flag(value: object, default: int | None = None) -> int | None:
    if nullable_text(value) is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Category and urgency flags must be 0 or 1.") from exc
    if number not in (0, 1):
        raise ValueError("Category and urgency flags must be 0 or 1.")
    return int(number)


class IngestionService:
    def __init__(self, repository: VoiceRepository):
        self.repository = repository

    @staticmethod
    def validate_columns(frame: pd.DataFrame, forced_source: str | None = None) -> None:
        required = {"content", "date_posted"}
        if forced_source is None:
            required.add("source")
        missing = required - set(frame.columns)
        if missing:
            raise ValueError("Kolom wajib belum tersedia: " + ", ".join(sorted(missing)))
        if forced_source is not None and forced_source not in SOURCES:
            raise ValueError(f"Unknown source: {forced_source}")

    def ingest(self, frame: pd.DataFrame, forced_source: str | None = None) -> IngestionResult:
        self.validate_columns(frame, forced_source)
        records: list[VoiceRecord] = []
        errors: list[str] = []
        topic_ids = self.repository.topic_ids()
        for position, (_, row) in enumerate(frame.iterrows(), start=2):
            try:
                record = self._normalize_row(row, forced_source)
                if record.topic_id is not None and record.topic_id not in topic_ids:
                    raise ValueError(f"Unknown topic_id: {record.topic_id}")
                records.append(record)
            except ValueError as exc:
                errors.append(f"CSV row {position}: {exc}")
        inserted, duplicates = self.repository.insert(records)
        return IngestionResult(inserted, len(errors) + duplicates, tuple(errors))

    @staticmethod
    def _normalize_row(row: pd.Series, forced_source: str | None) -> VoiceRecord:
        content = nullable_text(row.get("content"))
        if content is None:
            raise ValueError("Content cannot be blank.")
        posted = pd.to_datetime(row.get("date_posted"), errors="coerce")
        if pd.isna(posted):
            raise ValueError("date_posted must be a valid date.")
        # Store local naive Jakarta time, matching the existing database schema.
        if posted.tzinfo is not None:
            posted = posted.tz_convert("Asia/Jakarta").tz_localize(None)
        source = forced_source or nullable_text(row.get("source"))
        if source not in SOURCES:
            raise ValueError(f"Unknown source: {source}")
        topic_text = nullable_text(row.get("topic_id"))
        topic_id = None
        if topic_text is not None:
            try:
                number = float(topic_text)
                if not number.is_integer():
                    raise ValueError
                topic_id = int(number)
            except (ValueError, OverflowError) as exc:
                raise ValueError("topic_id must be an integer.") from exc
        sentiment = nullable_text(row.get("sentiment"))
        if sentiment is not None and sentiment not in SENTIMENTS:
            raise ValueError(f"Unknown sentiment: {sentiment}")
        status = nullable_text(row.get("escalation_status")) or "Open"
        if status not in ESCALATION_STATUSES:
            raise ValueError("escalation_status must be Open, Escalated, or Resolved.")
        return VoiceRecord(
            suara_nasabah_id=nullable_text(row.get("suara_nasabah_id"))
            or f"VOC-{uuid.uuid4().hex[:12].upper()}",
            content=content,
            date_posted=posted.isoformat(),
            source=source,
            sentiment=sentiment,
            is_IT=binary_flag(row.get("is_IT")),
            is_Bisnis=binary_flag(row.get("is_Bisnis")),
            is_Operasional=binary_flag(row.get("is_Operasional")),
            is_Urgent=binary_flag(row.get("is_Urgent")),
            escalation_status=status,
            topic_cluster=nullable_text(row.get("topic_cluster")),
            topic_id=topic_id,
            author_id_or_name=nullable_text(row.get("author_id_or_name")),
        )
