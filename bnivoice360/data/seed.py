"""Validate and initialize the two-table database from CSV exports."""

import argparse
import io
import json
import ssl
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from bnivoice360.config import SENTIMENTS, SOURCES, Settings
from bnivoice360.data.database import Database
from bnivoice360.data.repository import INSERT_VOICE, VoiceRepository
from bnivoice360.domain import VoiceRecord
from bnivoice360.services.ingestion import binary_flag, nullable_text

TOPIC_COLUMNS = (
    "topic_id",
    "topic_count",
    "topic_share_pct",
    "bertopic_name",
    "keywords",
    "keyword_scores",
    "representative_docs",
    "topic_title",
    "topic_summary",
    "customer_issue",
    "labeling_status",
)
VOICE_COLUMNS = (
    "data_id",
    "source_platform",
    "raw_content",
    "author_id_or_name",
    "created_at",
    "sentiment",
    "is_IT",
    "is_Operasional",
    "is_Bisnis",
    "is_Urgent",
)


@dataclass(frozen=True)
class SeedData:
    topics: tuple[dict, ...]
    voices: tuple[VoiceRecord, ...]
    cutoff: str


def read_csv(location: str) -> pd.DataFrame:
    if location.startswith(("https://", "http://")):
        # Avoid inherited SSLKEYLOGFILE settings writing TLS secrets to unrelated paths.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_default_certs()
        with urlopen(location, timeout=60, context=context) as response:
            content = response.read()
        if content.lstrip().lower().startswith((b"<!doctype html", b"<html")):
            raise ValueError("CSV source returned an HTML page. Check Google sharing permissions.")
        source = io.BytesIO(content)
    else:
        source = Path(location).expanduser()
    frame = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("CSV contains duplicate column names.")
    return frame


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {', '.join(sorted(missing))}")


def integer(value: object, name: str) -> int:
    try:
        number = float(value)
        if not number.is_integer():
            raise ValueError
        return int(number)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer: {value!r}") from exc


def prepare_seed(
    voices: pd.DataFrame,
    topics: pd.DataFrame,
    mapping: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> SeedData:
    voices, topics, mapping = (frame.copy() for frame in (voices, topics, mapping))
    for frame in (voices, topics, mapping):
        frame.columns = frame.columns.astype(str).str.strip()
    require_columns(voices, VOICE_COLUMNS, "voices")
    require_columns(topics, TOPIC_COLUMNS, "topics")
    require_columns(mapping, ("data_id", "topic_id"), "mapping")
    if voices.empty:
        raise ValueError("Voice source is empty; refusing initialization.")
    for name, frame, key in (
        ("voices", voices, "data_id"),
        ("topics", topics, "topic_id"),
        ("mapping", mapping, "data_id"),
    ):
        frame[key] = frame[key].map(nullable_text)
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{name}: blank or duplicate {key}")
    for frame in (topics, mapping):
        frame["topic_id"] = frame["topic_id"].map(lambda value: integer(value, "topic_id"))
        if frame["topic_id"].lt(-1).any():
            raise ValueError("Source topic_id must be -1 or a nonnegative integer.")
    if topics["topic_id"].duplicated().any():
        raise ValueError("topics: duplicate normalized topic_id")
    if not set(mapping["data_id"]).issubset(set(voices["data_id"])):
        raise ValueError("Mapping references data_id values absent from the voice source.")
    topic_rows = []
    for _, row in topics.iterrows():
        values = {column: nullable_text(row[column]) for column in TOPIC_COLUMNS}
        values["topic_id"] = int(row["topic_id"])
        if values["topic_count"] is not None:
            values["topic_count"] = integer(values["topic_count"], "topic_count")
        if values["topic_share_pct"] is not None:
            values["topic_share_pct"] = float(values["topic_share_pct"])
        topic_rows.append(values)
    if -1 not in set(topics["topic_id"]):
        topic_rows.append(
            {**dict.fromkeys(TOPIC_COLUMNS), "topic_id": -1, "topic_title": "Outlier"}
        )
    unknown = set(mapping["topic_id"]) - {row["topic_id"] for row in topic_rows}
    if unknown:
        raise ValueError(f"Mapping references missing topic IDs: {sorted(unknown)}")
    assignments = dict(zip(mapping["data_id"], mapping["topic_id"], strict=True))
    now = pd.Timestamp.now(tz="Asia/Jakarta") if now is None else pd.Timestamp(now)
    now = now.tz_localize("Asia/Jakarta") if now.tzinfo is None else now.tz_convert("Asia/Jakarta")
    cutoff = now.tz_localize(None) - pd.DateOffset(months=1)
    records = []
    for position, (_, row) in enumerate(voices.iterrows(), start=2):
        try:
            content = nullable_text(row["raw_content"])
            if content is None:
                raise ValueError("raw_content cannot be blank")
            date_text = nullable_text(row["created_at"])
            posted = pd.to_datetime(date_text, errors="coerce") if date_text else pd.NaT
            if date_text and pd.isna(posted):
                raise ValueError("created_at is invalid")
            if not pd.isna(posted) and posted.tzinfo is not None:
                posted = posted.tz_convert("Asia/Jakarta").tz_localize(None)
            source = nullable_text(row["source_platform"])
            source = "Laporan Nasabah" if source == "Internal_CRM" else source
            sentiment = nullable_text(row["sentiment"])
            sentiment = "Neutral" if sentiment == "Netral" else sentiment
            if source not in SOURCES or sentiment not in SENTIMENTS:
                raise ValueError(f"Unknown source/sentiment: {source}/{sentiment}")
            flags = {
                key: binary_flag(row[key])
                for key in ("is_IT", "is_Bisnis", "is_Operasional", "is_Urgent")
            }
            if any(value is None for value in flags.values()):
                raise ValueError("Seed category and urgency flags cannot be blank")
            records.append(
                VoiceRecord(
                    suara_nasabah_id=row["data_id"],
                    content=content,
                    masked_content=nullable_text(row.get("masked_content")),
                    date_posted=None if pd.isna(posted) else posted.isoformat(),
                    source=source,
                    sentiment=sentiment,
                    **flags,
                    escalation_status="Escalated" if pd.isna(posted) or posted < cutoff else "Open",
                    topic_id=assignments.get(row["data_id"]),
                    author_id_or_name=nullable_text(row["author_id_or_name"]),
                )
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"voices CSV row {position}: {exc}") from exc
    return SeedData(tuple(topic_rows), tuple(records), cutoff.isoformat())


def load_seed(settings: Settings, now: pd.Timestamp | None = None) -> SeedData:
    return prepare_seed(
        read_csv(settings.voice_csv),
        read_csv(settings.topic_csv),
        read_csv(settings.mapping_csv),
        now,
    )


def initialize_data(database: Database, data: SeedData, overwrite: bool = False) -> dict:
    backup = None
    with database.transaction() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute("SELECT COUNT(*) FROM suara_nasabah").fetchone()[0]
        if existing and not overwrite:
            return {"status": "preserved", "voices": existing}
        if overwrite:
            backup = str(database.backup())
            connection.execute("DELETE FROM suara_nasabah")
            connection.execute("DELETE FROM topic_detail")
        columns = ", ".join(TOPIC_COLUMNS)
        placeholders = ", ".join(f":{column}" for column in TOPIC_COLUMNS)
        connection.executemany(
            f"INSERT INTO topic_detail ({columns}) VALUES ({placeholders}) "
            "ON CONFLICT(topic_id) DO UPDATE SET "
            + ", ".join(
                f"{column}=excluded.{column}" for column in TOPIC_COLUMNS if column != "topic_id"
            ),
            data.topics,
        )
        connection.executemany(INSERT_VOICE, [asdict(record) for record in data.voices])
    return {
        "status": "initialized",
        "voices": len(data.voices),
        "topics": len(data.topics),
        "outliers": sum(record.topic_id == -1 for record in data.voices),
        "unlabeled": sum(record.topic_id is None for record in data.voices),
        "undated": sum(record.date_posted is None for record in data.voices),
        "open": sum(record.escalation_status == "Open" for record in data.voices),
        "escalated": sum(record.escalation_status == "Escalated" for record in data.voices),
        "resolved": sum(record.escalation_status == "Resolved" for record in data.voices),
        "cutoff": data.cutoff,
        "backup": backup,
    }


def refresh_masked_content(database: Database, voices: pd.DataFrame) -> dict:
    """Refresh only masks for matching IDs and original text, preserving reviewed ticket state."""
    require_columns(voices, ("data_id", "raw_content", "masked_content"), "voices")
    records = []
    identifiers = set()
    for position, (_, row) in enumerate(voices.iterrows(), start=2):
        identifier = nullable_text(row["data_id"])
        original = nullable_text(row["raw_content"])
        if identifier is None or identifier in identifiers or original is None:
            raise ValueError(f"voices CSV row {position}: blank/duplicate ID or blank raw_content")
        identifiers.add(identifier)
        records.append((identifier, original, nullable_text(row["masked_content"])))
    counts = dict(updated=0, unchanged=0, missing_mask=0, unmatched=0, content_mismatch=0)
    with database.transaction() as connection:
        connection.execute("BEGIN IMMEDIATE")
        for identifier, original, masked in records:
            existing = connection.execute(
                "SELECT content,masked_content FROM suara_nasabah WHERE suara_nasabah_id=?",
                (identifier,),
            ).fetchone()
            if existing is None:
                counts["unmatched"] += 1
                continue
            if existing["content"] != original:
                counts["content_mismatch"] += 1
                continue
            counts["missing_mask"] += int(masked is None)
            if existing["masked_content"] == masked:
                counts["unchanged"] += 1
                continue
            connection.execute(
                "UPDATE suara_nasabah SET masked_content=? WHERE suara_nasabah_id=?",
                (masked, identifier),
            )
            counts["updated"] += 1
    return {"status": "masks_refreshed", "source_rows": len(records), **counts}


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", dest="overwrite", action="store_true")
    mode.add_argument("--preserve", dest="overwrite", action="store_false")
    mode.add_argument("--refresh-masked-content", action="store_true")
    parser.set_defaults(overwrite=settings.overwrite_initialization)
    parser.add_argument("--as-of", help="Override initialization time (Jakarta), e.g. 2026-10-11")
    args = parser.parse_args()
    database = Database(settings.database_path)
    if args.refresh_masked_content:
        voices = read_csv(settings.voice_csv)
        require_columns(voices, ("data_id", "raw_content", "masked_content"), "voices")
        database.initialize()
        print(json.dumps(refresh_masked_content(database, voices), indent=2))
        return
    # Preserve mode needs neither network access nor source validation for an existing DB.
    if database.path.exists() and not args.overwrite:
        database.initialize()
        repository = VoiceRepository(database)
        if not repository.is_empty():
            print(json.dumps({"status": "preserved", **repository.health()}, indent=2))
            return
    data = load_seed(settings, pd.Timestamp(args.as_of) if args.as_of else None)
    database.initialize()
    print(json.dumps(initialize_data(database, data, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
