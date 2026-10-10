"""Persistence regressions: seed safety, escalation, and transaction boundaries."""

import sqlite3

import pandas as pd
import pytest

from bnivoice360.config import Settings
from bnivoice360.container import create_services
from bnivoice360.domain import VoiceRecord


def test_initialization_preserves_existing_data(tmp_path):
    settings = Settings(tmp_path / "voices.db")
    first = create_services(settings)
    frame = first.repository.load()
    assert len(frame) == 96
    assert frame["source"].nunique() == 8
    record_id = frame.iloc[0]["suara_nasabah_id"]
    first.repository.escalate(record_id)

    second = create_services(settings)
    reloaded = second.repository.load().set_index("suara_nasabah_id")
    assert len(reloaded) == 96
    assert reloaded.loc[record_id, "is_escalated"] == 1


def test_escalation_is_persistent_and_idempotent(services):
    services.ingestion.ingest(
        pd.DataFrame(
            [
                {
                    "suara_nasabah_id": "one",
                    "content": "Transfer gagal",
                    "date_posted": "2026-10-10",
                    "source": "X",
                }
            ]
        )
    )
    assert services.repository.escalate("one") is True
    assert services.repository.escalate("one") is False
    assert services.repository.escalate("missing") is False
    assert services.repository.load().iloc[0]["is_escalated"] == 1


def test_failed_batch_rolls_back_previous_rows(services):
    valid = VoiceRecord("valid", "Hello", "2026-10-10", "X")
    invalid = VoiceRecord("invalid", None, "2026-10-10", "X")
    # A failure in a later row rolls back the whole batch.
    with pytest.raises(sqlite3.IntegrityError):
        services.repository.insert([valid, invalid])
    assert services.repository.load().empty


def test_transaction_closes_connection(services):
    with services.repository.database.transaction() as connection:
        connection.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
