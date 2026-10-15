"""Source masks survive persistence and are the only complaint text sent to models."""

import json
from datetime import date

import pandas as pd
import pytest

from bnivoice360.data.database import Database
from bnivoice360.data.seed import TOPIC_COLUMNS, prepare_seed, refresh_masked_content
from bnivoice360.domain import Role, VoiceRecord
from bnivoice360.services.agent.guardrails import GuardrailError
from bnivoice360.services.analysis_tools import AnalysisTools
from bnivoice360.services.companion_draft import draft_prose
from bnivoice360.services.companion_state import CompanionState
from bnivoice360.services.remote_content import masked_text

RAW = "PRIVATE_CUSTOMER melaporkan QRIS gagal dari rekening 1234567890123456."
MASKED = "[NASABAH] melaporkan QRIS gagal dari rekening [REKENING]."
TODAY = date(2026, 10, 15)


@pytest.fixture
def masked_service(services):
    services.repository.insert(
        [
            VoiceRecord(
                "masked-row",
                RAW,
                "2026-10-14",
                "X",
                "Negative",
                is_IT=1,
                is_Operasional=1,
                is_Urgent=1,
                topic_cluster="QRIS",
                masked_content=MASKED,
            )
        ]
    )
    return services


@pytest.mark.parametrize("value", [None, "", "  ", float("nan"), pd.NA])
def test_missing_mask_never_falls_back_to_original(value):
    with pytest.raises(GuardrailError, match="masked_content_unavailable"):
        masked_text({"content": RAW, "masked_content": value})


def test_ingestion_stores_original_and_mask_independently(services):
    result = services.ingestion.ingest(
        pd.DataFrame(
            [
                {
                    "suara_nasabah_id": "ingested",
                    "content": RAW,
                    "masked_content": MASKED,
                    "date_posted": "2026-10-14",
                    "source": "X",
                }
            ]
        )
    )
    assert result.inserted == 1
    row = services.repository.load().iloc[0]
    assert row["content"] == RAW and row["masked_content"] == MASKED


def test_df_clean_seed_keeps_source_mask():
    voices = pd.DataFrame(
        [
            {
                "data_id": "clean",
                "raw_content": RAW,
                "masked_content": MASKED,
                "source_platform": "X",
                "created_at": "2026-10-14",
                "author_id_or_name": "author",
                "sentiment": "Negative",
                "is_IT": 1,
                "is_Operasional": 1,
                "is_Bisnis": 0,
                "is_Urgent": 1,
            }
        ]
    )
    topics = pd.DataFrame(
        [{**dict.fromkeys(TOPIC_COLUMNS, ""), "topic_id": 0, "topic_title": "QRIS"}]
    )
    data = prepare_seed(
        voices, topics, pd.DataFrame([{"data_id": "clean", "topic_id": 0}]), now=pd.Timestamp(TODAY)
    )
    assert data.voices[0].content == RAW and data.voices[0].masked_content == MASKED


def test_mask_refresh_preserves_state_and_skips_mismatched_originals(masked_service):
    repository = masked_service.repository
    repository.escalate("masked-row")
    frame = pd.DataFrame(
        [
            {
                "data_id": "masked-row",
                "raw_content": RAW,
                "masked_content": "Updated [NASABAH] QRIS.",
            },
            {"data_id": "absent", "raw_content": RAW, "masked_content": MASKED},
        ]
    )
    result = refresh_masked_content(repository.database, frame)
    assert result["updated"] == 1 and result["unmatched"] == 1
    row = repository.load().iloc[0]
    assert row["content"] == RAW and row["escalation_status"] == "Escalated"
    assert row["masked_content"] == "Updated [NASABAH] QRIS."
    frame.loc[0, "raw_content"] = "Different original"
    frame.loc[0, "masked_content"] = "Wrong mask"
    assert refresh_masked_content(repository.database, frame)["content_mismatch"] == 1
    assert repository.load().iloc[0]["masked_content"] == "Updated [NASABAH] QRIS."


def test_mask_refresh_rejects_invalid_source_before_updates(masked_service):
    frame = pd.DataFrame(
        [
            {"data_id": "masked-row", "raw_content": RAW, "masked_content": "New mask"},
            {"data_id": "masked-row", "raw_content": RAW, "masked_content": "Duplicate"},
        ]
    )
    with pytest.raises(ValueError):
        refresh_masked_content(masked_service.repository.database, frame)
    assert masked_service.repository.load().iloc[0]["masked_content"] == MASKED


def test_additive_migration_never_copies_original_into_mask(tmp_path):
    path = tmp_path / "old.db"
    database = Database(path)
    database.initialize()
    with database.transaction() as connection:
        connection.execute("ALTER TABLE suara_nasabah DROP COLUMN masked_content")
        connection.execute(
            "INSERT INTO suara_nasabah (suara_nasabah_id,content,source,escalation_status) VALUES (?,?,?,?)",
            ("legacy", RAW, "X", "Escalated"),
        )
    database.initialize()
    database.initialize()
    with database.transaction() as connection:
        row = connection.execute("SELECT * FROM suara_nasabah").fetchone()
        assert row["content"] == RAW and row["masked_content"] is None
        assert row["escalation_status"] == "Escalated"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4


def test_examples_and_workflow_context_only_expose_mask(masked_service):
    conversation = masked_service.conversation
    api = AnalysisTools(conversation.batch, TODAY, "Contoh keluhan", {}, {})
    report = api.execute("complaint_examples", {"period": "all", "limit": 5}, "T1")
    assert report["examples"][0]["excerpt"] == MASKED
    assert "PRIVATE_CUSTOMER" not in json.dumps(report)
    state = CompanionState()
    conversation.execute_workflow(
        state,
        {"action": "display", "period": "all", "keyword": "QRIS"},
        "masked-session",
        Role.MANAGER.value,
        today=TODAY,
    )
    context = conversation.model_context(state)
    assert context["visible_results"][0]["excerpt"] == MASKED
    assert "PRIVATE_CUSTOMER" not in json.dumps(context)
    conversation.execute_workflow(
        state,
        {"action": "select", "match_phrase": MASKED},
        "masked-session",
        Role.MANAGER.value,
        today=TODAY,
    )
    assert state.selected_ids == ("masked-row",)


def test_planner_and_answer_payload_exclude_original_and_legacy_history(masked_service):
    seen = []

    class Probe:
        used = False

        def choose_tools(self, messages, definitions, **kwargs):
            seen.append(json.loads(json.dumps(messages)))
            if self.used:
                return {"tool_calls": []}, {"tokens": 1}
            self.used = True
            return {
                "tool_calls": [
                    {
                        "id": "masked-examples",
                        "type": "function",
                        "function": {
                            "name": "complaint_examples",
                            "arguments": json.dumps({"period": "all", "limit": 5}),
                        },
                    }
                ]
            }, {"tokens": 1}

        def structured(self, system, data, schema, name, **kwargs):
            seen.append(data)
            return {"paragraphs": [{"text": "{{T1.examples.0.excerpt}}", "sources": ["T1"]}]}, {
                "tokens": 1
            }

    state = CompanionState(recent_turns=[{"user": "Contoh sebelumnya", "assistant": RAW}])
    answer = masked_service.conversation.analysis.answer(
        state, "Tampilkan contoh keluhan QRIS", "masked-session", TODAY, Probe()
    )
    assert MASKED in answer and "PRIVATE_CUSTOMER" not in json.dumps(seen)
    assert "1234567890123456" not in json.dumps(seen)
    assert state.remote_content_version == 1


def test_missing_mask_omits_remote_examples(masked_service):
    with masked_service.repository.database.transaction() as connection:
        connection.execute("UPDATE suara_nasabah SET masked_content=NULL")
    api = AnalysisTools(masked_service.batch_agent, TODAY, "Contoh", {}, {})
    report = api.execute("complaint_examples", {"period": "all", "limit": 5}, "T1")
    assert report["total"] == 1 and report["returned"] == 0 and report["omitted_unsafe_texts"] == 1
    assert "PRIVATE_CUSTOMER" not in json.dumps(report)


@pytest.mark.parametrize("available", [True, False])
def test_companion_draft_payload_uses_mask_and_missing_mask_blocks_request(
    masked_service, available
):
    with masked_service.repository.database.transaction() as connection:
        row = dict(connection.execute("SELECT * FROM suara_nasabah").fetchone())
    if not available:
        row["masked_content"] = None
    calls = []

    class Probe:
        def structured(self, system, data, schema, name, **kwargs):
            calls.append(data)
            assert data["complaints"][0]["data"]["content"] == MASKED
            assert "PRIVATE_CUSTOMER" not in json.dumps(data)
            return {
                "summary": [{"text": "{{V1.content}}", "sources": ["V1"]}],
                "suggestions": [{"text": "Periksa transaksi yang dilaporkan.", "sources": ["V1"]}],
            }, {"tokens": 1}

    state = CompanionState(selected_ids=("masked-row",), team="IT", request="Periksa transaksi.")
    summary, _, mode, reason = draft_prose(
        {"masked-row": row}, state, Probe(), masked_service.batch_agent.audit, "masked-session"
    )
    assert len(calls) == int(available)
    assert mode == ("groq" if available else "manual")
    if available:
        assert MASKED in summary[0]
    else:
        assert reason == "masked_content_unavailable"
