"""Chat delegates language decisions to the model while tools retain data/write authority."""

import json
import time
from datetime import date

import pytest

from bnivoice360.config import Settings
from bnivoice360.container import create_services
from bnivoice360.domain import Role, VoiceRecord
from bnivoice360.services.agent.groq_client import GroqClient, ModelUnavailable
from bnivoice360.services.companion_state import CompanionState, Stage

TODAY = date(2026, 10, 14)
ROLE = Role.MANAGER.value


class ScriptedModel:
    def __init__(self, name, arguments, prose):
        self.name, self.arguments, self.prose = name, arguments, prose
        self.requests = []

    def choose_tools(self, messages, definitions, **kwargs):
        self.requests.append(json.loads(messages[1]["content"]))
        self.definitions = definitions
        if messages[-1]["role"] == "tool":
            return {"tool_calls": []}, {"tokens": 1}
        return {
            "tool_calls": [
                {
                    "id": "planned",
                    "type": "function",
                    "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
                }
            ]
        }, {"tokens": 1}

    def structured(self, system, data, schema, name, **kwargs):
        self.results = data["results"]
        return {"paragraphs": [{"text": self.prose, "sources": ["T1"]}]}, {"tokens": 1}


@pytest.fixture
def conversation(tmp_path, synthetic_masks):
    services = create_services(
        Settings(
            tmp_path / "voices.db",
            seed_initial_data=False,
            download_embeddings=False,
        )
    )
    services.repository.insert(
        [
            VoiceRecord("one", "Kartu kredit ditolak.", "2026-10-12", "X", "Negative"),
            VoiceRecord("two", "Kartu kredit tertunda.", "2026-10-01", "X", "Negative"),
            VoiceRecord("three", "QRIS lancar.", "2026-10-12", "PlayStore", "Positive"),
        ]
    )
    synthetic_masks(services.repository)
    return services.conversation


def chat(conversation, state, text, model, monkeypatch):
    monkeypatch.setattr(conversation, "_client", lambda key: model)
    return conversation.handle(state, text, "session", ROLE, api_key="synthetic", today=TODAY)


def test_model_interprets_arbitrary_wording_dates_and_generates_search_reply(
    conversation, monkeypatch
):
    model = ScriptedModel(
        "complaint_workflow",
        {
            "action": "display",
            "keyword": "kartu kredit",
            "period": "custom",
            "start": "2026-09-28",
            "end": "2026-10-14",
        },
        "Ada {{T1.total_results}} laporan soal kartu kredit. Silakan pilih yang ingin ditelusuri.",
    )
    state = CompanionState()
    text = "Bisa bantu tengok kendala kartu kredit selama tujuh belas hari belakangan?"
    reply = chat(conversation, state, text, model, monkeypatch)
    assert model.requests[0]["question"] == text
    assert state.query.start == date(2026, 9, 28)
    assert state.result_ids == ("one", "two") and not state.selected_ids
    assert reply == "Ada 2 laporan soal kartu kredit. Silakan pilih yang ingin ditelusuri. [T1]"
    assert "Daftar ditampilkan di bawah" not in reply
    assert {item["function"]["name"] for item in model.definitions} >= {
        "complaint_workflow",
        "compare_periods",
        "backlog_age",
    }


def test_application_does_not_rewrite_model_filters_from_input_keywords(conversation, monkeypatch):
    model = ScriptedModel(
        "volume_breakdown",
        {
            "group_by": "source",
            "keyword": "kartu kredit",
            "source": "X",
            "period": "all",
            "sentiment": "Negative",
        },
        "Ada {{T1.total}} laporan yang sesuai.",
    )
    state = CompanionState()
    chat(
        conversation,
        state,
        "Bandingkan dengan QRIS di PlayStore selama 7 hari, fokus kartu kredit di X sepanjang waktu.",
        model,
        monkeypatch,
    )
    assert model.results[0]["data"]["total"] == 2
    scope = model.results[0]["data"]["scope"]
    assert scope["keyword"] == "kartu kredit" and scope["source"] == "X"
    assert scope["period"] == "all"


def test_followup_context_selection_and_draft_never_commit(conversation, monkeypatch):
    state = CompanionState()
    model = ScriptedModel(
        "complaint_workflow",
        {
            "action": "display",
            "keyword": "kartu kredit",
            "period": "all",
        },
        "Silakan pilih keluhan untuk diperiksa.",
    )
    chat(conversation, state, "Lihat keluhan kartu kredit", model, monkeypatch)
    model = ScriptedModel(
        "complaint_workflow",
        {
            "action": "select",
            "indices": [2],
        },
        "Keluhan pilihan Anda siap ditelusuri.",
    )
    chat(conversation, state, "Ambil yang kedua dulu ya", model, monkeypatch)
    assert state.selected_ids == ("two",)
    assert model.requests[0]["recent_turns"][-1]["user"] == "Lihat keluhan kartu kredit"
    assert [item["number"] for item in model.requests[0]["conversation"]["visible_results"]] == [
        1,
        2,
    ]
    assert conversation.repository.load()["escalation_status"].eq("Open").all()
    assert state.proposal is None


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("execute_sql", {"sql": "DELETE FROM suara_nasabah"}),
        ("complaint_workflow", {"action": "commit"}),
        (
            "complaint_workflow",
            {"action": "details", "examination": "Petugas sudah memeriksa mutasi."},
        ),
    ],
)
def test_unauthorized_tools_and_invented_operator_notes_fail_closed(
    conversation, monkeypatch, name, arguments
):
    state = CompanionState()
    before = conversation.repository.database.path.read_bytes()
    reply = chat(
        conversation,
        state,
        "Tolong bantu tindak lanjut",
        ScriptedModel(name, arguments, "Sukses."),
        monkeypatch,
    )
    assert state.reply_kind == "blocked" and "Sukses." not in reply
    assert conversation.repository.database.path.read_bytes() == before


@pytest.mark.parametrize(
    "name,arguments,prose",
    [
        (
            "clarify_question",
            {"question": "Periode mana yang dimaksud?"},
            "Untuk menelusurinya, periode mana yang ingin Anda lihat?",
        ),
        (
            "decline_request",
            {"reason": "outside_scope"},
            "Saya bisa membantu menelusuri suara nasabah; prakiraan cuaca memerlukan sumber lain.",
        ),
    ],
)
def test_clarifications_and_refusals_are_model_written(
    conversation, monkeypatch, name, arguments, prose
):
    state = CompanionState()
    model = ScriptedModel(name, arguments, prose)
    reply = chat(conversation, state, "Bagaimana cuaca memengaruhi keluhan?", model, monkeypatch)
    assert model.requests and reply == prose + " [T1]"
    assert state.analysis_waiting == (name == "clarify_question")


def test_offline_chat_does_not_parse_input_but_typed_tools_remain_available(conversation):
    state = CompanionState()
    reply = conversation.handle(state, "Tampilkan issue kartu kredit", "session", ROLE, today=TODAY)
    assert "Groq" in reply and state.query is None
    conversation.execute_workflow(
        state, {"action": "display", "period": "all"}, "session", ROLE, today=TODAY
    )
    assert state.stage == Stage.RESULTS and state.total == 2


def test_failed_model_output_confirms_actual_completed_feed_display(conversation, monkeypatch):
    class Unavailable(ScriptedModel):
        def structured(self, *args, **kwargs):
            raise ModelUnavailable("provider_unavailable")

    state = CompanionState()
    reply = chat(
        conversation,
        state,
        "Tolong tampilkan keluhan",
        Unavailable(
            "complaint_workflow",
            {"action": "display", "period": "all"},
            "",
        ),
        monkeypatch,
    )
    assert state.result_ids and state.reply_kind == "fallback"
    assert "Daftar 2 keluhan" in reply and "[T1]" in reply
    assert state.analysis_sources[0]["tool"] == "complaint_workflow"


@pytest.mark.parametrize(
    "override,expected_model,reasoning_effort",
    [
        (None, "openai/gpt-oss-20b", "low"),
        ("qwen/qwen3.8-27b", "qwen/qwen3.8-27b", "none"),
    ],
)
def test_model_payload_supports_tools_and_separate_strict_output(
    monkeypatch, override, expected_model, reasoning_effort
):
    if override is None:
        monkeypatch.delenv("BNIVOICE360_AGENT_MODEL", raising=False)
    else:
        monkeypatch.setenv("BNIVOICE360_AGENT_MODEL", override)
    assert Settings().agent_model == expected_model
    client = GroqClient("synthetic", Settings().agent_model)
    requests = []

    def request(payload):
        requests.append(payload)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}

    monkeypatch.setattr(client, "_request", request)
    client.choose_tools([], [], deadline=time.monotonic() + 5)
    client.structured("system", {}, {"type": "object"}, "reply")
    assert requests[0]["reasoning_effort"] == requests[1]["reasoning_effort"] == reasoning_effort
    assert "tools" in requests[0] and "response_format" not in requests[0]
    assert requests[1]["response_format"]["json_schema"]["strict"]
    assert "tools" not in requests[1]
