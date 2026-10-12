"""Evaluation permissions, isolation, reproducibility, telemetry, persistence and real UI flows."""

import json
import sqlite3
from datetime import date

import pytest

from bnivoice360.domain import Role, VoiceRecord
from bnivoice360.pages.companion_evaluation import csv_bytes
from bnivoice360.services.agent.groq_client import GroqClient
from bnivoice360.services.companion_state import CompanionState
from bnivoice360.services.evaluation_store import utc_window
from bnivoice360.services.evaluation_suite import CASES, EvaluationMockClient

ADMIN = Role.DATA_ADMIN.value
MANAGER = Role.MANAGER.value


@pytest.mark.parametrize("mode", ["Local", "Mock"])
def test_all_scenarios_pass_and_preserve_operational_database(services, mode, monkeypatch):
    monkeypatch.setattr(
        GroqClient, "_request", lambda *args: pytest.fail("offline evaluation reached provider")
    )
    services.repository.insert(
        [
            VoiceRecord(
                "production-sentinel", "Feedback operasional asli.", "2026-10-12", "X", "Negative"
            )
        ]
    )
    before = services.repository.database.path.read_bytes()
    progress = []
    run_id = services.evaluation.run(
        ADMIN, mode, [case.case_id for case in CASES], progress=lambda *args: progress.append(args)
    )
    runs = services.evaluation.store.runs(ADMIN)
    results = services.evaluation.store.results(ADMIN, run_id)
    assert len(results) == len(CASES)
    assert all(result["status"] == "passed" for result in results), [
        (r["case_id"], r["checks"], r["error_code"]) for r in results if r["status"] != "passed"
    ]
    assert runs[0]["summary"]["passed"] == len(CASES) and runs[0]["status"] == "completed"
    assert runs[0]["config"]["synthetic"] and runs[0]["code_digest"] and runs[0]["prompt_digest"]
    assert services.repository.database.path.read_bytes() == before
    assert progress[-1][0] == len(CASES)
    assert "production-sentinel" not in json.dumps(results)
    if mode == "Mock":
        assert runs[0]["summary"]["tokens"] == 0
        assert any(
            row["event"] == "model_call_finished" for result in results for row in result["trace"]
        )


def test_all_evaluation_operations_require_admin(services):
    evaluation = services.evaluation
    store = evaluation.store
    calls = [
        lambda: evaluation.scenarios(MANAGER),
        lambda: evaluation.run(MANAGER, "Local", ["display"]),
        lambda: evaluation.compare(MANAGER, "a", "b"),
        lambda: store.runs(MANAGER),
        lambda: store.results(MANAGER, "a"),
        lambda: store.logs(MANAGER, date(2026, 10, 12), date(2026, 10, 12)),
        lambda: store.log_options(MANAGER, date(2026, 10, 12), date(2026, 10, 12)),
        lambda: store.overview(MANAGER, date(2026, 10, 12), date(2026, 10, 12)),
        lambda: store.interaction_events(MANAGER, "a"),
        lambda: store.review(MANAGER, "a", "b", "Sesuai", ""),
        lambda: store.start_run(MANAGER, {}),
        lambda: store.save_result(MANAGER, "a", {}),
        lambda: store.finish_run(MANAGER, "a", {}),
    ]
    for call in calls:
        with pytest.raises(PermissionError):
            call()
    assert not store.path.exists()


def test_invalid_modes_cases_and_live_configuration_are_rejected(services):
    for mode, ids in [
        ("invalid", ["display"]),
        ("Local", []),
        ("Local", ["unknown"]),
        ("Mock", ["display", "display"]),
    ]:
        with pytest.raises(ValueError):
            services.evaluation.run(ADMIN, mode, ids)
    with pytest.raises(ValueError):
        services.evaluation.run(ADMIN, "Live Groq", ["draft"])
    with pytest.raises(ValueError):
        services.evaluation.run(
            ADMIN, "Live Groq", ["display", "draft", "revision", "analysis"], api_key="test"
        )
    assert not services.evaluation.store.path.exists()


def test_review_persists_separately_and_comparison_detects_different_modes(services):
    evaluation = services.evaluation
    left = evaluation.run(ADMIN, "Local", ["display", "draft"])
    evaluation.store.review(
        ADMIN, left, "draft", "Sebagian sesuai", "Perlu ringkasan lebih spesifik."
    )
    recreated = type(evaluation)(services.batch_agent.settings)
    result = next(row for row in recreated.store.results(ADMIN, left) if row["case_id"] == "draft")
    assert result["review_rating"] == "Sebagian sesuai" and result["reviewed_at"]
    assert result["status"] == "passed"  # Human review never rewrites automated checks.
    right = evaluation.run(ADMIN, "Mock", ["display", "draft"])
    comparison = evaluation.compare(ADMIN, left, right)
    assert not comparison["comparable"] and len(comparison["rows"]) == 2
    same_mode = evaluation.run(ADMIN, "Local", ["display", "draft"])
    assert evaluation.compare(ADMIN, left, same_mode)["comparable"]
    with pytest.raises(ValueError):
        evaluation.store.review(ADMIN, left, "absent", "Sesuai", "")
    with pytest.raises(ValueError):
        evaluation.store.review(ADMIN, left, "draft", "invalid", "")


def test_failed_grounding_is_stored_with_expected_actual_and_full_original(services, monkeypatch):
    original = EvaluationMockClient.structured

    def bad(self, system, data, schema, name, **kwargs):
        if name == "complaint_draft":
            return {
                "summary": [{"text": "Penyebabnya adalah kerusakan server.", "sources": ["V1"]}],
                "suggestions": [{"text": "Periksa gejala.", "sources": ["V1"]}],
            }, {"tokens": 1}
        return original(self, system, data, schema, name, **kwargs)

    monkeypatch.setattr(EvaluationMockClient, "structured", bad)
    run = services.evaluation.run(ADMIN, "Mock", ["draft"])
    result = services.evaluation.store.results(ADMIN, run)[0]
    assert result["status"] == "failed" and result["actual"]["originals_present"]
    assert any(check["check"] == "draft_mode" and not check["passed"] for check in result["checks"])
    assert "kerusakan server" not in result["actual"]["draft"]


def test_live_mode_integration_with_simulated_provider_never_persists_key(services, monkeypatch):
    monkeypatch.setattr(
        GroqClient,
        "structured",
        lambda self, *args, **kwargs: EvaluationMockClient().structured(*args, **kwargs),
    )
    key = "gsk_fake_test_credential_not_a_real_key"
    run = services.evaluation.run(ADMIN, "Live Groq", ["draft"], api_key=key)
    results = services.evaluation.store.results(ADMIN, run)
    assert results[0]["status"] == "passed" and results[0]["actual"]["draft_mode"] == "groq"
    assert key not in json.dumps(
        results
    ) and key not in services.evaluation.store.path.read_bytes().decode(errors="ignore")


def test_utc_filters_exact_metrics_no_token_double_count_and_malformed_metadata(services):
    store, audit = services.evaluation.store, services.batch_agent.audit
    with audit.transaction() as connection:
        samples = [
            (
                "2026-10-11T16:59:59+00:00",
                "interaction_finished",
                {"status": "ok", "duration_ms": 999},
            ),
            (
                "2026-10-11T17:00:00+00:00",
                "interaction_finished",
                {"status": "ok", "duration_ms": 100},
            ),
            (
                "2026-10-12T01:00:00+00:00",
                "interaction_finished",
                {"status": "fallback", "duration_ms": 200},
            ),
            (
                "2026-10-12T02:00:00+00:00",
                "interaction_finished",
                {"status": "ok", "duration_ms": 300},
            ),
            (
                "2026-10-12T03:00:00+00:00",
                "model_call_finished",
                {"status": "ok", "tokens": 12, "duration_ms": 50},
            ),
            (
                "2026-10-12T03:01:00+00:00",
                "analysis_finished",
                {"status": "ok", "tokens": 12, "duration_ms": 50},
            ),
            ("2026-10-12T03:02:00+00:00", "companion_model_fallback", {"reason": "unavailable"}),
        ]
        connection.executemany(
            "INSERT INTO audit.events(timestamp,session_id,event,metadata) VALUES (?,?,?,?)",
            [
                (timestamp, "s", event, json.dumps(metadata))
                for timestamp, event, metadata in samples
            ],
        )
        connection.execute(
            "INSERT INTO audit.events(timestamp,session_id,event,metadata) VALUES (?,?,?,?)",
            ("2026-10-12T04:00:00+00:00", "s", "legacy", "[]"),
        )
    day = date(2026, 10, 12)
    metrics = store.overview(ADMIN, day, day)
    assert metrics["interactions"] == 3 and metrics["tokens"] == 12
    assert metrics["p95_ms"] == 300 and metrics["ok_percent"] == pytest.approx(200 / 3)
    assert metrics["fallback_events"] == 1
    logs = store.logs(ADMIN, day, day, limit=2)
    assert logs["total"] == 7 and len(logs["rows"]) == 2
    assert logs["rows"][0]["metadata"] == {"invalid_metadata": True}
    assert store.logs(ADMIN, day, day, session="' OR 1=1 --")["total"] == 0
    assert store.logs(ADMIN, day, day, status="fallback")["total"] == 2
    assert utc_window(day, day)[0] == "2026-10-11T17:00:00+00:00"


def test_new_interaction_groups_nested_draft_and_records_model_usage(services, monkeypatch):
    settings = services.batch_agent.settings
    object.__setattr__(settings, "companion_remote_enabled", True)
    services.repository.insert(
        [VoiceRecord("c1", "wondr gagal login.", "2026-10-12", "X", "Negative", 1, 0, 0, 1)]
    )
    monkeypatch.setattr(
        GroqClient,
        "structured",
        lambda self, *args, **kwargs: EvaluationMockClient().structured(*args, **kwargs),
    )
    state = CompanionState()
    engine = services.conversation
    engine.handle(state, "Tampilkan issue wondr semua periode", "session", MANAGER)
    engine.handle(
        state,
        "Escalate nomor 1 ke IT; permintaan: periksa login",
        "session",
        MANAGER,
        api_key="test",
    )
    logs = services.evaluation.store.logs(ADMIN, engine.today(), engine.today(), limit=500)["rows"]
    finished = [row for row in logs if row["event"] == "interaction_finished"]
    assert len(finished) == 2
    timeline = services.evaluation.store.interaction_events(ADMIN, finished[0]["interaction_id"])
    assert any(row["event"] == "companion_draft_ready" for row in timeline)
    assert any(row["event"] == "model_call_finished" for row in timeline)
    assert len({row["metadata"]["interaction_id"] for row in timeline}) == 1
    assert "wondr gagal login" not in json.dumps(timeline)


def test_interaction_exception_records_only_class_and_context_is_reset(services):
    audit = services.batch_agent.audit
    with pytest.raises(RuntimeError):
        with audit.interaction("s", "test", ADMIN, "local"):
            raise RuntimeError("secret-value-never-log")
    assert audit.current_session() is None
    with sqlite3.connect(audit.path) as connection:
        rows = connection.execute("SELECT event,metadata FROM events").fetchall()
    assert "secret-value-never-log" not in str(rows)
    assert json.loads(rows[-1][1])["status"] == "failed"


def test_export_neutralizes_spreadsheet_formulas():
    import pandas as pd

    assert "'=HYPERLINK" in csv_bytes(pd.DataFrame({"note": ["=HYPERLINK(test)"]})).decode(
        "utf-8-sig"
    )


def test_admin_page_run_review_comparison_and_role_switch(streamlit_app):
    app = streamlit_app
    assert "Companion Evaluation" not in app.sidebar.radio(key="current_page").options
    app.sidebar.selectbox(key="selected_role").set_value(ADMIN).run()
    app.sidebar.radio(key="current_page").set_value("Companion Evaluation").run()
    assert not app.exception
    assert app.button(key="eval_run")
    # Rendering alone does not create or execute an evaluation.
    assert not app.selectbox(key="eval_mode").value == "Live Groq"
    app.multiselect(key="eval_cases_Local").set_value(["display", "draft"]).run()
    app.button(key="eval_run").click().run()
    assert not app.exception
    first = app.session_state["eval_selected_run"]
    app.selectbox(key="eval_result_case_" + first).set_value("draft").run()
    app.selectbox(key="eval_review_rating_" + first + "_draft").set_value("Sebagian sesuai")
    app.text_area(key="eval_review_note_" + first + "_draft").set_value(
        "Perlu saran yang lebih spesifik."
    )
    app.button(key="eval_save_review").click().run()
    assert not app.exception
    app.selectbox(key="eval_mode").set_value("Mock").run()
    app.multiselect(key="eval_cases_Mock").set_value(["display", "draft"]).run()
    app.button(key="eval_run").click().run()
    assert not app.exception and app.selectbox(key="eval_compare_left")
    app.sidebar.selectbox(key="selected_role").set_value(MANAGER).run()
    assert not app.exception and app.sidebar.radio(key="current_page").value == "Home"
    assert "Companion Evaluation" not in app.sidebar.radio(key="current_page").options


def test_page_guard_rejects_direct_manager_render(services, monkeypatch):
    from bnivoice360.pages import companion_evaluation

    monkeypatch.setattr(companion_evaluation.st, "session_state", {"selected_role": MANAGER})
    errors = []
    monkeypatch.setattr(companion_evaluation.st, "error", errors.append)
    companion_evaluation.render(services)
    assert errors and not services.evaluation.store.path.exists()
