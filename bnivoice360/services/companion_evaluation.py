"""Admin-only controlled evaluation of the real Companion on isolated synthetic databases."""

import hashlib
import tempfile
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from bnivoice360.domain import Role
from bnivoice360.services.agent.audit import fingerprint, utc_now
from bnivoice360.services.agent.telemetry import AuditedModelClient
from bnivoice360.services.agent.workflow import ReviewRequired
from bnivoice360.services.companion_state import CompanionState
from bnivoice360.services.evaluation_store import EvaluationStore, decode_json, require_admin
from bnivoice360.services.evaluation_suite import (
    CASES,
    SCENARIO_PLANS,
    SUITE_VERSION,
    EvaluationMockClient,
    fixture_records,
)

MODES = ("Local", "Mock", "Live Groq")


def source_digest(names: tuple[str, ...]) -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


class CompanionEvaluation:
    def __init__(self, settings):
        self.settings = settings
        self.store = EvaluationStore(settings.database_path)

    def scenarios(self, role: str) -> list[dict]:
        require_admin(role)
        return [
            {"case_id": case.case_id, "label": case.label, "category": case.category}
            for case in CASES
        ]

    def run(self, role: str, mode: str, case_ids: list[str], *, api_key="", progress=None) -> str:
        require_admin(role)
        by_id = {case.case_id: case for case in CASES}
        if (
            mode not in MODES
            or not case_ids
            or len(set(case_ids)) != len(case_ids)
            or not set(case_ids) <= by_id.keys()
        ):
            raise ValueError("Pilih mode dan skenario evaluasi yang valid.")
        if mode == "Live Groq" and not api_key:
            raise ValueError("Live Groq memerlukan API key.")
        from bnivoice360.services.companion_workflow import CompanionConversation

        today = CompanionConversation.today()
        records = fixture_records(today)
        run_id = str(uuid.uuid4())
        config = {
            "case_ids": case_ids,
            "as_of": today.isoformat(),
            "synthetic": True,
            "configured_model": self.settings.agent_model,
            "remote_enabled": mode != "Local",
            "agent_similarity_threshold": self.settings.agent_similarity_threshold,
            "agent_candidate_limit": self.settings.agent_candidate_limit,
        }
        run = {
            "run_id": run_id,
            "started_at": utc_now(),
            "mode": mode,
            "model": self.settings.agent_model if mode == "Live Groq" else mode.lower(),
            "code_digest": source_digest(
                (
                    "companion_workflow.py",
                    "companion_intent.py",
                    "companion_state.py",
                    "companion_draft.py",
                    "remote_content.py",
                    "analysis_agent.py",
                    "analysis_tools.py",
                    "evaluation_suite.py",
                    "companion_evaluation.py",
                    "agent/groq_client.py",
                    "agent/telemetry.py",
                    "agent/audit.py",
                    "agent/workflow.py",
                    "analytics.py",
                )
            ),
            "prompt_digest": source_digest(
                (
                    "companion_intent.py",
                    "companion_draft.py",
                    "analysis_agent.py",
                    "companion_workflow.py",
                )
            ),
            "dataset_digest": fingerprint({"records": [asdict(record) for record in records]}),
            "suite_version": SUITE_VERSION,
            "config": config,
        }
        self.store.start_run(role, run)
        results = []
        try:
            for index, case_id in enumerate(case_ids):
                case = by_id[case_id]
                if progress:
                    progress(index, len(case_ids), case.label)
                result = self._run_case(case, mode, records, today, api_key)
                self.store.save_result(role, run_id, result)
                results.append(result)
            summary = self._summary(results)
            self.store.finish_run(role, run_id, summary)
        except Exception as exc:
            self.store.finish_run(
                role, run_id, {**self._summary(results), "error_code": type(exc).__name__}, "failed"
            )
            raise
        if progress:
            progress(len(case_ids), len(case_ids), "Selesai")
        return run_id

    @staticmethod
    def _summary(results):
        passed = sum(result["status"] == "passed" for result in results)
        known_tokens = [
            result["actual"]["tokens"]
            for result in results
            if result["actual"].get("tokens") is not None
        ]
        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_percent": 100 * passed / len(results) if results else None,
            "duration_ms": sum(result["duration_ms"] for result in results),
            "tokens": sum(known_tokens) if known_tokens else None,
        }

    def _run_case(self, case, mode, records, today, api_key):
        # Lazy import avoids coupling the composition root back to the evaluation runner.
        from bnivoice360.container import create_services

        started = time.monotonic()
        transcript, previews, error_code = [], [], ""
        with tempfile.TemporaryDirectory(
            prefix="bnivoice360-evaluation-", ignore_cleanup_errors=True
        ) as directory:
            settings = replace(
                self.settings,
                database_path=Path(directory) / "synthetic.db",
                seed_initial_data=False,
                download_embeddings=False,
                embedding_directory=Path(directory) / "vectors",
                companion_remote_enabled=mode != "Local",
            )
            services = create_services(settings)
            services.repository.insert(records)
            conversation, state = services.conversation, CompanionState()
            session_id = "evaluation-" + str(uuid.uuid4())
            baseline = {record.suara_nasabah_id: record.escalation_status for record in records}
            if mode == "Mock":
                conversation._client = lambda key: AuditedModelClient(
                    EvaluationMockClient(), conversation.batch.audit, session_id, "mock"
                )
            key = "evaluation-mock" if mode == "Mock" else api_key if mode == "Live Groq" else ""
            approval_rejected, committed = False, 0
            try:
                for step in case.steps:
                    if isinstance(step, str):
                        if mode == "Local" and step in SCENARIO_PLANS:
                            # Local evaluates typed tool execution, not natural-language understanding.
                            reply = conversation.execute_workflow(
                                state,
                                SCENARIO_PLANS[step],
                                session_id,
                                Role.MANAGER.value,
                                today=today,
                            )
                        elif mode == "Local" and case.case_id in {
                            "analysis",
                            "urgent_operational",
                            "display_after_analysis",
                        }:
                            from bnivoice360.services.analysis_tools import AnalysisTools

                            analysis = AnalysisTools(conversation.batch, today, step, {}, {})
                            if case.case_id in {"urgent_operational", "display_after_analysis"}:
                                scope = {
                                    "period": "all",
                                    "category": "Operasional",
                                    "status": "Active",
                                    "sentiment": "Negative",
                                    "urgent_only": True,
                                    "limit": 5,
                                }
                                plan = [("backlog_age", scope), ("complaint_examples", scope)]
                            else:
                                plan = [("volume_breakdown", {"period": "7d", "group_by": "topic"})]
                            state.analysis_sources = [
                                {
                                    "id": f"T{index}",
                                    "tool": name,
                                    "data": analysis.execute(name, scope, f"T{index}"),
                                }
                                for index, (name, scope) in enumerate(plan, 1)
                            ]
                            state.reply_kind = "analysis"
                            state.analysis_scope = analysis.scope.copy()
                            state.analysis_question = step
                            reply = f"Tools lokal menghitung {state.analysis_sources[0]['data']['total']} keluhan."
                        else:
                            reply = conversation.handle(
                                state,
                                step,
                                session_id,
                                Role.MANAGER.value,
                                api_key=key,
                                today=today,
                            )
                        transcript += [
                            {"role": "user", "content": step},
                            {"role": "assistant", "content": reply},
                        ]
                    else:
                        try:
                            committed = conversation.commit(
                                state, session_id, Role.MANAGER.value, approved=step["commit"]
                            )
                            transcript.append(
                                {
                                    "role": "system",
                                    "content": "Persetujuan diterima; status tiket sintetis diperbarui.",
                                }
                            )
                        except ReviewRequired:
                            if step["commit"]:
                                raise
                            approval_rejected = True
                            transcript.append(
                                {
                                    "role": "system",
                                    "content": "Perubahan tanpa persetujuan ditolak.",
                                }
                            )
                    if state.proposal and state.proposal.operation_id not in previews:
                        previews.append(state.proposal.operation_id)
            except Exception as exc:
                error_code = type(exc).__name__
            current = (
                services.repository.load()
                .set_index("suara_nasabah_id")["escalation_status"]
                .to_dict()
            )
            audit_rows = EvaluationStore._read(
                conversation.batch.audit.path,
                "events",
                "SELECT * FROM events ORDER BY event_id LIMIT 2000",
            )
            trace = [{**row, "metadata": decode_json(row["metadata"])} for row in audit_rows]
            calls = [row["metadata"] for row in trace if row["event"] == "model_call_finished"]
            known_tokens = [row["tokens"] for row in calls if row.get("tokens") is not None]
            old_cancelled = False
            if len(previews) > 1:
                operations = EvaluationStore._read(
                    conversation.batch.audit.path,
                    "operations",
                    "SELECT state FROM operations WHERE operation_id=?",
                    (previews[0],),
                )
                old_cancelled = bool(operations and operations[0]["state"] == "cancelled")
            query = state.query
            selected = [
                record for record in records if record.suara_nasabah_id in state.selected_ids
            ]
            actual = {
                "stage": state.stage.value,
                "total": state.total,
                "reply_kind": state.reply_kind,
                "result_ids": list(state.result_ids),
                "selected_ids": list(state.selected_ids),
                "keyword": query.keyword.lower() if query else None,
                "category": list(query.categories) if query else [],
                "sources": list(query.sources) if query else [],
                "urgent_only": query.urgent_only if query else False,
                "status": query.status if query else None,
                "period_days": (query.end - query.start).days + 1
                if query and query.start and query.end
                else None,
                "has_proposal": state.proposal is not None,
                "draft_available": bool(state.draft_text),
                "draft_mode": state.draft_mode,
                "draft": state.draft_text,
                "originals_present": bool(selected)
                and all(record.content in state.draft_text for record in selected),
                "metadata_present": bool(selected)
                and all(
                    all(
                        value in state.draft_text
                        for value in (record.suara_nasabah_id, record.source, record.date_posted)
                    )
                    for record in selected
                ),
                "status_unchanged": current == baseline,
                "other_statuses_unchanged": all(
                    current[ticket] == status
                    for ticket, status in baseline.items()
                    if ticket != "E001"
                ),
                "E001_status": current["E001"],
                "approval_rejected": approval_rejected,
                "committed": committed,
                "old_preview_cancelled": old_cancelled,
                "analysis_total": next(
                    (
                        source["data"]["total"]
                        for source in state.analysis_sources
                        if "total" in source["data"]
                    ),
                    None,
                ),
                "analysis_tools": [source["tool"] for source in state.analysis_sources],
                "last_tool": state.tool_trace[-1]["tool"] if state.tool_trace else None,
                "analysis_scopes_match": bool(state.analysis_sources)
                and all(
                    source["data"].get("scope") == state.analysis_sources[0]["data"].get("scope")
                    for source in state.analysis_sources
                ),
                "analysis_statuses": next(
                    (
                        source["data"]["status_counts"]
                        for source in state.analysis_sources
                        if "status_counts" in source["data"]
                    ),
                    None,
                ),
                "analysis_period": next(
                    (
                        source["data"]["scope"]["period"]
                        for source in state.analysis_sources
                        if "scope" in source["data"]
                    ),
                    None,
                ),
                "analysis_urgent_only": next(
                    (
                        source["data"]["scope"]["urgent_only"]
                        for source in state.analysis_sources
                        if "scope" in source["data"]
                    ),
                    None,
                ),
                "example_statuses": sorted(
                    {
                        example["status"]
                        for source in state.analysis_sources
                        for example in source["data"].get("examples", [])
                    }
                ),
                "tokens": sum(known_tokens) if known_tokens else None,
                "model_calls": len(calls),
            }
            expected = dict(case.expected)
            if mode != "Local" and case.category == "Draft":
                expected["draft_mode"] = "groq"
            checks = [
                {
                    "check": field,
                    "expected": value,
                    "actual": actual.get(field),
                    "passed": actual.get(field) == value,
                }
                for field, value in expected.items()
            ]
            if error_code:
                checks.append(
                    {
                        "check": "execution",
                        "expected": "completed",
                        "actual": error_code,
                        "passed": False,
                    }
                )
            return {
                "case_id": case.case_id,
                "label": case.label,
                "category": case.category,
                "status": "passed" if all(check["passed"] for check in checks) else "failed",
                "duration_ms": int((time.monotonic() - started) * 1000),
                "expected": expected,
                "actual": actual,
                "checks": checks,
                "transcript": transcript,
                "trace": trace,
                "error_code": error_code,
            }

    def compare(self, role: str, left: str, right: str) -> dict:
        require_admin(role)
        runs = {run["run_id"]: run for run in self.store.runs(role, 500)}
        if left == right or left not in runs or right not in runs:
            raise ValueError("Pilih dua run berbeda yang tersedia.")
        left_results = {result["case_id"]: result for result in self.store.results(role, left)}
        right_results = {result["case_id"]: result for result in self.store.results(role, right)}
        rows = []
        for case_id in sorted(left_results.keys() | right_results.keys()):
            a, b = left_results.get(case_id), right_results.get(case_id)
            rows.append(
                {
                    "Skenario": case_id,
                    "Run A": a["status"] if a else "Tidak diuji",
                    "Run B": b["status"] if b else "Tidak diuji",
                    "Durasi A (ms)": a["duration_ms"] if a else None,
                    "Durasi B (ms)": b["duration_ms"] if b else None,
                    "Perubahan": "Regresi"
                    if a and b and a["status"] == "passed" and b["status"] != "passed"
                    else "Membaik"
                    if a and b and a["status"] != "passed" and b["status"] == "passed"
                    else "Sama"
                    if a and b
                    else "Cakupan berbeda",
                }
            )
        comparable = (
            runs[left]["dataset_digest"] == runs[right]["dataset_digest"]
            and runs[left]["suite_version"] == runs[right]["suite_version"]
            and set(left_results) == set(right_results)
            and runs[left]["mode"] == runs[right]["mode"]
        )
        return {"rows": rows, "comparable": comparable, "left": runs[left], "right": runs[right]}
