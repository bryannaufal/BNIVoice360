"""Admin-only evaluation persistence and bounded, read-only access to operational audit logs."""

import json
import math
import sqlite3
from contextlib import closing, contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from bnivoice360.domain import Role
from bnivoice360.services.agent.audit import utc_now

JAKARTA = ZoneInfo("Asia/Jakarta")
META = "CASE WHEN json_valid(metadata) THEN metadata ELSE '{}' END"
STATUS = f"COALESCE(json_extract({META}, '$.status'), CASE WHEN event LIKE '%fallback%' THEN 'fallback' WHEN event LIKE '%guardrail%' OR event LIKE '%rejected%' THEN 'blocked' ELSE 'event' END)"


def require_admin(role: str) -> None:
    if role != Role.DATA_ADMIN.value:
        raise PermissionError("Companion Evaluation hanya tersedia untuk Data Admin.")


def decode_json(value: str):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {"invalid_metadata": True}


def utc_window(start: date, end: date) -> tuple[str, str]:
    if start > end:
        raise ValueError("Tanggal mulai harus sebelum atau sama dengan tanggal akhir.")
    return tuple(
        datetime.combine(day, time.min, JAKARTA).astimezone(timezone.utc).isoformat()
        for day in (start, end + timedelta(days=1))
    )


class EvaluationStore:
    def __init__(self, business_path: Path):
        self.path = business_path.with_name(business_path.stem + ".companion-evaluation.db")
        self.audit_path = business_path.with_name(business_path.stem + ".agent-audit.db")

    @staticmethod
    def _read(path: Path, table: str, sql: str, params=()) -> list[dict]:
        if not path.exists():
            return []
        with closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
        ) as connection:
            connection.row_factory = sqlite3.Row
            if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name=? AND type='table'", (table,)
            ).fetchone():
                return []
            return [dict(row) for row in connection.execute(sql, params)]

    @contextmanager
    def _write(self, role: str):
        require_admin(role)
        connection = sqlite3.connect(self.path, timeout=20)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
                    mode TEXT NOT NULL, status TEXT NOT NULL, model TEXT NOT NULL,
                    code_digest TEXT NOT NULL, prompt_digest TEXT NOT NULL,
                    dataset_digest TEXT NOT NULL, suite_version TEXT NOT NULL,
                    config TEXT NOT NULL, summary TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id),
                    case_id TEXT NOT NULL, label TEXT NOT NULL, category TEXT NOT NULL,
                    status TEXT NOT NULL, duration_ms INTEGER NOT NULL,
                    expected TEXT NOT NULL, actual TEXT NOT NULL, checks TEXT NOT NULL,
                    transcript TEXT NOT NULL, trace TEXT NOT NULL, error_code TEXT NOT NULL,
                    PRIMARY KEY(run_id, case_id)
                );
                CREATE TABLE IF NOT EXISTS evaluation_reviews (
                    run_id TEXT NOT NULL, case_id TEXT NOT NULL,
                    rating TEXT NOT NULL, note TEXT NOT NULL, reviewed_at TEXT NOT NULL,
                    reviewer_role TEXT NOT NULL,
                    PRIMARY KEY(run_id, case_id),
                    FOREIGN KEY(run_id, case_id) REFERENCES evaluation_results(run_id, case_id)
                );
            """)
            with connection:
                yield connection
        finally:
            connection.close()

    def start_run(self, role: str, run: dict) -> None:
        with self._write(role) as connection:
            connection.execute(
                "INSERT INTO evaluation_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run["run_id"],
                    run["started_at"],
                    None,
                    run["mode"],
                    "running",
                    run["model"],
                    run["code_digest"],
                    run["prompt_digest"],
                    run["dataset_digest"],
                    run["suite_version"],
                    json.dumps(run["config"]),
                    "{}",
                ),
            )

    def save_result(self, role: str, run_id: str, result: dict) -> None:
        with self._write(role) as connection:
            connection.execute(
                "INSERT INTO evaluation_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    result["case_id"],
                    result["label"],
                    result["category"],
                    result["status"],
                    result["duration_ms"],
                    *(
                        json.dumps(result[field], ensure_ascii=False)
                        for field in ("expected", "actual", "checks", "transcript", "trace")
                    ),
                    result.get("error_code", ""),
                ),
            )

    def finish_run(self, role: str, run_id: str, summary: dict, status="completed") -> None:
        with self._write(role) as connection:
            connection.execute(
                "UPDATE evaluation_runs SET finished_at=?,status=?,summary=? WHERE run_id=?",
                (
                    utc_now(),
                    status,
                    json.dumps(summary),
                    run_id,
                ),
            )

    def runs(self, role: str, limit=100) -> list[dict]:
        require_admin(role)
        rows = self._read(
            self.path,
            "evaluation_runs",
            "SELECT * FROM evaluation_runs ORDER BY started_at DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        )
        for row in rows:
            for field in ("summary", "config"):
                row[field] = decode_json(row[field])
        return rows

    def results(self, role: str, run_id: str) -> list[dict]:
        require_admin(role)
        rows = self._read(
            self.path,
            "evaluation_results",
            """
            SELECT r.*,v.rating AS review_rating,v.note AS review_note,v.reviewed_at
            FROM evaluation_results r LEFT JOIN evaluation_reviews v
            ON r.run_id=v.run_id AND r.case_id=v.case_id WHERE r.run_id=? ORDER BY r.case_id
        """,
            (run_id,),
        )
        for row in rows:
            for field in ("expected", "actual", "checks", "transcript", "trace"):
                row[field] = decode_json(row[field])
        return rows

    def review(self, role: str, run_id: str, case_id: str, rating: str, note: str) -> None:
        require_admin(role)
        if rating not in {"Sesuai", "Sebagian sesuai", "Tidak sesuai"} or len(note) > 2000:
            raise ValueError("Rating atau panjang catatan review tidak valid.")
        with self._write(role) as connection:
            if not connection.execute(
                "SELECT 1 FROM evaluation_results WHERE run_id=? AND case_id=?", (run_id, case_id)
            ).fetchone():
                raise ValueError("Hasil evaluasi tidak ditemukan.")
            connection.execute(
                """INSERT INTO evaluation_reviews VALUES (?,?,?,?,?,?)
                ON CONFLICT(run_id,case_id) DO UPDATE SET rating=excluded.rating,note=excluded.note,
                reviewed_at=excluded.reviewed_at,reviewer_role=excluded.reviewer_role""",
                (run_id, case_id, rating, note.strip(), utc_now(), role),
            )

    def log_options(self, role: str, start: date, end: date) -> dict:
        require_admin(role)
        window = utc_window(start, end)
        rows = self._read(
            self.audit_path,
            "events",
            "SELECT DISTINCT event FROM events WHERE timestamp>=? AND timestamp<? ORDER BY event",
            window,
        )
        return {"events": [row["event"] for row in rows]}

    def logs(
        self,
        role: str,
        start: date,
        end: date,
        *,
        event=None,
        session=None,
        status=None,
        offset=0,
        limit=100,
    ) -> dict:
        require_admin(role)
        conditions, params = ["timestamp>=?", "timestamp<?"], list(utc_window(start, end))
        if event:
            conditions.append("event=?")
            params.append(event)
        if session:
            conditions.append("session_id=?")
            params.append(session)
        if status:
            conditions.append(STATUS + "=?")
            params.append(status)
        where = " AND ".join(conditions)
        total = self._read(
            self.audit_path, "events", f"SELECT COUNT(*) AS n FROM events WHERE {where}", params
        )
        rows = self._read(
            self.audit_path,
            "events",
            f"SELECT *,{STATUS} AS status FROM events WHERE {where} ORDER BY event_id DESC LIMIT ? OFFSET ?",
            (*params, min(500, max(1, limit)), max(0, offset)),
        )
        for row in rows:
            row["metadata"] = decode_json(row["metadata"])
            if not isinstance(row["metadata"], dict):
                row["metadata"] = {"invalid_metadata": True}
            row["interaction_id"] = row["metadata"].get("interaction_id")
        return {"total": total[0]["n"] if total else 0, "rows": rows}

    def interaction_events(self, role: str, interaction_id: str, limit=500) -> list[dict]:
        require_admin(role)
        rows = self._read(
            self.audit_path,
            "events",
            f"SELECT * FROM events WHERE json_extract({META}, '$.interaction_id')=? ORDER BY event_id LIMIT ?",
            (interaction_id, min(500, max(1, limit))),
        )
        for row in rows:
            row["metadata"] = decode_json(row["metadata"])
            if not isinstance(row["metadata"], dict):
                row["metadata"] = {"invalid_metadata": True}
        return rows

    def overview(self, role: str, start: date, end: date) -> dict:
        require_admin(role)
        window = utc_window(start, end)
        rows = self._read(
            self.audit_path,
            "events",
            f"""
            SELECT event,{STATUS} AS status,COUNT(*) AS n,
            SUM(json_extract({META}, '$.tokens')) AS tokens,
            COUNT(json_extract({META}, '$.tokens')) AS token_samples,
            COUNT(json_extract({META}, '$.duration_ms')) AS duration_samples,
            AVG(json_extract({META}, '$.duration_ms')) AS mean_ms
            FROM events WHERE timestamp>=? AND timestamp<? GROUP BY event,status
        """,
            window,
        )
        interactions = [row for row in rows if row["event"] == "interaction_finished"]
        total = sum(row["n"] for row in interactions)
        duration_samples = sum(row["duration_samples"] for row in interactions)
        ok = sum(row["n"] for row in interactions if row["status"] == "ok")
        models = [row for row in rows if row["event"] == "model_call_finished"]
        tools = [
            row
            for row in rows
            if row["event"] == "analysis_tool_finished" and row["mean_ms"] is not None
        ]
        durations = (
            self._read(
                self.audit_path,
                "events",
                f"""
            SELECT json_extract({META}, '$.duration_ms') AS duration FROM events
            WHERE timestamp>=? AND timestamp<? AND event='interaction_finished'
            AND json_extract({META}, '$.duration_ms') IS NOT NULL
            ORDER BY duration LIMIT 1 OFFSET ?
        """,
                (*window, max(0, math.ceil(duration_samples * 0.95) - 1)),
            )
            if duration_samples
            else []
        )

        def mean(samples):
            known = [row for row in samples if row["mean_ms"] is not None]
            return (
                sum(row["n"] * row["mean_ms"] for row in known) / sum(row["n"] for row in known)
                if known
                else None
            )

        return {
            "events": sum(row["n"] for row in rows),
            "interactions": total,
            "ok_percent": 100 * ok / total if total else None,
            "fallback_events": sum(
                row["n"]
                for row in rows
                if row["status"] == "fallback" and row["event"] != "interaction_finished"
            ),
            "blocked_events": sum(
                row["n"]
                for row in rows
                if row["status"] == "blocked" and row["event"] != "interaction_finished"
            ),
            "p95_ms": durations[0]["duration"] if durations else None,
            "tokens": sum(row["tokens"] or 0 for row in models)
            if any(row["token_samples"] for row in models)
            else None,
            "model_calls": sum(row["n"] for row in models),
            "model_mean_ms": mean(models),
            "tool_mean_ms": mean(tools),
            "event_counts": rows,
        }
