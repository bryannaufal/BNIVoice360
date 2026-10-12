"""Durable metadata and atomic two-file SQLite commits using rollback journals."""

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

_interaction = ContextVar("companion_audit_interaction", default=None)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()


class AuditStore:
    def __init__(self, business_path: Path):
        self.business_path = business_path
        self.path = business_path.with_name(business_path.stem + ".agent-audit.db")

    @contextmanager
    def transaction(self):
        connection = sqlite3.connect(self.business_path, timeout=20)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("ATTACH DATABASE ? AS audit", (str(self.path),))
            for database in ("main", "audit"):
                journal = connection.execute(f"PRAGMA {database}.journal_mode").fetchone()[0]
                if journal != "delete":
                    raise ValueError(
                        "Agent requires DELETE journal mode for atomic status and audit writes."
                    )
                connection.execute(f"PRAGMA {database}.synchronous=FULL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS audit.events (
                    event_id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL, event TEXT NOT NULL, metadata TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit.operations (
                    operation_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    role TEXT NOT NULL, context TEXT NOT NULL, snapshots TEXT NOT NULL,
                    created_at TEXT NOT NULL, state TEXT NOT NULL, count INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS audit.events_time ON events(timestamp,event);
            """)
            connection.execute("BEGIN IMMEDIATE")
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def event(connection, session_id: str, event: str, metadata: dict):
        context = _interaction.get()
        if context:
            metadata = {**metadata, "interaction_id": context["interaction_id"]}
        connection.execute(
            "INSERT INTO audit.events(timestamp,session_id,event,metadata) VALUES (?,?,?,?)",
            (utc_now(), session_id, event, json.dumps(metadata, sort_keys=True)),
        )

    def log(self, session_id: str, event: str, **metadata):
        # Callers pass reason codes, IDs/digests and metrics only, never customer content.
        with self.transaction() as connection:
            self.event(connection, session_id, event, metadata)

    @contextmanager
    def interaction(self, session_id: str, action: str, role: str, model: str):
        """Nested actions share an ID; context stays isolated across concurrent sessions."""
        current = _interaction.get()
        if current and current["store"] == str(self.path):
            yield current["outcome"]
            return
        outcome = {"status": "ok"}
        context = {
            "interaction_id": str(uuid.uuid4()),
            "store": str(self.path),
            "outcome": outcome,
            "session_id": session_id,
        }
        token = _interaction.set(context)
        started = time.monotonic()
        try:
            self.log(session_id, "interaction_started", action=action, role=role, model=model)
            try:
                yield outcome
            except Exception as exc:
                outcome.update(status="failed", error_code=type(exc).__name__)
                raise
            finally:
                self.log(
                    session_id,
                    "interaction_finished",
                    action=action,
                    role=role,
                    model=model,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    **outcome,
                )
        finally:
            _interaction.reset(token)

    def current_session(self) -> str | None:
        context = _interaction.get()
        return context["session_id"] if context and context["store"] == str(self.path) else None

    def mark_outcome(self, **metadata):
        context = _interaction.get()
        if context and context["store"] == str(self.path):
            context["outcome"].update(metadata)

    def cancel(self, operation_id: str, session_id: str):
        with self.transaction() as connection:
            connection.execute(
                "UPDATE audit.operations SET state='cancelled' "
                "WHERE operation_id=? AND session_id=? AND state='pending'",
                (operation_id, session_id),
            )
            self.event(connection, session_id, "proposal_cancelled", {"operation_id": operation_id})
