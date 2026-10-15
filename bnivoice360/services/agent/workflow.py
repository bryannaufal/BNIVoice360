"""Server-owned search snapshots, bounded drafting, and approved atomic escalation."""

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from bnivoice360.config import Settings
from bnivoice360.data.repository import VoiceRepository
from bnivoice360.domain import IssueFilters, Role
from bnivoice360.services.agent.audit import AuditStore, fingerprint, utc_now
from bnivoice360.services.agent.groq_client import GroqClient, ModelUnavailable
from bnivoice360.services.agent.guardrails import GuardrailError, screen_input, validate_output
from bnivoice360.services.agent.retrieval import similar_voices
from bnivoice360.services.analytics import category_labels, filter_issues
from bnivoice360.services.remote_content import masked_text


class ReviewRequired(ValueError):
    pass


@dataclass(frozen=True)
class SearchResult:
    anchor_id: str
    candidates: pd.DataFrame
    snapshots: dict[str, str]
    context: str
    missing_embeddings: int
    created_at: float


@dataclass(frozen=True)
class Proposal:
    operation_id: str
    ids: tuple[str, ...]
    draft: dict
    aliases: dict[str, str]
    context: str
    mode: str
    reason: str


def context_key(filters: IssueFilters, topic_id: int | None) -> str:
    return fingerprint({"filters": repr(filters), "deep_dive": topic_id})


def _rows(connection, ids) -> dict[str, dict]:
    result = {}
    for record_id in ids:
        row = connection.execute(
            "SELECT * FROM suara_nasabah WHERE suara_nasabah_id=?", (record_id,)
        ).fetchone()
        if row is not None:
            result[record_id] = dict(row)
    return result


class BatchAgent:
    def __init__(self, repository: VoiceRepository, settings: Settings):
        self.repository = repository
        self.settings = settings
        self.audit = AuditStore(repository.database.path)

    def search(
        self,
        anchor_id: str,
        filters: IssueFilters,
        topic_id: int | None,
        *,
        cross_topic: bool,
        threshold: float,
        limit: int,
        session_id: str,
    ) -> SearchResult:
        started = time.monotonic()
        frame = filter_issues(self.repository.load(), filters)
        frame = frame.loc[frame["sentiment"].eq("Negative") & frame["escalation_status"].eq("Open")]
        anchors = frame.loc[frame["suara_nasabah_id"].eq(anchor_id)]
        if anchors.empty or (topic_id is not None and not anchors["topic_id"].eq(topic_id).all()):
            raise ReviewRequired("Keluhan awal tidak lagi Open atau berada di luar filter aktif.")
        anchor = anchors.iloc[0]
        if not cross_topic:
            if pd.isna(anchor["topic_id"]) or anchor["topic_id"] == -1:
                raise ReviewRequired(
                    "Outlier/Unlabeled memerlukan pencarian lintas topik secara eksplisit."
                )
            frame = frame.loc[frame["topic_id"].eq(anchor["topic_id"])]
        # Deep-dive limits the anchor. Explicit cross-topic search still respects global filters.
        try:
            candidates = similar_voices(
                self.settings.embedding_directory / "data_id_embedding_mapping.parquet",
                frame,
                anchor_id,
                threshold,
                limit,
            )
        except (OSError, ValueError, ImportError):
            self.audit.log(session_id, "search_failed", reason="embedding_unavailable")
            raise
        missing = candidates.attrs.get("missing_embeddings", 0)
        display = pd.concat(
            [anchors.assign(similarity=1.0, has_embedding=True), candidates], ignore_index=True
        )
        ids = tuple(display["suara_nasabah_id"].astype(str))
        with self.repository.database.transaction() as connection:
            rows = _rows(connection, ids)
        if len(rows) != len(ids) or any(
            row["escalation_status"] != "Open" for row in rows.values()
        ):
            raise ReviewRequired("Data berubah saat pencarian. Cari kembali.")
        # Ensure the display corresponds to the same live snapshot, including concurrent edits.
        for _, row in display.iterrows():
            current = rows[str(row["suara_nasabah_id"])]
            if any(
                str(current[key]) != str(row[key])
                for key in ("content", "source", "escalation_status")
            ):
                raise ReviewRequired("Data berubah saat pencarian. Cari kembali.")
        snapshots = {record_id: fingerprint(row) for record_id, row in rows.items()}
        self.audit.log(
            session_id,
            "similarity_search",
            anchor_digest=fingerprint({"id": anchor_id}),
            candidates=len(candidates),
            missing_embeddings=missing,
            threshold=threshold,
            cross_topic=cross_topic,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return SearchResult(
            anchor_id, display, snapshots, context_key(filters, topic_id), missing, time.time()
        )

    @staticmethod
    def _validate_search(search: SearchResult, ids: tuple[str, ...], context: str):
        if context != search.context or time.time() - search.created_at > 900:
            raise ReviewRequired("Filter atau masa preview berubah. Cari kembali.")
        if not ids or len(ids) > 51 or len(set(ids)) != len(ids):
            raise ReviewRequired("Pilih keluhan yang unik, maksimal 51 tiket.")
        if search.anchor_id not in ids or not set(ids) <= search.snapshots.keys():
            raise ReviewRequired(
                "Pilihan harus berasal dari hasil pencarian dan menyertakan keluhan awal."
            )

    @staticmethod
    def _template() -> dict:
        return {
            "findings": [],
            "hypotheses": [],
            "actions": [
                {
                    "text": "Periksa keluhan terpilih, pastikan pola masalah yang sama, "
                    "lalu tentukan unit tujuan dan langkah pemeriksaan.",
                    "team": None,
                    "evidence": [],
                }
            ],
        }

    def prepare(
        self,
        search: SearchResult,
        ids: tuple[str, ...],
        context: str,
        session_id: str,
        role: str,
        *,
        instruction: str,
        use_ai: bool = False,
        api_key: str = "",
    ) -> Proposal:
        self._validate_search(search, ids, context)
        if role not in {item.value for item in Role}:
            raise ReviewRequired("Role tidak valid.")
        with self.repository.database.transaction() as connection:
            rows = _rows(connection, ids)
        if any(
            record_id not in rows or fingerprint(rows[record_id]) != search.snapshots[record_id]
            for record_id in ids
        ):
            raise ReviewRequired("Tiket berubah sejak pencarian. Cari dan tinjau kembali.")
        aliases = {f"V{i + 1}": record_id for i, record_id in enumerate(ids)}
        draft, mode, reason = self._template(), "manual", "manual_template"
        try:
            clean_instruction = screen_input(instruction, instruction=True)
            if use_ai:
                if not self.settings.agent_remote_enabled:
                    raise ModelUnavailable("remote_disabled")
                evidence = {}
                for alias, record_id in aliases.items():
                    evidence[alias] = masked_text(rows[record_id])[:2000]
                if sum(len(text) for text in evidence.values()) > 30000:
                    raise GuardrailError("context_budget")
                selected = search.candidates.loc[search.candidates["suara_nasabah_id"].isin(ids)]
                stats = {
                    "selected_count": len(ids),
                    "categories": sorted(
                        {
                            category
                            for _, row in selected.iterrows()
                            for category in category_labels(row)
                        }
                    ),
                }
                raw, usage = GroqClient(api_key, self.settings.agent_model).draft(
                    evidence, stats, clean_instruction
                )
                draft = validate_output(raw, evidence)
                mode, reason = "ai", ""
                self.audit.log(
                    session_id,
                    "draft_generated",
                    model=self.settings.agent_model,
                    selected=len(ids),
                    **usage,
                )
        except GuardrailError as exc:
            reason = str(exc)
            self.audit.log(session_id, "guardrail_blocked", reason=reason, selected=len(ids))
        except ModelUnavailable as exc:
            reason = str(exc)
            self.audit.log(session_id, "model_fallback", reason=reason, selected=len(ids))
        operation_id = str(uuid.uuid4())
        snapshots = {record_id: search.snapshots[record_id] for record_id in ids}
        # Durable operation binds approval to exact IDs, snapshots, session and role. No draft text is stored.
        with self.audit.transaction() as connection:
            connection.execute(
                "INSERT INTO audit.operations VALUES (?,?,?,?,?,?,?,?)",
                (
                    operation_id,
                    session_id,
                    role,
                    context,
                    json.dumps(snapshots),
                    utc_now(),
                    "pending",
                    len(ids),
                ),
            )
            self.audit.event(
                connection,
                session_id,
                "proposal_prepared",
                {"operation_id": operation_id, "count": len(ids), "mode": mode, "reason": reason},
            )
        return Proposal(operation_id, ids, draft, aliases, context, mode, reason)

    def commit(
        self,
        proposal: Proposal,
        session_id: str,
        role: str,
        context: str,
        ids: tuple[str, ...],
        *,
        approved: bool,
    ) -> int:
        if approved is not True:
            raise ReviewRequired("Konfirmasi pengguna diperlukan.")
        try:
            with self.audit.transaction() as connection:
                operation = connection.execute(
                    "SELECT * FROM audit.operations WHERE operation_id=?", (proposal.operation_id,)
                ).fetchone()
                if (
                    operation is None
                    or operation["session_id"] != session_id
                    or operation["role"] != role
                ):
                    raise ReviewRequired("Konfirmasi bukan milik sesi atau role ini.")
                snapshots = json.loads(operation["snapshots"])
                if (
                    context != operation["context"]
                    or set(ids) != set(snapshots)
                    or len(ids) != len(snapshots)
                ):
                    raise ReviewRequired("Pilihan atau filter berubah. Buat preview baru.")
                if operation["state"] == "completed":
                    return operation["count"]
                age = (
                    datetime.fromisoformat(utc_now())
                    - datetime.fromisoformat(operation["created_at"])
                ).total_seconds()
                if operation["state"] != "pending" or age > 900:
                    raise ReviewRequired(
                        "Konfirmasi dibatalkan atau kedaluwarsa. Buat preview baru."
                    )
                rows = _rows(connection, ids)
                if any(
                    record_id not in rows
                    or rows[record_id]["escalation_status"] != "Open"
                    or fingerprint(rows[record_id]) != digest
                    for record_id, digest in snapshots.items()
                ):
                    raise ReviewRequired(
                        "Ada tiket berubah sejak preview. Tidak ada status yang diperbarui."
                    )
                for record_id in ids:
                    cursor = connection.execute(
                        "UPDATE suara_nasabah SET escalation_status='Escalated' "
                        "WHERE suara_nasabah_id=? AND escalation_status='Open'",
                        (record_id,),
                    )
                    if cursor.rowcount != 1:
                        raise ReviewRequired("Status berubah; eskalasi batch dibatalkan.")
                connection.execute(
                    "UPDATE audit.operations SET state='completed' WHERE operation_id=?",
                    (proposal.operation_id,),
                )
                self.audit.event(
                    connection,
                    session_id,
                    "batch_escalated",
                    {"operation_id": proposal.operation_id, "count": len(ids), "role": role},
                )
                return len(ids)
        except ReviewRequired:
            self.audit.log(
                session_id,
                "batch_rejected",
                operation_id=proposal.operation_id,
                reason="review_required",
            )
            raise

    def register_preview(
        self,
        ids: tuple[str, ...],
        snapshots: dict[str, str],
        context: str,
        session_id: str,
        role: str,
    ) -> Proposal:
        """Create an approval operation for an explicit chat selection, even without vectors."""
        if (
            not ids
            or len(ids) > 51
            or len(set(ids)) != len(ids)
            or not set(ids) <= snapshots.keys()
            or role not in {item.value for item in Role}
        ):
            raise ReviewRequired("Pilihan atau role tidak valid.")
        operation_id = str(uuid.uuid4())
        selected_snapshots = {record_id: snapshots[record_id] for record_id in ids}
        with self.audit.transaction() as connection:
            rows = _rows(connection, ids)
            if any(
                record_id not in rows
                or rows[record_id]["escalation_status"] != "Open"
                or fingerprint(rows[record_id]) != digest
                for record_id, digest in selected_snapshots.items()
            ):
                raise ReviewRequired("Tiket berubah atau bukan Open. Cari dan tinjau kembali.")
            connection.execute(
                "INSERT INTO audit.operations VALUES (?,?,?,?,?,?,?,?)",
                (
                    operation_id,
                    session_id,
                    role,
                    context,
                    json.dumps(selected_snapshots),
                    utc_now(),
                    "pending",
                    len(ids),
                ),
            )
            self.audit.event(
                connection,
                session_id,
                "companion_preview_prepared",
                {"operation_id": operation_id, "count": len(ids)},
            )
        return Proposal(
            operation_id,
            ids,
            {},
            {f"V{i + 1}": ticket for i, ticket in enumerate(ids)},
            context,
            "companion",
            "",
        )
