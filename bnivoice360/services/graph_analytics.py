"""Bounded complaint relationships, reviewed snapshots, and grounded group explanations."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from bnivoice360.domain import IssueFilters
from bnivoice360.services.agent.audit import fingerprint
from bnivoice360.services.agent.groq_client import GroqClient, ModelUnavailable
from bnivoice360.services.agent.guardrails import GuardrailError
from bnivoice360.services.agent.retrieval import load_voice_vectors
from bnivoice360.services.agent.telemetry import AuditedModelClient
from bnivoice360.services.agent.workflow import ReviewRequired, _rows
from bnivoice360.services.analysis_agent import ANSWER_PROMPT, answer_schema, render_answer
from bnivoice360.services.analysis_tools import safe_data
from bnivoice360.services.analytics import category_labels, filter_issues
from bnivoice360.services.remote_content import masked_text

EDGE_COLUMNS = ["source_id", "target_id", "similarity"]
GROUP_COLUMNS = [
    "group_id",
    "count",
    "open",
    "escalated",
    "resolved",
    "urgent_open",
    "channels",
    "units",
    "oldest_open_date",
    "oldest_open_days",
    "edges",
]


@dataclass
class ComplaintNetwork:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    groups: pd.DataFrame
    scope_count: int
    missing_embeddings: int
    omitted_by_limit: int
    threshold: float
    neighbors: int
    as_of: date
    snapshots: dict[str, str] = field(default_factory=dict)

    def members(self, group_id: str) -> pd.DataFrame:
        return self.nodes.loc[self.nodes["group_id"].eq(group_id)].copy()


def build_network(
    frame: pd.DataFrame,
    path: Path,
    filters: IssueFilters,
    *,
    topic_id: int | None,
    cross_topic: bool = False,
    statuses: tuple[str, ...] = ("Open", "Escalated"),
    threshold: float = 0.80,
    max_nodes: int = 120,
    neighbors: int = 5,
    as_of: date,
) -> ComplaintNetwork:
    """Build a union-of-top-k graph only AFTER bounding eligible nodes to at most 200."""
    if (
        not np.isfinite(threshold)
        or not 0 <= threshold <= 1
        or not 2 <= max_nodes <= 200
        or not 1 <= neighbors <= 10
        or not statuses
        or not set(statuses) <= {"Open", "Escalated", "Resolved"}
    ):
        raise ValueError("Parameter jaringan tidak valid.")
    if topic_id is None and not cross_topic:
        raise ValueError("Semua topik memerlukan konfirmasi pencarian lintas topik.")
    if topic_id == -1 and not cross_topic:
        raise ValueError("Outlier memerlukan konfirmasi pencarian lintas topik.")
    scoped = filter_issues(frame, filters)
    scoped = scoped.loc[
        scoped["sentiment"].eq("Negative") & scoped["escalation_status"].isin(statuses)
    ].copy()
    if topic_id is not None:
        scoped = scoped.loc[scoped["topic_id"].eq(topic_id)].copy()
    if scoped["suara_nasabah_id"].duplicated().any():
        raise ValueError("ID keluhan harus unik.")
    scoped["suara_nasabah_id"] = scoped["suara_nasabah_id"].astype(str)
    scope_count = len(scoped)
    if scoped.empty:
        scoped["group_id"], scoped["degree"] = "", 0
        return ComplaintNetwork(
            scoped,
            pd.DataFrame(columns=EDGE_COLUMNS),
            pd.DataFrame(columns=GROUP_COLUMNS),
            0,
            0,
            0,
            threshold,
            neighbors,
            as_of,
        )
    ids, vectors = load_voice_vectors(path)
    index = {ticket: i for i, ticket in enumerate(ids)}
    has_vector = scoped["suara_nasabah_id"].isin(index)
    missing = int((~has_vector).sum())
    eligible = scoped.loc[has_vector].copy()
    # Stable, transparent sampling: urgent Open, other Open, Escalated, Resolved; newest first.
    eligible["_priority"] = eligible["escalation_status"].map(
        {"Open": 1, "Escalated": 2, "Resolved": 3}
    ) - (eligible["is_Urgent"].eq(1) & eligible["escalation_status"].eq("Open")).astype(int)
    nodes = (
        eligible.sort_values(
            ["_priority", "date_posted", "suara_nasabah_id"],
            ascending=[True, False, True],
            na_position="last",
        )
        .head(max_nodes)
        .drop(columns="_priority")
        .reset_index(drop=True)
    )
    tickets = nodes["suara_nasabah_id"].tolist()
    matrix = vectors[[index[ticket] for ticket in tickets]]
    scores = np.clip(matrix @ matrix.T, -1, 1)
    np.fill_diagonal(scores, -np.inf)
    pairs = set()
    for i in range(len(nodes)):
        candidates = [j for j in range(len(nodes)) if scores[i, j] >= threshold]
        candidates.sort(key=lambda j: (-float(scores[i, j]), tickets[j]))
        for j in candidates[:neighbors]:
            pairs.add((min(i, j), max(i, j)))
    edges = pd.DataFrame(
        [(tickets[i], tickets[j], float(scores[i, j])) for i, j in sorted(pairs)],
        columns=EDGE_COLUMNS,
    )
    adjacency = {ticket: set() for ticket in tickets}
    for source, target, _ in edges.itertuples(index=False, name=None):
        adjacency[source].add(target)
        adjacency[target].add(source)
    memberships = {}
    for ticket in sorted(tickets):
        if ticket in memberships:
            continue
        stack, component = [ticket], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        group_id = fingerprint({"tickets": sorted(component)})[:16]
        memberships.update(dict.fromkeys(component, group_id))
    nodes["group_id"] = nodes["suara_nasabah_id"].map(memberships)
    nodes["degree"] = nodes["suara_nasabah_id"].map(lambda ticket: len(adjacency[ticket]))
    summaries = []
    for group_id, group in nodes.groupby("group_id", sort=True):
        open_rows = group.loc[group["escalation_status"].eq("Open")]
        oldest = open_rows["date_posted"].min()
        summaries.append(
            dict(
                group_id=group_id,
                count=len(group),
                open=len(open_rows),
                escalated=int(group["escalation_status"].eq("Escalated").sum()),
                resolved=int(group["escalation_status"].eq("Resolved").sum()),
                urgent_open=int(open_rows["is_Urgent"].eq(1).sum()),
                channels=int(group["source"].nunique()),
                units=", ".join(
                    sorted({unit for _, row in group.iterrows() for unit in category_labels(row)})
                ),
                oldest_open_date=oldest.date().isoformat() if pd.notna(oldest) else None,
                oldest_open_days=max(0, (as_of - oldest.date()).days) if pd.notna(oldest) else None,
                edges=sum(len(adjacency[ticket]) for ticket in group["suara_nasabah_id"]) // 2,
            )
        )
    groups = (
        pd.DataFrame(summaries, columns=GROUP_COLUMNS)
        .sort_values(
            ["urgent_open", "open", "oldest_open_date", "group_id"],
            ascending=[False, False, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    return ComplaintNetwork(
        nodes,
        edges,
        groups,
        scope_count,
        missing,
        max(0, len(eligible) - len(nodes)),
        threshold,
        neighbors,
        as_of,
    )


def bind_snapshots(repository, network: ComplaintNetwork) -> None:
    """Bind the actual displayed rows, rejecting concurrent edits before publication."""
    tickets = tuple(network.nodes["suara_nasabah_id"])
    with repository.database.transaction() as connection:
        rows = _rows(connection, tickets)
    for _, displayed in network.nodes.iterrows():
        current = rows.get(displayed["suara_nasabah_id"])
        if current is None:
            raise ReviewRequired("Data berubah saat membangun jaringan. Bangun ulang.")
        for key, value in current.items():
            before = displayed[key]
            after = pd.to_datetime(value) if key == "date_posted" else value
            if pd.isna(before) and pd.isna(after):
                continue
            if pd.isna(before) or pd.isna(after) or before != after:
                raise ReviewRequired("Data berubah saat membangun jaringan. Bangun ulang.")
    network.snapshots = {ticket: fingerprint(row) for ticket, row in rows.items()}


def explain_group(batch, network, group_id, session_id, role, api_key=""):
    """Read-only, explicitly invoked explanation; never changes Companion selection or approval."""
    members = network.members(group_id)
    if members.empty:
        raise ReviewRequired("Pilih kelompok dari jaringan yang ditampilkan.")
    with batch.repository.database.transaction() as connection:
        live = _rows(connection, tuple(members["suara_nasabah_id"]))
    if any(
        ticket not in live or fingerprint(live[ticket]) != network.snapshots.get(ticket)
        for ticket in members["suara_nasabah_id"]
    ):
        raise ReviewRequired("Keluhan berubah sejak jaringan dibangun. Bangun ulang.")
    summary = network.groups.set_index("group_id").loc[group_id]
    stats = {
        key: int(summary[key])
        for key in (
            "count",
            "open",
            "escalated",
            "resolved",
            "urgent_open",
            "channels",
            "edges",
        )
    }
    stats.update(
        threshold=network.threshold,
        max_neighbors=network.neighbors,
        as_of=network.as_of.isoformat(),
        start_date=members["date_posted"].min().date().isoformat()
        if members["date_posted"].notna().any()
        else None,
        end_date=members["date_posted"].max().date().isoformat()
        if members["date_posted"].notna().any()
        else None,
        units=summary["units"],
        source_counts=[
            {"source": str(k), "count": int(v)} for k, v in members["source"].value_counts().items()
        ],
        undated=int(members["date_posted"].isna().sum()),
    )
    local = (
        f"Kelompok ini berisi {stats['count']} keluhan dari {stats['channels']} kanal: "
        f"{stats['open']} Open, {stats['escalated']} Escalated, dan {stats['resolved']} Resolved. "
        f"Ada {stats['urgent_open']} keluhan urgent Open. [T1]\n\n"
        "Hubungan menunjukkan kemiripan teks, bukan insiden atau akar penyebab yang terkonfirmasi. "
        "Kelompok dapat terhubung melalui rantai; tidak semua pasangan memiliki hubungan langsung."
    )
    reports = [{"id": "T1", "tool": "network_group", "data": safe_data(stats)}]
    with batch.audit.interaction(
        session_id, "graph_explanation", role, batch.settings.agent_model
    ) as outcome:
        if not batch.settings.companion_remote_enabled or not api_key:
            outcome.update(status="ok", mode="local")
            return local, reports, "local"
        try:
            for _, row in members.iterrows():
                try:
                    text = masked_text(row)
                except GuardrailError:
                    continue
                reports.append(
                    {
                        "id": f"T{len(reports) + 1}",
                        "tool": "complaint_example",
                        "data": safe_data(
                            {
                                "excerpt": text[:700],
                                "source": row["source"],
                                "topic": row["topic_cluster"],
                                "status": row["escalation_status"],
                                "urgent": bool(row["is_Urgent"] == 1),
                            }
                        ),
                    }
                )
                if len(reports) == 6:
                    break
            if len(reports) == 1:
                raise GuardrailError("graph_no_screened_examples")
            client = AuditedModelClient(
                GroqClient(api_key, batch.settings.agent_model),
                batch.audit,
                session_id,
                batch.settings.agent_model,
            )
            result, _ = client.structured(
                ANSWER_PROMPT
                + "\nExplain this complaint network group: shared reported symptoms, differences, "
                "channel coverage and conditional verification suggestions. These are at most five screened "
                "examples, not exhaustive symptom frequencies. Edges mean text similarity, not causality or "
                "verified duplicates. Connectivity may be transitive; not every pair is similar. State this caveat.",
                {
                    "question": "Jelaskan kelompok keluhan terkait ini untuk review tim.",
                    "results": reports,
                },
                answer_schema(reports),
                "graph_explanation",
            )
            answer = render_answer(result, reports)
            batch.audit.log(
                session_id, "graph_explained", count=len(members), examples=len(reports) - 1
            )
            outcome.update(status="ok", mode="groq")
            return answer, reports, "groq"
        except (GuardrailError, ModelUnavailable) as exc:
            batch.audit.log(session_id, "graph_explanation_fallback", reason=str(exc))
            outcome.update(status="fallback", mode="local")
            return local, reports, "fallback"
