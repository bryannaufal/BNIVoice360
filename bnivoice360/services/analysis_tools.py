"""Allowlisted read-only analysis over one database snapshot per conversation turn."""

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from bnivoice360.config import CATEGORY_COLUMNS, ESCALATION_STATUSES, SENTIMENTS, SOURCES
from bnivoice360.services.agent.audit import fingerprint
from bnivoice360.services.agent.guardrails import TOXIC, GuardrailError, screen_input
from bnivoice360.services.agent.retrieval import similar_voices
from bnivoice360.services.agent.workflow import _rows
from bnivoice360.services.analytics import assigned_topics, topic_assignment_counts
from bnivoice360.services.companion_intent import object_schema, validate_schema

PERIODS = ["7d", "30d", "today", "this_week", "this_month", "all", "custom"]
FILTER_PROPERTIES = {
    "period": {"type": "string", "enum": PERIODS},
    "start": {"type": ["string", "null"]},
    "end": {"type": ["string", "null"]},
    "keyword": {"type": ["string", "null"]},
    "topic": {"type": ["string", "null"]},
    "category": {"type": ["string", "null"], "enum": [*CATEGORY_COLUMNS, None]},
    "source": {"type": ["string", "null"], "enum": [*SOURCES, None]},
    "status": {"type": "string", "enum": ["All", *ESCALATION_STATUSES]},
    "sentiment": {"type": "string", "enum": ["All", *SENTIMENTS]},
    "urgent_only": {"type": "boolean"},
}
GROUPS = ["topic", "source", "category", "status", "sentiment"]


def definition(name, description, properties):
    schema = object_schema(properties)
    if "period" in properties:
        # Read filters have application-owned defaults; models need only provide overrides.
        schema["required"] = [key for key in ("group_by", "anchor") if key in properties]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


TOOL_DEFINITIONS = [
    definition("dataset_coverage", "Read available dates, missing dates, and dataset size.", {}),
    definition(
        "volume_breakdown",
        "Count voices by topic/channel/category/status/sentiment. Topic rankings exclude Outlier/Unlabeled.",
        {
            **FILTER_PROPERTIES,
            "group_by": {"type": "string", "enum": GROUPS},
            "limit": {"type": "integer"},
        },
    ),
    definition(
        "compare_periods",
        "Compare with the immediately preceding equal-length period, including topics absent in either period. No causal conclusions.",
        {
            **FILTER_PROPERTIES,
            "group_by": {"type": "string", "enum": ["topic", "source", "category", "all"]},
            "limit": {"type": "integer"},
        },
    ),
    definition(
        "daily_trend",
        "Daily counts including zero days; maximum ninety days. Does not infer incidents or causes.",
        FILTER_PROPERTIES,
    ),
    definition(
        "backlog_age",
        "Rank TOPICS by Open/Escalated complaint counts and oldest posting-age; also overall totals/median age. NOT time in status or SLA. All means both active statuses; default Open. No user period means all posting dates.",
        {**FILTER_PROPERTIES, "limit": {"type": "integer"}},
    ),
    definition(
        "complaint_examples",
        "Read recent representative excerpts (maximum five), with aliases and metadata. Samples do not establish causes or complete subtopic frequencies.",
        {**FILTER_PROPERTIES, "limit": {"type": "integer"}},
    ),
    definition(
        "similar_complaints",
        "Existing voice embeddings: compare an authorized example/selected alias against Open voices within its topic. Does not prove a shared cause.",
        {**FILTER_PROPERTIES, "anchor": {"type": "string"}, "limit": {"type": "integer"}},
    ),
    definition(
        "decline_request",
        "Refuse questions outside complaint analytics, unsupported actions, or unavailable evidence (SOP, internal checks, real-time system health, costs, SLA/history).",
        {
            "reason": {
                "type": "string",
                "enum": ["outside_scope", "missing_data", "unsupported_action"],
            }
        },
    ),
    definition(
        "clarify_question",
        "Ask one concise follow-up only when the required scope is ambiguous.",
        {"question": {"type": "string"}},
    ),
]
DEFINITIONS = {item["function"]["name"]: item["function"] for item in TOOL_DEFINITIONS}


def default_scope() -> dict:
    return dict(
        period="7d",
        start=None,
        end=None,
        keyword=None,
        topic=None,
        category=None,
        source=None,
        status="All",
        sentiment="Negative",
        urgent_only=False,
    )


def safe_data(value):
    """Screen every string, including topic labels, before it crosses the provider boundary."""
    if isinstance(value, dict):
        return {key: safe_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_data(item) for item in value]
    if isinstance(value, str):
        try:
            clean = screen_input(value)
            if TOXIC.search(clean):
                raise GuardrailError("unsafe_source_text")
            return clean
        except GuardrailError:
            return "[teks tidak ditampilkan]"
    return value


@dataclass
class AnalysisEvidence:
    ticket: str
    digest: str
    scope: dict


class AnalysisTools:
    def __init__(
        self,
        batch,
        today: date,
        question: str,
        previous_scope: dict,
        selected_rows: dict,
    ):
        self.batch = batch
        self.today = today
        self.question = question
        self.previous_scope = previous_scope
        self._frame = None
        self.evidence: dict[str, AnalysisEvidence] = {}
        self.scope = previous_scope.copy() or default_scope()
        self.selected_rows = selected_rows
        self.known_terms: set[str] = set()
        self.omitted_texts = 0

    @property
    def frame(self):
        if self._frame is None:
            self._frame = self.batch.repository.load()
        return self._frame

    def dates(self, args):
        period = args["period"]
        if period == "custom":
            try:
                first, last = date.fromisoformat(args["start"]), date.fromisoformat(args["end"])
            except (TypeError, ValueError):
                raise GuardrailError("invalid_analysis_dates") from None
            if first > last or last > self.today or (last - first).days > 3650:
                raise GuardrailError("invalid_analysis_dates")
            return first, last
        if args["start"] is not None or args["end"] is not None:
            raise GuardrailError("unexpected_analysis_dates")
        if period == "all":
            return None, self.today
        first = {
            "7d": self.today - timedelta(days=6),
            "30d": self.today - timedelta(days=29),
            "today": self.today,
            "this_week": self.today - timedelta(days=self.today.weekday()),
            "this_month": self.today.replace(day=1),
        }[period]
        return first, self.today

    def filtered(self, args, *, dates=None):
        first, last = self.dates(args) if dates is None else dates
        frame = self.frame.copy()
        # Missing dates remain visible for all-time totals only; future-dated records are excluded.
        mask = frame["date_posted"].isna() if first is None else pd.Series(False, index=frame.index)
        dated = frame["date_posted"].dt.date.le(last)
        if first:
            dated &= frame["date_posted"].dt.date.ge(first)
        frame = frame.loc[mask | dated]
        for parameter, column in (
            ("source", "source"),
            ("status", "escalation_status"),
            ("sentiment", "sentiment"),
            ("topic", "topic_cluster"),
        ):
            value = args[parameter]
            if value and value != "All":
                frame = frame.loc[frame[column].eq(value)]
        if args["category"]:
            frame = frame.loc[frame[CATEGORY_COLUMNS[args["category"]]].eq(1)]
        if args["urgent_only"]:
            frame = frame.loc[frame["is_Urgent"].eq(1)]
        if args["keyword"]:
            keyword = args["keyword"]
            frame = frame.loc[
                frame["content"].str.contains(keyword, case=False, regex=False, na=False)
                | frame["topic_cluster"].str.contains(keyword, case=False, regex=False, na=False)
            ]
        return frame

    def validate_arguments(self, name, args):
        if name not in DEFINITIONS:
            raise GuardrailError("unauthorized_analysis_tool")
        validate_schema(args, DEFINITIONS[name]["parameters"])
        if "limit" in args and not 1 <= args["limit"] <= (
            5 if name in {"complaint_examples", "similar_complaints"} else 10
        ):
            raise GuardrailError("analysis_result_budget")
        for field in ("keyword", "topic"):
            value = args.get(field)
            if value is not None:
                if not value.strip() or len(value) > 120:
                    raise GuardrailError("analysis_filter_size")
                screen_input(value)
        if "period" in args:
            self.dates(args)

    def complete_arguments(self, name, args):
        if name not in DEFINITIONS:
            raise GuardrailError("unauthorized_analysis_tool")
        validate_schema(args, DEFINITIONS[name]["parameters"])
        if "period" not in DEFINITIONS[name]["parameters"]["properties"]:
            return args
        values = {**default_scope(), **self.previous_scope}
        if "period" in args:
            values.update(start=None, end=None)
        if "limit" in DEFINITIONS[name]["parameters"]["properties"]:
            values["limit"] = 5
        return {**values, **args}

    def grouped(self, frame, group):
        if group == "all":
            return {"Semua keluhan": len(frame)}
        if group == "topic":
            return {
                str(key): int(count)
                for key, count in assigned_topics(frame)["topic_cluster"].value_counts().items()
            }
        if group == "category":
            counts = {
                key: int(frame[column].eq(1).sum()) for key, column in CATEGORY_COLUMNS.items()
            }
            counts["Tanpa kategori"] = int(
                (~frame[list(CATEGORY_COLUMNS.values())].eq(1).any(axis=1)).sum()
            )
            return counts
        column = {"source": "source", "status": "escalation_status", "sentiment": "sentiment"}[
            group
        ]
        return {
            str(key): int(count)
            for key, count in frame[column].fillna("Belum dilabeli").value_counts().items()
        }

    def examples(self, frame, reference, scope, limit):
        records = []
        # Screen full content before truncating. Do not hide a rejected suffix in a safe-looking prefix.
        for _, row in frame.sort_values(
            ["date_posted", "suara_nasabah_id"], ascending=[False, True]
        ).iterrows():
            try:
                text = screen_input(str(row["content"]))
                if TOXIC.search(text):
                    raise GuardrailError("unsafe_source_text")
            except GuardrailError:
                self.omitted_texts += 1
                continue
            alias = f"{reference}.E{len(records) + 1}"
            with self.batch.repository.database.transaction() as connection:
                snapshot = _rows(connection, (str(row["suara_nasabah_id"]),)).get(
                    str(row["suara_nasabah_id"])
                )
            if snapshot is None:
                raise GuardrailError("analysis_evidence_changed")
            for key, after in snapshot.items():
                before = row[key]
                if key == "date_posted":
                    after = pd.to_datetime(after)
                if pd.isna(before) and pd.isna(after):
                    continue
                if pd.isna(before) or pd.isna(after) or before != after:
                    raise GuardrailError("analysis_evidence_changed")
            self.evidence[alias] = AnalysisEvidence(
                str(row["suara_nasabah_id"]),
                fingerprint(snapshot),
                {**scope, "as_of": self.today.isoformat()},
            )
            records.append(
                dict(
                    alias=alias,
                    excerpt=text[:700],
                    posted=row["date_posted"].date().isoformat()
                    if pd.notna(row["date_posted"])
                    else None,
                    source=row["source"],
                    topic=row["topic_cluster"],
                    status=row["escalation_status"],
                    urgent=bool(row["is_Urgent"] == 1),
                )
            )
            if len(records) == limit:
                break
        return records

    def execute(self, name, args, reference):
        args = self.complete_arguments(name, args)
        self.validate_arguments(name, args)
        if name == "dataset_coverage":
            frame = self.frame
            dates = frame["date_posted"].dropna()
            return safe_data(
                dict(
                    total=len(frame),
                    earliest=dates.min().date().isoformat() if len(dates) else None,
                    latest=dates.max().date().isoformat() if len(dates) else None,
                    missing_dates=int(frame["date_posted"].isna().sum()),
                    negative=int(frame["sentiment"].eq("Negative").sum()),
                    future_dates=int(frame["date_posted"].dt.date.gt(self.today).sum()),
                )
            )
        scope = {key: args[key] for key in FILTER_PROPERTIES}
        self.scope = scope
        frame = self.filtered(args)
        first, last = self.dates(args)
        result = dict(
            scope={
                **scope,
                "start_date": first.isoformat() if first else None,
                "end_date": last.isoformat(),
                "days": (last - first).days + 1 if first else None,
            },
            total=len(frame),
            **topic_assignment_counts(frame),
        )
        result["status_counts"] = {
            status: int(frame["escalation_status"].eq(status).sum())
            for status in ESCALATION_STATUSES
        }
        if name == "volume_breakdown":
            counts = self.grouped(frame, args["group_by"])
            ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            result.update(
                group_by=args["group_by"],
                items=[
                    dict(
                        label=key,
                        count=count,
                        share_pct=round(count / len(frame) * 100, 2) if len(frame) else 0,
                    )
                    for key, count in ordered[: args["limit"]]
                ],
                omitted_groups=max(0, len(ordered) - args["limit"]),
            )
            if args["group_by"] == "category":
                result["note"] = (
                    "Satu keluhan dapat masuk beberapa kategori; jumlah kategori dapat melebihi total."
                )
        elif name == "compare_periods":
            if first is None:
                raise GuardrailError("comparison_requires_finite_period")
            width = (last - first).days + 1
            previous_start, previous_end = first - timedelta(days=width), first - timedelta(days=1)
            previous = self.filtered(args, dates=(previous_start, previous_end))
            current_counts, previous_counts = (
                self.grouped(frame, args["group_by"]),
                self.grouped(previous, args["group_by"]),
            )
            changes = []
            for label in current_counts.keys() | previous_counts.keys():
                current, prior = current_counts.get(label, 0), previous_counts.get(label, 0)
                changes.append(
                    dict(
                        label=label,
                        current=current,
                        previous=prior,
                        delta=current - prior,
                        growth_pct=round((current - prior) / prior * 100, 2) if prior else None,
                    )
                )
            changes.sort(key=lambda item: (-item["delta"], -item["current"], item["label"]))
            result.update(
                previous_start=previous_start.isoformat(),
                previous_end=previous_end.isoformat(),
                previous_total=len(previous),
                delta=len(frame) - len(previous),
                growth_pct=round((len(frame) - len(previous)) / len(previous) * 100, 2)
                if len(previous)
                else None,
                items=changes[: args["limit"]],
                omitted_groups=max(0, len(changes) - args["limit"]),
                note="Perbandingan dua rentang dengan jumlah hari sama; baseline nol tidak memiliki persen pertumbuhan. Ini bukan bukti masalah baru atau sebab teknis.",
            )
        elif name == "daily_trend":
            if first is None or (last - first).days >= 90:
                raise GuardrailError("trend_requires_maximum_ninety_days")
            counts = frame.groupby(frame["date_posted"].dt.date).size()
            days = [
                dict(
                    day=(first + timedelta(days=i)).isoformat(),
                    count=int(counts.get(first + timedelta(days=i), 0)),
                )
                for i in range((last - first).days + 1)
            ]
            peak = max(days, key=lambda day: day["count"])
            result.update(
                days=days,
                peak_day=peak["day"],
                peak_count=peak["count"],
                note="Puncak volume laporan tidak menetapkan waktu kejadian gangguan atau penyebabnya.",
            )
        elif name == "backlog_age":
            if args["status"] == "Resolved":
                raise GuardrailError("backlog_requires_active_status")
            active = frame.loc[
                frame["escalation_status"].isin(
                    ["Open", "Escalated"] if args["status"] == "All" else [args["status"]]
                )
            ].copy()
            age = (pd.Timestamp(self.today) - active["date_posted"].dt.normalize()).dt.days
            known = age.dropna()
            topics = []
            for topic, rows in assigned_topics(active).groupby("topic_cluster"):
                ages = (
                    pd.Timestamp(self.today) - rows["date_posted"].dt.normalize()
                ).dt.days.dropna()
                topics.append(
                    dict(
                        label=topic,
                        count=len(rows),
                        oldest_days=int(ages.max()) if len(ages) else None,
                        urgent=int(rows["is_Urgent"].eq(1).sum()),
                    )
                )
            topics.sort(
                key=lambda item: (-item["count"], -(item["oldest_days"] or 0), item["label"])
            )
            result.update(topic_assignment_counts(active))
            result["status_counts"] = {
                status: int(active["escalation_status"].eq(status).sum())
                for status in ESCALATION_STATUSES
            }
            result.update(
                total=len(active),
                open=int(active["escalation_status"].eq("Open").sum()),
                escalated=int(active["escalation_status"].eq("Escalated").sum()),
                urgent=int(active["is_Urgent"].eq(1).sum()),
                known_age=len(known),
                unknown_age=len(active) - len(known),
                oldest_days=int(known.max()) if len(known) else None,
                median_days=float(known.median()) if len(known) else None,
                mean_days=round(float(known.mean()), 2) if len(known) else None,
                items=topics[: args["limit"]],
                omitted_groups=max(0, len(topics) - args["limit"]),
                oldest_named_topic=max(
                    topics,
                    key=lambda item: item["oldest_days"] if item["oldest_days"] is not None else -1,
                )["label"]
                if topics
                else None,
                note="Umur sejak tanggal posting hingga tanggal analisis, bukan durasi Open/Escalated, waktu penyelesaian, atau pelanggaran SLA.",
            )
        elif name == "complaint_examples":
            result.update(
                examples=self.examples(frame, reference, scope, args["limit"]),
                omitted_unsafe_texts=self.omitted_texts,
                note="Cuplikan terbaru yang lolos screening; bukan seluruh keluhan atau pengelompokan akar penyebab.",
            )
        elif name == "similar_complaints":
            alias = args["anchor"]
            ticket = (
                self.evidence[alias].ticket
                if alias in self.evidence
                else self.selected_rows.get(alias)
            )
            if not ticket:
                raise GuardrailError("unauthorized_analysis_anchor")
            anchor = self.frame.loc[self.frame["suara_nasabah_id"].eq(ticket)]
            if anchor.empty or anchor.iloc[0]["escalation_status"] != "Open":
                raise GuardrailError("similarity_requires_open_anchor")
            topic = anchor.iloc[0]["topic_id"]
            if pd.isna(topic) or topic == -1:
                raise GuardrailError("similarity_requires_named_topic")
            candidates = frame.loc[frame["topic_id"].eq(topic)]
            matches = similar_voices(
                self.batch.settings.embedding_directory / "data_id_embedding_mapping.parquet",
                candidates,
                ticket,
                self.batch.settings.agent_similarity_threshold,
                args["limit"],
            )
            result.update(
                examples=self.examples(matches, reference, scope, args["limit"]),
                returned=len(matches),
                missing_embeddings=matches.attrs.get("missing_embeddings", 0),
                threshold=self.batch.settings.agent_similarity_threshold,
                note="Kandidat Open dalam topik yang sama; kemiripan tidak membuktikan penyebab atau kejadian yang sama. Hasil dibatasi, bukan hitungan seluruh pasangan serupa.",
            )
        else:
            raise GuardrailError("unauthorized_analysis_tool")
        for item in result.get("items", []):
            self.known_terms.add(str(item["label"]))
        return safe_data(result)
