"""Negative issue diagnostics; each voice counts once outside unit overlap charts."""

from datetime import date

import pandas as pd

from bnivoice360.config import CATEGORY_COLUMNS
from bnivoice360.services.analytics import assigned_topics, category_labels, open_urgent_count


def negative_issues(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["sentiment"].eq("Negative")].copy()


def topic_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    topics = (
        assigned_topics(negative_issues(frame))
        .groupby(["topic_id", "topic_cluster"])
        .size()
        .reset_index(name="Volume")
        .sort_values(["Volume", "topic_id"], ascending=[False, True])
        .rename(columns={"topic_cluster": "Topic"})
    )
    duplicates = topics["Topic"].duplicated(keep=False)
    topics["Label"] = topics["Topic"]
    if duplicates.any():
        topics.loc[duplicates, "Label"] = topics.loc[duplicates].apply(
            lambda row: f"{row['Topic']} (#{int(row['topic_id'])})", axis=1
        )
    return topics


def issue_metrics(frame: pd.DataFrame) -> dict:
    frame = negative_issues(frame)
    topics = topic_catalog(frame)
    return {
        "total": int(topics["Volume"].sum()),
        "pending": open_urgent_count(frame),
        "active": int(
            (frame["is_Urgent"].eq(1) & frame["escalation_status"].eq("Escalated")).sum()
        ),
        "driver": f"{topics.iloc[0]['Label']} ({topics.iloc[0]['Volume']})" if len(topics) else "—",
    }


def category_topic_tiles(frame: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Show each unit's largest named negative topics without aggregating the tail."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    catalog = topic_catalog(frame)
    labels = catalog.set_index("topic_id")["Label"].to_dict()
    frame = assigned_topics(negative_issues(frame))
    rows = []
    for category, flag in CATEGORY_COLUMNS.items():
        volumes = frame.loc[frame[flag].eq(1)].groupby("topic_id").size()
        ranked = (
            volumes.rename("Volume")
            .reset_index()
            .sort_values(["Volume", "topic_id"], ascending=[False, True])
        )
        for _, row in ranked.head(top_n).iterrows():
            rows.append(
                {
                    "Category": category,
                    "Topic": labels[row["topic_id"]],
                    "Volume": int(row["Volume"]),
                    "topic_id": int(row["topic_id"]),
                    "kind": "topic",
                }
            )
    return pd.DataFrame(rows, columns=["Category", "Topic", "Volume", "topic_id", "kind"])


def time_binning(start: date, end: date) -> str:
    days = (end - start).days + 1
    return "Daily" if days <= 30 else "Weekly" if days <= 90 else "Monthly"


def binned_topic_volumes(frame: pd.DataFrame, start: date, end: date, binning: str) -> pd.DataFrame:
    """Use Monday-start weeks/month starts and include zero-count intervals."""
    frequencies = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "M"}
    frequency = frequencies[binning]
    frame = assigned_topics(negative_issues(frame))
    frame = frame.loc[
        frame["date_posted"].notna() & frame["date_posted"].dt.date.between(start, end)
    ]
    catalog = topic_catalog(frame)
    if catalog.empty:
        return pd.DataFrame(columns=["Date", "topic_id", "Topic", "Volume"])
    periods = frame["date_posted"].dt.to_period(frequency).dt.start_time
    counts = frame.assign(Date=periods).groupby(["Date", "topic_id"]).size()
    intervals = pd.period_range(start, end, freq=frequency).start_time
    grid = pd.MultiIndex.from_product([intervals, catalog["topic_id"]], names=["Date", "topic_id"])
    result = counts.reindex(grid, fill_value=0).rename("Volume").reset_index()
    return result.merge(
        catalog[["topic_id", "Label"]].rename(columns={"Label": "Topic"}), on="topic_id"
    )


def spike_aware_topics(binned: pd.DataFrame) -> pd.DataFrame:
    """Union of four largest totals and two largest peaks in the active interval size."""
    if binned.empty:
        return binned.copy()
    stats = binned.groupby("topic_id")["Volume"].agg(Total="sum", Peak="max").reset_index()
    totals = stats.sort_values(["Total", "topic_id"], ascending=[False, True]).head(4)
    peaks = stats.sort_values(["Peak", "Total", "topic_id"], ascending=[False, False, True]).head(2)
    selected = list(dict.fromkeys([*totals["topic_id"], *peaks["topic_id"]]))
    result = binned.loc[binned["topic_id"].isin(selected)].copy()
    order = {topic_id: index for index, topic_id in enumerate(selected)}
    return (
        result.assign(order=result["topic_id"].map(order))
        .sort_values(["order", "Date"])
        .drop(columns="order")
    )


def department_involvement(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Department": label,
                "Percent": float(frame[column].eq(1).mean() * 100) if len(frame) else 0,
            }
            for label, column in CATEGORY_COLUMNS.items()
        ]
    )


def topic_summary(frame: pd.DataFrame) -> dict:
    sources = frame["source"].value_counts()
    units = [label for label, column in CATEGORY_COLUMNS.items() if frame[column].eq(1).any()]
    cross_unit = frame.apply(category_labels, axis=1).map(len).gt(1).any() if len(frame) else False
    coverage = (
        f"Cross-Unit ({' & '.join(units)})"
        if cross_unit
        else f"Multi-Unit ({' & '.join(units)})"
        if len(units) > 1
        else f"{units[0]} Only"
        if units
        else "Uncategorized"
    )
    return {
        "volume": len(frame),
        "statuses": f"{frame['escalation_status'].eq('Open').sum()} Open / {frame['escalation_status'].eq('Escalated').sum()} Escalated",
        "channel": f"{sources.index[0]} ({sources.iloc[0] / len(frame):.0%})"
        if len(sources)
        else "—",
        "coverage": coverage,
    }


def selected_topic(event: dict, catalog: pd.DataFrame) -> int | None:
    """Ignore category parents and selections that do not identify a real topic."""
    points = event.get("selection", {}).get("points", [])
    if not points:
        return None
    point = points[-1]
    custom = point.get("customdata")
    if isinstance(custom, (list, tuple)) and len(custom) >= 2:
        if custom[1] != "topic" or custom[0] is None:
            return None
        try:
            topic_id = int(custom[0])
        except (ValueError, TypeError):
            return None
        return topic_id if topic_id in set(catalog["topic_id"]) else None
    # Streamlit exposes labels and parents even when customdata isn't returned.
    label = point.get("label")
    if not point.get("parent"):
        return None
    matches = catalog.loc[catalog["Label"].eq(label), "topic_id"]
    return int(matches.iloc[0]) if len(matches) == 1 else None
