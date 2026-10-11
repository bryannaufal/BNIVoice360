"""Negative scope, category top-five topics, interval peaks, and deep-dive regressions."""

from datetime import date, timedelta

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from bnivoice360.config import PROJECT_ROOT
from bnivoice360.services import issues
from bnivoice360.ui import charts


def diagnostic_frame():
    rows = []
    for topic_id, volume in enumerate([70, 60, 50, 40, 30, 20, 10]):
        for index in range(volume):
            rows.append(
                {
                    "topic_id": topic_id,
                    "topic_cluster": f"Topic {topic_id}",
                    "content": "negative quote",
                    "sentiment": "Negative",
                    "source": "X",
                    "date_posted": pd.Timestamp(date(2026, 1, 1) + timedelta(days=index % 20)),
                    "is_IT": 1,
                    "is_Bisnis": int(topic_id % 2 == 0),
                    "is_Operasional": 0,
                    "is_Urgent": int(index % 2 == 0),
                    "escalation_status": ["Open", "Escalated", "Resolved"][index % 3],
                }
            )
    rows.extend(
        [
            {**rows[0], "topic_id": -1, "topic_cluster": "Outlier"},
            {**rows[0], "topic_id": None, "topic_cluster": "Unlabeled"},
            *[{**rows[0], "sentiment": "Positive", "topic_id": 100, "topic_cluster": "Praise"}]
            * 200,
        ]
    )
    return pd.DataFrame(rows)


def test_negative_metrics_and_category_tiles_show_only_top_five_named_topics():
    frame = diagnostic_frame()
    metrics = issues.issue_metrics(frame)
    assert metrics["total"] == 280
    assert metrics["driver"] == "Topic 0 (70)"
    negative = frame[frame.sentiment.eq("Negative")]
    assert metrics["pending"] == int(
        (negative.is_Urgent.eq(1) & negative.escalation_status.eq("Open")).sum()
    )
    assert metrics["active"] == int(
        (negative.is_Urgent.eq(1) & negative.escalation_status.eq("Escalated")).sum()
    )
    tiles = issues.category_topic_tiles(frame)
    it_tiles = tiles[tiles.Category.eq("IT")]
    assert it_tiles.Volume.sum() == 250
    assert set(it_tiles.topic_id) == set(range(5))
    business_tiles = tiles[tiles.Category.eq("Bisnis")]
    assert business_tiles.Volume.sum() == 160
    assert set(business_tiles.topic_id) == {0, 2, 4, 6}
    assert tiles.groupby("Category").size().le(5).all()
    assert tiles.kind.eq("topic").all()
    assert tiles.topic_id.notna().all()
    assert not tiles.Topic.str.startswith("Lainnya").any()
    assert set(tiles.Category) == {"IT", "Bisnis"}
    assert not tiles.Topic.isin(["Outlier", "Unlabeled", "Praise"]).any()
    # The deep-dive catalog still includes every named topic, including the IT tail.
    assert set(issues.topic_catalog(frame).topic_id) == set(range(7))


@pytest.mark.parametrize(
    "days,expected", [(1, "Daily"), (30, "Daily"), (31, "Weekly"), (90, "Weekly"), (91, "Monthly")]
)
def test_binning_uses_inclusive_filter_range(days, expected):
    start = date(2026, 1, 1)
    assert issues.time_binning(start, start + timedelta(days=days - 1)) == expected


def test_binned_counts_keep_empty_intervals_and_use_calendar_boundaries():
    frame = diagnostic_frame()
    start, end = date(2026, 1, 1), date(2026, 3, 31)
    for binning in ["Daily", "Weekly", "Monthly"]:
        binned = issues.binned_topic_volumes(frame, start, end, binning)
        assert binned.Volume.sum() == 280
        assert binned.Volume.eq(0).any()
        if binning == "Daily":
            assert binned.Date.nunique() == 90
        elif binning == "Weekly":
            assert binned.Date.dt.dayofweek.eq(0).all()
        else:
            assert binned.Date.dt.day.eq(1).all()
            assert binned.Date.nunique() == 3


def test_peak_selection_uses_active_bins_and_never_aggregates_long_tail():
    rows = []
    for topic_id in range(7):
        for index in range(20):
            volume = (
                10
                if topic_id < 4
                else (20 if topic_id == 4 else 15 if topic_id == 5 else 2)
                if index == 0
                else 0
            )
            rows.append(
                {
                    "Date": pd.Timestamp(date(2026, 1, 1) + timedelta(days=index)),
                    "topic_id": topic_id,
                    "Topic": f"Topic {topic_id}",
                    "Volume": volume,
                }
            )
    daily = pd.DataFrame(rows)
    selected = issues.spike_aware_topics(daily)
    assert set(selected.topic_id) == set(range(6))
    weekly = (
        daily.assign(Date=daily.Date.dt.to_period("W-SUN").dt.start_time)
        .groupby(["Date", "topic_id", "Topic"])
        .Volume.sum()
        .reset_index()
    )
    assert set(issues.spike_aware_topics(weekly).topic_id) == set(range(4))
    assert selected.Topic.str.startswith("Topic ").all()
    assert selected.Volume.sum() == daily[daily.topic_id.lt(6)].Volume.sum()


def test_selection_ignores_non_topics_and_handles_duplicate_titles():
    frame = diagnostic_frame()
    frame.loc[frame.topic_id.eq(1), "topic_cluster"] = "Topic 0"
    catalog = issues.topic_catalog(frame)
    assert catalog.Label.nunique() == len(catalog)
    assert (
        issues.selected_topic({"selection": {"points": [{"customdata": [1, "topic"]}]}}, catalog)
        == 1
    )
    for custom in [[None, "other"], [None, "category"], [-1, "topic"], [999, "topic"]]:
        assert (
            issues.selected_topic({"selection": {"points": [{"customdata": custom}]}}, catalog)
            is None
        )
    assert (
        issues.selected_topic({"selection": {"points": [{"label": "IT", "parent": ""}]}}, catalog)
        is None
    )
    label = catalog.loc[catalog.topic_id.eq(1), "Label"].iloc[0]
    assert (
        issues.selected_topic(
            {"selection": {"points": [{"label": label, "parent": "unit:IT"}]}}, catalog
        )
        == 1
    )


def test_summary_overlap_and_fixed_chart_dimensions():
    frame = diagnostic_frame()
    topic = issues.negative_issues(frame[frame.topic_id.eq(0)])
    summary = issues.topic_summary(topic)
    assert summary["volume"] == 70
    assert summary["channel"] == "X (100%)"
    assert summary["coverage"] == "Cross-Unit (IT & Bisnis)"
    overlap = issues.department_involvement(topic)
    assert overlap.Percent.sum() == 200
    trend = issues.spike_aware_topics(
        issues.binned_topic_volumes(frame, date(2026, 1, 1), date(2026, 1, 30), "Daily")
    )
    treemap = charts.issue_topic_treemap(issues.category_topic_tiles(frame))
    lines = charts.topic_trend(trend, "Global topic trend", "Daily")
    assert treemap.layout.height == lines.layout.height == 400
    assert lines.layout.hovermode == "x unified"
    assert len(lines.data) <= 6


def test_page_deep_dive_defaults_persists_and_keeps_negative_only(services, tmp_path, monkeypatch):
    frame = diagnostic_frame()
    frame["suara_nasabah_id"] = [f"row-{index}" for index in range(len(frame))]
    # Ingestion resolves topic titles into topic IDs; omit the synthetic fixture IDs.
    services.ingestion.ingest(frame.drop(columns="topic_id"))
    monkeypatch.setenv("BNIVOICE360_DB_PATH", str(services.repository.database.path))
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    app.sidebar.radio(key="current_page").set_value("Issue Analytics & Voice Feed").run()
    assert not app.exception
    assert app.metric[0].value == "280"
    assert app.metric[3].value == "Topic 0 (70)"
    assert app.metric[4].value == "70"
    chosen = app.selectbox(key="issues_deep_dive_topic")
    labels = (
        services.repository.load()
        .drop_duplicates("topic_id")
        .set_index("topic_id")["topic_cluster"]
    )
    alternate = int(labels[labels.eq("Topic 1")].index[0])
    chosen.set_value(alternate).run()
    assert not app.exception
    assert app.metric[4].value == "60"
    app.sidebar.radio(key="current_page").set_value("Home").run()
    app.sidebar.radio(key="current_page").set_value("Issue Analytics & Voice Feed").run()
    assert app.selectbox(key="issues_deep_dive_topic").value == alternate
    app.multiselect(key="issues_topics").set_value(["Topic 0"]).run()
    assert not app.exception
    assert app.selectbox(key="issues_deep_dive_topic").value != alternate
    app.multiselect(key="issues_sentiments").set_value(["Positive"]).run()
    assert not app.exception
    assert app.metric[0].value == "0"
    assert len(app.expander) == 0


def test_special_only_topics_show_feed_without_deep_dive(services, monkeypatch):
    frame = diagnostic_frame()
    frame = frame[frame.topic_cluster.isin(["Outlier", "Unlabeled"])].drop(columns="topic_id")
    services.ingestion.ingest(frame)
    monkeypatch.setenv("BNIVOICE360_DB_PATH", str(services.repository.database.path))
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    app.sidebar.radio(key="current_page").set_value("Issue Analytics & Voice Feed").run()
    assert not app.exception
    assert app.metric[0].value == "0"
    assert len(app.expander) == 2
    assert not any(widget.key == "issues_deep_dive_topic" for widget in app.selectbox)
