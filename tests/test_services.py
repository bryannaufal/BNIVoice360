"""Behavior tests for ingestion, enrichment, analytics, and replaceable adapters."""

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from bnivoice360.domain import CrawlRequest, IssueFilters, Prediction
from bnivoice360.services.analytics import filter_issues, overview_metrics, voice_feed
from bnivoice360.services.companion import answer_question
from bnivoice360.services.crawler import SimulatedCrawler
from bnivoice360.services.prediction import PredictionService


def test_ingestion_reports_invalid_rows_and_duplicate_ids(services):
    frame = pd.DataFrame(
        [
            {
                "suara_nasabah_id": "valid",
                "content": "Hello",
                "date_posted": "2026-10-10",
                "source": "X",
            },
            {"content": None, "date_posted": "2026-10-10", "source": "X"},
            {"content": "Invalid date", "date_posted": "not-a-date", "source": "X"},
            {"content": "Invalid flag", "date_posted": "2026-10-10", "source": "X", "is_IT": 2},
        ]
    )
    result = services.ingestion.ingest(frame)
    assert (result.inserted, result.skipped) == (1, 3)
    assert len(result.errors) == 3
    duplicate = services.ingestion.ingest(frame.iloc[:1])
    assert (duplicate.inserted, duplicate.skipped) == (0, 1)
    assert len(services.repository.load()) == 1


def test_ingestion_maps_source_normalizes_timezone_and_generates_id(services):
    frame = pd.DataFrame(
        [
            {
                "suara_nasabah_id": float("nan"),
                "content": "  QRIS bagus  ",
                "date_posted": "2026-10-10T01:00:00Z",
                "source": "unknown",
            }
        ]
    )
    result = services.ingestion.ingest(frame, forced_source="PlayStore")
    row = services.repository.load().iloc[0]
    assert result.inserted == 1
    assert row["source"] == "PlayStore"
    assert row["content"] == "QRIS bagus"
    assert row["date_posted"] == pd.Timestamp("2026-10-10 08:00:00")
    assert row["suara_nasabah_id"].startswith("VOC-")


def test_missing_columns_fail_before_database_changes(services):
    with pytest.raises(ValueError, match="date_posted"):
        services.ingestion.ingest(pd.DataFrame([{"content": "Incomplete"}]), "X")
    assert services.repository.load().empty


def test_prediction_fills_partial_labels_without_overwriting_reviewed_fields(services):
    services.ingestion.ingest(
        pd.DataFrame(
            [
                {
                    "suara_nasabah_id": "partial",
                    "content": "Transfer gagal, mohon segera",
                    "date_posted": "2026-10-10",
                    "source": "X",
                    "sentiment": "Neutral",
                    "topic_cluster": "Reviewed topic",
                    "is_escalated": 1,
                }
            ]
        )
    )
    assert services.prediction.run("PlayStore") == 0
    assert services.prediction.run("X") == 1
    row = services.repository.load().iloc[0]
    assert row["sentiment"] == "Neutral"
    assert row["topic_cluster"] == "Reviewed topic"
    assert row["is_IT"] == 1
    assert row["is_Urgent"] == 1
    assert row["is_escalated"] == 1
    assert services.prediction.run("X") == 0


def test_classifier_can_be_replaced_without_changing_pages(services):
    class StubClassifier:
        def classify(self, text):
            return Prediction("Positive", 1, 0, 0, 0, "Custom model topic")

    services.ingestion.ingest(
        pd.DataFrame(
            [
                {
                    "content": "Test content",
                    "date_posted": "2026-10-10",
                    "source": "X",
                }
            ]
        )
    )
    service = PredictionService(services.repository, StubClassifier())
    assert service.run("X") == 1
    assert services.repository.load().iloc[0]["topic_cluster"] == "Custom model topic"


def test_crawler_obeys_date_bounds_and_validates_limits():
    request = CrawlRequest("X", limit=12, start_date=date(2026, 10, 1), end_date=date(2026, 10, 3))
    crawler = SimulatedCrawler()
    first = crawler.crawl(request)
    pd.testing.assert_frame_equal(first, crawler.crawl(request))
    assert len(first) == 12
    assert first["date_posted"].dt.date.between(request.start_date, request.end_date).all()
    with pytest.raises(ValueError, match="Limit"):
        crawler.crawl(replace(request, limit=0))


def issue_fixture(services):
    services.ingestion.ingest(
        pd.DataFrame(
            [
                {
                    "suara_nasabah_id": "urgent",
                    "content": "transfer.a",
                    "date_posted": "2026-10-01",
                    "source": "X",
                    "sentiment": "Negative",
                    "is_IT": 1,
                    "is_Urgent": 1,
                    "topic_cluster": "Transfer",
                },
                {
                    "suara_nasabah_id": "recent",
                    "content": "transferxa",
                    "date_posted": "2026-10-10",
                    "source": "PlayStore",
                    "sentiment": "Positive",
                    "is_Bisnis": 1,
                    "is_Urgent": 0,
                    "topic_cluster": "QRIS",
                },
                {
                    "suara_nasabah_id": "escalated",
                    "content": "transfer.a",
                    "date_posted": "2026-10-09",
                    "source": "X",
                    "sentiment": "Negative",
                    "is_IT": 1,
                    "is_Urgent": 1,
                    "is_escalated": 1,
                    "topic_cluster": "Transfer",
                },
            ]
        )
    )
    return services.repository.load()


def test_issue_filters_use_or_for_categories_and_and_for_other_dimensions(services):
    frame = issue_fixture(services)
    selected = filter_issues(
        frame,
        IssueFilters(
            sources=("X",),
            categories=("IT", "Bisnis"),
            escalation_status="Pending Escalation",
            urgent_only=True,
        ),
    )
    assert selected["suara_nasabah_id"].tolist() == ["urgent"]
    assert filter_issues(frame, IssueFilters(sources=())).empty
    assert filter_issues(frame, IssueFilters(categories=())).empty


def test_feed_prioritizes_urgent_pending_and_search_is_literal(services):
    frame = issue_fixture(services)
    assert voice_feed(frame).iloc[0]["suara_nasabah_id"] == "urgent"
    searched = voice_feed(frame, "transfer.a")
    assert set(searched["suara_nasabah_id"]) == {"urgent", "escalated"}


def test_overview_handles_empty_data_and_excludes_unlabeled_from_nss(services):
    empty = overview_metrics(services.repository.load(), now=pd.Timestamp("2026-10-10"))
    assert empty.total_volume == 0
    assert empty.net_sentiment == 0
    services.ingestion.ingest(
        pd.DataFrame(
            [
                {
                    "content": "Great",
                    "date_posted": "2026-10-09",
                    "source": "X",
                    "sentiment": "Positive",
                },
                {"content": "Unlabeled", "date_posted": "2026-10-09", "source": "X"},
            ]
        )
    )
    metrics = overview_metrics(services.repository.load(), now=pd.Timestamp("2026-10-10"))
    assert metrics.total_volume == 2
    assert metrics.net_sentiment == 100


def test_companion_filters_source_without_matching_x_inside_other_words(services):
    frame = issue_fixture(services)
    assert "**1 voice**" in answer_question("How many PlayStore voices?", frame)
    assert "**2 voice**" in answer_question("How many X voices?", frame)
    assert "**3 voice**" in answer_question("How many executive voices?", frame)
    assert "**1 voice**" in answer_question("Berapa urgent IT issue yang belum dieskalasi?", frame)
    top = answer_question(
        "Top 3 negative topic clusters from X this week", frame, now=pd.Timestamp("2026-10-10")
    )
    assert "**Transfer** — 1 voice" in top
