"""Readability and label-layout regressions for every dashboard figure."""

import pandas as pd
import pytest

from bnivoice360.services import analytics
from bnivoice360.ui import charts


def _luminance(color):
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return sum(
        value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
    )


def _contrast(foreground, background):
    light, dark = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


@pytest.mark.parametrize(
    "foreground,background",
    [
        (charts.TEXT_COLOR, "#FFFFFF"),
        *[(charts.TEXT_COLOR, fill) for fill in charts.CATEGORY_COLORS.values()],
        *[("#FFFFFF", fill) for fill in charts.TOPIC_COLORS],
        ("#52636A", "#FFFFFF"),
        ("#9F2C20", "#FFF0EC"),
        ("#166534", "#EAF7EF"),
        ("#4B5563", "#EFF2F6"),
        ("#1E4D75", "#E8F2FA"),
        ("#663399", "#F1EAF7"),
        ("#785400", "#FFF4D6"),
        ("#005A5A", "#E6F3F3"),
    ],
)
def test_chart_and_badge_text_pairs_have_readable_contrast(foreground, background):
    assert _contrast(foreground, background) >= 4.5


def test_all_figures_define_font_background_and_auto_margins(services):
    services.ingestion.ingest(
        pd.DataFrame(
            [
                {
                    "content": "Transfer gagal",
                    "date_posted": "2026-10-10",
                    "source": "X",
                    "sentiment": "Negative",
                    "is_IT": 1,
                    "is_Urgent": 1,
                    "topic_cluster": "A very long customer topic label that should wrap safely",
                }
            ]
        )
    )
    frame = services.repository.load()
    figures = [
        charts.channel_mix(analytics.channel_volume(frame)),
        charts.sentiment_signal(analytics.sentiment_trend(frame)),
        charts.top_issue_clusters(analytics.top_topics(frame)),
        charts.category_topic_map(analytics.category_topics(frame)),
        charts.topic_sentiment_distribution(analytics.topic_sentiments(frame)),
    ]
    for figure in figures:
        assert figure.layout.autosize is True
        assert figure.layout.width is None
        assert figure.layout.paper_bgcolor == "#FFFFFF"
        assert figure.layout.font.color == charts.TEXT_COLOR
        assert figure.layout.legend.font.color == charts.TEXT_COLOR
        assert figure.layout.hoverlabel.font.color == charts.TEXT_COLOR
        assert figure.layout.xaxis.automargin is True
        assert figure.layout.yaxis.tickfont.color == charts.TEXT_COLOR
    assert figures[0].data[0].textposition == "outside"
    assert figures[3].data[0].textfont.color == charts.TEXT_COLOR
    assert "<br>" in figures[4].layout.yaxis.ticktext[0]
    assert figures[4].layout.height >= 430
