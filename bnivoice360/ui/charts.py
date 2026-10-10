"""Responsive Plotly figures with explicit light-surface contrast."""

import textwrap
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

TEXT_COLOR = "#22313F"
GRID_COLOR = "#E4EAED"
SENTIMENT_COLORS = {
    "Negative": "#C0392B",
    "Neutral": "#64748B",
    "Positive": "#167A45",
    "Unlabeled": "#94A3B8",
}
TOPIC_COLORS = [
    "#006666",
    "#B84300",
    "#246B9C",
    "#7D3C98",
    "#947000",
    "#167A45",
    "#444B70",
    "#87572B",
]
# Pale fills allow one consistent dark text color on hierarchy tiles.
CATEGORY_COLORS = {"IT": "#DCECF7", "Bisnis": "#EDE2F5", "Operasional": "#FFF0CC"}


def style_figure(figure: go.Figure, height: int = 390) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        autosize=True,
        height=height,
        margin=dict(l=24, r=28, t=62, b=52, pad=8, autoexpand=True),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Arial", size=12, color=TEXT_COLOR),
        title=dict(
            font=dict(size=16, color=TEXT_COLOR),
            x=0.03,
            xanchor="left",
            y=0.96,
            yanchor="top",
            automargin=True,
        ),
        legend=dict(
            title_text="",
            font=dict(color=TEXT_COLOR),
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            bgcolor="#FFFFFF",
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=TEXT_COLOR, bordercolor="#CBD5E1"),
        uniformtext=dict(minsize=10, mode="hide"),
    )
    figure.update_xaxes(
        automargin=True,
        tickfont=dict(color=TEXT_COLOR),
        title_font=dict(color=TEXT_COLOR),
        gridcolor=GRID_COLOR,
        linecolor="#CBD5E1",
        zerolinecolor=GRID_COLOR,
    )
    figure.update_yaxes(
        automargin=True,
        tickfont=dict(color=TEXT_COLOR),
        title_font=dict(color=TEXT_COLOR),
        gridcolor=GRID_COLOR,
        linecolor="#CBD5E1",
        zerolinecolor=GRID_COLOR,
    )
    figure.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color=TEXT_COLOR),
            title=dict(font=dict(color=TEXT_COLOR)),
            outlinewidth=0,
            thickness=14,
        )
    )
    return figure


def render_chart(figure: go.Figure, key: str) -> None:
    """Prevent Streamlit's chart theme from overriding explicit accessible colors."""
    st.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        config={"responsive": True, "displaylogo": False, "scrollZoom": False},
        key=key,
    )


def _wrap_label(label: object, limit: int = 24) -> str:
    """Wrap long category labels without changing data keys."""
    return "<br>".join(escape(line) for line in textwrap.wrap(str(label), width=limit))


def _topic_axis(figure: go.Figure, topics: pd.Series) -> None:
    labels = topics.drop_duplicates().tolist()
    figure.update_yaxes(
        title=None,
        tickmode="array",
        tickvals=labels,
        ticktext=[_wrap_label(label) for label in labels],
        automargin=True,
    )


def channel_mix(data: pd.DataFrame) -> go.Figure:
    figure = px.pie(
        data,
        names="source",
        values="Volume",
        hole=0.64,
        title="Channel mix",
        color_discrete_sequence=TOPIC_COLORS,
    )
    # Keep channel names in the wrapping legend instead of outside the donut.
    figure.update_traces(
        textposition="inside",
        textinfo="percent",
        insidetextfont=dict(color="#FFFFFF", size=12),
        marker=dict(line=dict(color="#FFFFFF", width=2)),
        hovertemplate="%{label}<br>%{value} voices · %{percent}<extra></extra>",
        sort=False,
    )
    style_figure(figure, 430)
    figure.update_layout(margin=dict(l=20, r=20, t=60, b=90), legend=dict(y=-0.04))
    return figure


def sentiment_signal(data: pd.DataFrame) -> go.Figure:
    figure = px.line(
        data,
        x="day",
        y="Volume",
        color="sentiment",
        markers=True,
        title="Daily sentiment signal",
        color_discrete_map=SENTIMENT_COLORS,
    )
    style_figure(figure, 430)
    figure.update_xaxes(title=None, nticks=7, tickformat="%d %b", tickangle=0)
    figure.update_yaxes(title=None, rangemode="tozero")
    figure.update_layout(margin=dict(b=85))
    return figure


def top_issue_clusters(data: pd.DataFrame) -> go.Figure:
    figure = px.bar(
        data.sort_values("Volume"),
        x="Volume",
        y="topic_cluster",
        orientation="h",
        color="Negative",
        color_continuous_scale=["#FDE2D8", "#E55300", "#A8241A"],
        title="Top issue clusters",
    )
    style_figure(figure, max(380, 150 + len(data) * 46))
    _topic_axis(figure, data["topic_cluster"])
    figure.update_xaxes(title="Voice volume", rangemode="tozero")
    figure.update_coloraxes(colorbar_tickformat=".0%", colorbar_title="Negative share")
    return figure


def category_topic_map(data: pd.DataFrame) -> go.Figure:
    figure = px.treemap(
        data,
        path=["Category", "Topic"],
        values="Volume",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        title="Category → Topic map",
    )
    style_figure(figure, 430)
    figure.update_traces(
        textfont=dict(color=TEXT_COLOR, size=12),
        insidetextfont=dict(color=TEXT_COLOR),
        outsidetextfont=dict(color=TEXT_COLOR),
        marker=dict(line=dict(color="#FFFFFF", width=2)),
        root=dict(color="#F1F5F9"),
    )
    figure.update_layout(margin=dict(l=8, r=8, t=60, b=12))
    return figure


def topic_sentiment_distribution(data: pd.DataFrame) -> go.Figure:
    figure = px.bar(
        data,
        x="Volume",
        y="topic_cluster",
        color="sentiment",
        orientation="h",
        barmode="stack",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment inside topic clusters",
    )
    count = data["topic_cluster"].nunique()
    style_figure(figure, max(430, 170 + count * 40))
    _topic_axis(figure, data["topic_cluster"])
    figure.update_xaxes(title="Voice volume", rangemode="tozero")
    figure.update_layout(margin=dict(b=85))
    return figure
