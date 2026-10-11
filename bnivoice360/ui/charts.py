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


def issue_topic_treemap(data: pd.DataFrame) -> go.Figure:
    ids, labels, parents, values, custom, colors = [], [], [], [], [], []
    for category, tiles in data.groupby("Category", sort=False):
        root = f"unit:{category}"
        ids.append(root)
        labels.append(category)
        parents.append("")
        values.append(int(tiles["Volume"].sum()))
        custom.append([None, "category"])
        colors.append(CATEGORY_COLORS[category])
        for position, (_, tile) in enumerate(tiles.iterrows()):
            ids.append(f"{root}/tile:{position}")
            labels.append(tile["Topic"])
            parents.append(root)
            values.append(int(tile["Volume"]))
            custom.append(
                [int(tile["topic_id"]) if tile["kind"] == "topic" else None, tile["kind"]]
            )
            colors.append(CATEGORY_COLORS[category])
    figure = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            customdata=custom,
            text=[_wrap_label(label, 26) for label in labels],
            texttemplate="%{text}<br><b>%{value:,}</b>",
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            hovertemplate="%{label}<br>%{value:,} issue voices<extra></extra>",
            textfont=dict(color=TEXT_COLOR, size=12),
            insidetextfont=dict(color=TEXT_COLOR),
            outsidetextfont=dict(color=TEXT_COLOR),
            pathbar=dict(visible=False),
        )
    )
    style_figure(figure, 400)
    figure.update_layout(title="Category → Topic map", margin=dict(l=8, r=8, t=55, b=8))
    return figure


def topic_trend(data: pd.DataFrame, title: str, binning: str) -> go.Figure:
    figure = go.Figure()
    for topic_id, rows in data.groupby("topic_id", sort=False):
        label = rows.iloc[0]["Topic"]
        figure.add_trace(
            go.Scatter(
                x=rows["Date"],
                y=rows["Volume"],
                mode="lines+markers",
                name=textwrap.shorten(label, width=38, placeholder="…"),
                customdata=[[label]] * len(rows),
                line=dict(color=TOPIC_COLORS[int(topic_id) % len(TOPIC_COLORS)], width=2.5),
                marker=dict(size=5),
                hovertemplate="%{customdata[0]}: %{y:,}<extra></extra>",
            )
        )
    style_figure(figure, 400)
    figure.update_layout(
        title=title,
        hovermode="x unified",
        margin=dict(l=25, r=16, t=55, b=110),
        legend=dict(y=-0.22, font=dict(size=10)),
    )
    figure.update_xaxes(
        title=f"{binning} intervals",
        nticks=6,
        tickformat="%b %Y" if binning == "Monthly" else "%d %b",
    )
    figure.update_yaxes(
        title="Issue voices", rangemode="tozero", dtick=1 if data["Volume"].max() <= 5 else None
    )
    return figure


def issue_channel_breakdown(data: pd.DataFrame) -> go.Figure:
    figure = px.bar(
        data.sort_values("Volume"),
        x="Volume",
        y="source",
        orientation="h",
        title="Source channel breakdown",
        color_discrete_sequence=[TOPIC_COLORS[0]],
    )
    style_figure(figure, 250)
    figure.update_layout(margin=dict(l=16, r=16, t=55, b=40), showlegend=False)
    figure.update_xaxes(title="Issue voices", rangemode="tozero")
    figure.update_yaxes(title=None)
    return figure


def department_overlap(data: pd.DataFrame) -> go.Figure:
    figure = px.bar(
        data,
        x="Percent",
        y="Department",
        orientation="h",
        color="Department",
        color_discrete_map={
            "IT": TOPIC_COLORS[2],
            "Bisnis": TOPIC_COLORS[3],
            "Operasional": TOPIC_COLORS[1],
        },
        title="Department involvement",
        text="Percent",
    )
    style_figure(figure, 230)
    figure.update_layout(margin=dict(l=16, r=30, t=55, b=40), showlegend=False)
    figure.update_traces(
        texttemplate="%{x:.0f}%",
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}% of topic voices<extra></extra>",
    )
    figure.update_xaxes(title="Share of topic voices (%)", range=[0, 110], ticksuffix="%")
    figure.update_yaxes(title=None)
    return figure
