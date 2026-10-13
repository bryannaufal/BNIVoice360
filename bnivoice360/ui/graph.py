"""Deterministic bounded network visualization; selecting a point only opens evidence."""

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bnivoice360.config import SOURCES
from bnivoice360.services.agent.audit import fingerprint
from bnivoice360.ui.charts import TOPIC_COLORS, style_figure

STATUS_COLORS = {"Open": "#B84300", "Escalated": "#246B9C", "Resolved": "#167A45"}
CHANNEL_COLORS = {source: TOPIC_COLORS[i % len(TOPIC_COLORS)] for i, source in enumerate(SOURCES)}


def network_figure(nodes, edges) -> go.Figure:
    """Place components separately so disconnected groups do not look connected."""
    figure = go.Figure()
    positions = {}
    groups = list(nodes.groupby("group_id", sort=True))
    columns = max(1, int(np.ceil(np.sqrt(len(groups)))))
    for component, (_, members) in enumerate(groups):
        cx, cy = (component % columns) * 4.0, -(component // columns) * 4.0
        ordered = sorted(members["suara_nasabah_id"])
        radius = 0 if len(ordered) == 1 else min(1.5, 0.65 + len(ordered) * 0.025)
        for i, ticket in enumerate(ordered):
            angle = 2 * np.pi * i / len(ordered)
            positions[ticket] = (cx + radius * np.cos(angle), cy + radius * np.sin(angle))
    for lower, upper, width in ((0, 0.85, 1), (0.85, 0.95, 2), (0.95, 1.01, 3)):
        x, y = [], []
        for source, target, score in edges.itertuples(index=False, name=None):
            if source in positions and target in positions and lower <= score < upper:
                x.extend([positions[source][0], positions[target][0], None])
                y.extend([positions[source][1], positions[target][1], None])
        if x:
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=width, color="#AFBEC7"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    for source, members in nodes.groupby("source", sort=True):
        x, y, hover, custom = [], [], [], []
        for _, row in members.iterrows():
            ticket = row["suara_nasabah_id"]
            px, py = positions[ticket]
            x.append(px)
            y.append(py)
            posted = row["date_posted"]
            date_label = posted.isoformat() if pd.notna(posted) else "Tanggal kosong"
            hover.append(
                f"<b>{escape(ticket)}</b><br>{escape(str(source))} · {escape(str(row['topic_cluster']))}"
                f"<br>{escape(row['escalation_status'])} · {escape(date_label)}"
                f"<br>{'Urgent' if row['is_Urgent'] == 1 else 'Non-urgent'} · {row['degree']} hubungan"
                f"<br>{escape(str(row['content'])[:180])}"
            )
            custom.append([ticket])
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=str(source),
                mode="markers",
                customdata=custom,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    color=CHANNEL_COLORS.get(
                        source,
                        TOPIC_COLORS[
                            int(fingerprint({"source": source})[:8], 16) % len(TOPIC_COLORS)
                        ],
                    ),
                    size=[18 if urgent == 1 else 12 for urgent in members["is_Urgent"]],
                    line=dict(
                        width=3,
                        color=[STATUS_COLORS[status] for status in members["escalation_status"]],
                    ),
                ),
                selected=dict(marker=dict(opacity=1)),
                unselected=dict(marker=dict(opacity=0.55)),
            )
        )
    style_figure(figure, height=540)
    figure.update_layout(title_text="Jaringan keluhan terkait", dragmode="pan")
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return figure


def selected_ticket(event, valid_ids):
    """Only a node's known ticket ID can open the evidence inspector."""
    points = event.get("selection", {}).get("points", [])
    for point in points:
        data = point.get("customdata", [])
        if isinstance(data, (list, tuple)) and data and data[0] in valid_ids:
            return data[0]
    return None
