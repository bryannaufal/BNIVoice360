"""UI component for Source–Issue–Routing Heterogeneous Graph and Sankey Flow."""

import re
import pandas as pd
import plotly.graph_objects as go

# Canonical 7 prototype categories
PROTOTYPE_TOPIC_ORDER = [
    "Lainnya",
    "Wondr / Aplikasi",
    "Transfer / Pembayaran",
    "Kartu / ATM",
    "Layanan / Cabang",
    "Biaya / Produk",
    "Keamanan / Fraud",
]

SOURCE_DISPLAY_MAP = {
    "Laporan Nasabah": "Internal_CRM",
}


def classify_voice_prototype(content: object, topic_cluster: object) -> str:
    """Rule-based issue grouping for the 7 prototype categories."""
    text = (str(content or "") + " " + str(topic_cluster or "")).lower()
    if re.search(
        r"\b(fraud|phishing|penipuan|scam|hacker|bobol|keamanan|kejahatan|apk jahat|rekening dibobol)\b",
        text,
    ):
        return "Keamanan / Fraud"
    if re.search(
        r"\b(biaya|admin|bunga|tarif|potongan|administrasi|cashback|promo|saldo minimum|biaya bulanan)\b",
        text,
    ):
        return "Biaya / Produk"
    if re.search(
        r"\b(cabang|teller|cs|customer service|kantor|antrian|antre|petugas|satpam|pelayanan kantor)\b",
        text,
    ):
        return "Layanan / Cabang"
    if re.search(
        r"\b(atm|kartu|debit|kredit|mesin atm|tarik tunai|setor tunai|kartu tertelan|kartu rusak|pin atm|blokir kartu)\b",
        text,
    ):
        return "Kartu / ATM"
    if re.search(
        r"\b(transfer|tf|bayar|pembayaran|qris|topup|top up|kirim uang|transaksi|gopay|ovo|dana|shopeepay|va|virtual account)\b",
        text,
    ):
        return "Transfer / Pembayaran"
    if re.search(
        r"\b(wondr|aplikasi|app|bni mobile|mobile banking|mbanking|login|otp|password|fitur|update|versi|error|crash|force close|verifikasi|biometrik)\b",
        text,
    ):
        return "Wondr / Aplikasi"
    return "Lainnya"


def prepare_graph_data(
    frame: pd.DataFrame,
    topic_mode: str = "modeled",
    top_n_topics: int = 10,
    include_other: bool = True,
    include_outliers: bool = False,
    selected_topics: list[str] | None = None,
) -> dict:
    """Process repository frame into structured node and edge metadata."""
    if frame.empty:
        return {
            "total_volume": 0,
            "source_nodes": [],
            "topic_nodes": [],
            "routing_nodes": [],
            "source_topic_edges": [],
            "topic_routing_edges": [],
            "highlights": {
                "total_volume": 0,
                "largest_source": "—",
                "largest_source_count": 0,
                "urgent_count": 0,
                "urgent_share": "0.0%",
                "top_topics": [],
            },
            "processed_frame": pd.DataFrame(),
        }

    df = frame.copy()

    # Determine topic mapping
    if topic_mode == "prototype":
        df["computed_topic"] = df.apply(
            lambda row: classify_voice_prototype(row.get("content"), row.get("topic_cluster")),
            axis=1,
        )
        active_topics = list(PROTOTYPE_TOPIC_ORDER)
    else:
        # Topic modeling clusters from BERTopic
        if not include_outliers:
            df = df.loc[~df["topic_cluster"].isin(["Outlier", "Unlabeled"])]

        topic_counts = df["topic_cluster"].value_counts()
        if selected_topics:
            top_topic_names = [t for t in selected_topics if t in topic_counts]
        else:
            top_topic_names = list(topic_counts.head(top_n_topics).index)

        def map_modeled(val):
            if val in top_topic_names:
                return val
            return "Topik Lainnya" if include_other else None

        df["computed_topic"] = df["topic_cluster"].apply(map_modeled)
        if not include_other:
            df = df.dropna(subset=["computed_topic"])

        active_topics = top_topic_names + (["Topik Lainnya"] if include_other and "Topik Lainnya" in df["computed_topic"].values else [])

    # 1. Source Nodes
    source_counts = df["source"].value_counts()
    source_nodes = []
    for src_raw, count in source_counts.items():
        src_display = SOURCE_DISPLAY_MAP.get(src_raw, src_raw)
        source_nodes.append(
            {
                "id": src_raw,
                "name": src_display,
                "count": int(count),
                "color": "#008080",
                "bg": "#E6F4F4",
                "border": "#008080",
            }
        )

    # 2. Topic Nodes
    topic_volume_map = df["computed_topic"].value_counts().to_dict()
    topic_nodes = []
    for t_name in active_topics:
        count = int(topic_volume_map.get(t_name, 0))
        if count > 0:
            topic_nodes.append(
                {
                    "id": t_name,
                    "name": t_name,
                    "count": count,
                    "color": "#E55300",
                    "bg": "#FFF0E8",
                    "border": "#E55300",
                }
            )

    # 3. Routing Nodes
    it_count = int(df["is_IT"].eq(1).sum()) if "is_IT" in df.columns else 0
    ops_count = int(df["is_Operasional"].eq(1).sum()) if "is_Operasional" in df.columns else 0
    bisnis_count = int(df["is_Bisnis"].eq(1).sum()) if "is_Bisnis" in df.columns else 0
    urgent_count = int(df["is_Urgent"].eq(1).sum()) if "is_Urgent" in df.columns else 0

    routing_nodes = [
        {"id": "IT", "name": "IT", "count": it_count, "color": "#5E4B8B", "bg": "#F1ECF7", "border": "#5E4B8B"},
        {"id": "Operasional", "name": "Operasional", "count": ops_count, "color": "#5E4B8B", "bg": "#F1ECF7", "border": "#5E4B8B"},
        {"id": "Bisnis", "name": "Bisnis", "count": bisnis_count, "color": "#5E4B8B", "bg": "#F1ECF7", "border": "#5E4B8B"},
        {"id": "Urgent", "name": "Urgent", "count": urgent_count, "color": "#C0392B", "bg": "#FDECEB", "border": "#C0392B"},
    ]

    # 4. Edges
    src_topic_grouped = df.groupby(["source", "computed_topic"]).size().reset_index(name="weight")
    src_topic_edges = [
        (r["source"], r["computed_topic"], int(r["weight"]))
        for _, r in src_topic_grouped.iterrows()
    ]

    topic_routing_pairs = []
    for _, row in df.iterrows():
        top = row["computed_topic"]
        if row.get("is_IT") == 1:
            topic_routing_pairs.append((top, "IT"))
        if row.get("is_Operasional") == 1:
            topic_routing_pairs.append((top, "Operasional"))
        if row.get("is_Bisnis") == 1:
            topic_routing_pairs.append((top, "Bisnis"))
        if row.get("is_Urgent") == 1:
            topic_routing_pairs.append((top, "Urgent"))

    if topic_routing_pairs:
        tr_df = (
            pd.DataFrame(topic_routing_pairs, columns=["topic", "routing"])
            .groupby(["topic", "routing"])
            .size()
            .reset_index(name="weight")
        )
        topic_routing_edges = [
            (r["topic"], r["routing"], int(r["weight"]))
            for _, r in tr_df.iterrows()
        ]
    else:
        topic_routing_edges = []

    # 5. Highlights
    largest_src = source_nodes[0]["name"] if source_nodes else "—"
    largest_src_cnt = source_nodes[0]["count"] if source_nodes else 0
    total_vol = len(df)
    urgent_share_str = f"{(urgent_count / total_vol * 100):.1f}%" if total_vol > 0 else "0.0%"
    sorted_topics = sorted([t for t in topic_nodes if t["id"] != "Topik Lainnya"], key=lambda x: x["count"], reverse=True)

    highlights = {
        "total_volume": total_vol,
        "largest_source": largest_src,
        "largest_source_count": largest_src_cnt,
        "urgent_count": urgent_count,
        "urgent_share": urgent_share_str,
        "top_topics": sorted_topics[:3],
    }

    return {
        "total_volume": total_vol,
        "source_nodes": source_nodes,
        "topic_nodes": topic_nodes,
        "routing_nodes": routing_nodes,
        "source_topic_edges": src_topic_edges,
        "topic_routing_edges": topic_routing_edges,
        "highlights": highlights,
        "processed_frame": df,
    }


def create_source_routing_figure(
    graph_data: dict,
    height: int | None = None,
    selected_item: dict | None = None,
) -> go.Figure:
    """Build the 3-layer Source–Issue–Routing Network Graph with interactive selection."""
    source_nodes = graph_data["source_nodes"]
    topic_nodes = graph_data["topic_nodes"]
    routing_nodes = graph_data["routing_nodes"]
    src_topic_edges = graph_data["source_topic_edges"]
    topic_routing_edges = graph_data["topic_routing_edges"]

    if not source_nodes or not topic_nodes:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=400,
            annotations=[
                dict(text="Tidak ada data yang sesuai filter.", showarrow=False, font=dict(size=14, color="#64748B"))
            ],
        )
        return fig

    # Selection state flags
    has_selection = selected_item is not None
    sel_type = selected_item.get("type") if has_selection else None
    sel_layer = selected_item.get("layer") if has_selection else None
    sel_id = selected_item.get("id") if has_selection else None
    sel_src = selected_item.get("source") if has_selection else None
    sel_tgt = selected_item.get("target") if has_selection else None
    sel_edge_type = selected_item.get("edge_type") if has_selection else None

    # Calculate height dynamically based on topic node count to prevent overlapping
    n_topics = len(topic_nodes)
    actual_height = height or max(720, 160 + n_topics * 42)

    x_source = 0.08
    x_topic = 0.50
    x_routing = 0.92

    source_coords = {}
    n_source = max(1, len(source_nodes))
    for i, node in enumerate(source_nodes):
        y = 0.88 - (i / max(1, n_source - 1)) * 0.78 if n_source > 1 else 0.5
        source_coords[node["id"]] = (x_source, y, node)

    topic_coords = {}
    for i, node in enumerate(topic_nodes):
        y = 0.88 - (i / max(1, n_topics - 1)) * 0.78 if n_topics > 1 else 0.5
        topic_coords[node["id"]] = (x_topic, y, node)

    routing_coords = {}
    routing_y_map = {
        "IT": 0.86,
        "Operasional": 0.52,
        "Bisnis": 0.28,
        "Urgent": 0.10,
    }
    for node in routing_nodes:
        y = routing_y_map.get(node["id"], 0.5)
        routing_coords[node["id"]] = (x_routing, y, node)

    figure = go.Figure()

    # 1. Source -> Topic Edges
    max_src_weight = max([w for _, _, w in src_topic_edges] or [1])
    for src_id, topic_id, weight in src_topic_edges:
        if src_id in source_coords and topic_id in topic_coords:
            x0, y0, _ = source_coords[src_id]
            x1, y1, _ = topic_coords[topic_id]
            xm = (x0 + x1) / 2
            ym = (y0 + y1) / 2

            norm = weight / max_src_weight
            base_width = max(1.0, min(10.0, 1.2 + norm * 8.0))
            base_opacity = min(0.35, max(0.12, 0.12 + norm * 0.22))
            src_label = SOURCE_DISPLAY_MAP.get(src_id, src_id)

            if not has_selection:
                line_width = base_width
                line_opacity = base_opacity
                marker_size = [0, 8, 0]
                marker_color = f"rgba(0, 128, 128, {min(0.85, base_opacity + 0.3):.2f})"
            else:
                if sel_type == "node":
                    if sel_layer == "source" and src_id == sel_id:
                        is_active = True
                    elif sel_layer == "topic" and topic_id == sel_id:
                        is_active = True
                    else:
                        is_active = False
                elif sel_type == "edge":
                    is_active = (sel_edge_type == "source_topic" and src_id == sel_src and topic_id == sel_tgt)
                else:
                    is_active = False

                if is_active:
                    line_width = max(3.5, base_width * 1.6)
                    line_opacity = 0.90
                    marker_size = [0, 10, 0]
                    marker_color = "#008080"
                else:
                    line_width = 1.0
                    line_opacity = 0.05
                    marker_size = [0, 0, 0]
                    marker_color = "rgba(0, 128, 128, 0.05)"

            edge_cd = [
                ["edge", "source_topic", src_id, topic_id, weight, src_label],
                ["edge", "source_topic", src_id, topic_id, weight, src_label],
                ["edge", "source_topic", src_id, topic_id, weight, src_label],
            ]

            figure.add_trace(
                go.Scatter(
                    x=[x0, xm, x1],
                    y=[y0, ym, y1],
                    mode="lines+markers",
                    line=dict(color=f"rgba(0, 128, 128, {line_opacity:.2f})", width=line_width),
                    marker=dict(size=marker_size, color=marker_color, symbol="circle"),
                    hoverinfo="text",
                    hovertext=f"<b>Aliran: {src_label} ➔ {topic_id}</b><br>Volume: {weight:,} suara<br><i>Klik garis untuk investigasi aliran ini</i>",
                    customdata=edge_cd,
                    showlegend=False,
                )
            )

    # 2. Topic -> Routing Edges
    max_tr_weight = max([w for _, _, w in topic_routing_edges] or [1])
    for topic_id, route_id, weight in topic_routing_edges:
        if topic_id in topic_coords and route_id in routing_coords:
            x0, y0, _ = topic_coords[topic_id]
            x1, y1, _ = routing_coords[route_id]
            xm = (x0 + x1) / 2
            ym = (y0 + y1) / 2

            norm = weight / max_tr_weight
            base_width = max(1.0, min(12.0, 1.2 + norm * 9.5))
            base_opacity = min(0.35, max(0.12, 0.12 + norm * 0.20))
            color_base = "229, 83, 0" if route_id == "IT" else ("192, 57, 43" if route_id == "Urgent" else "94, 75, 139")

            if not has_selection:
                line_width = base_width
                line_opacity = base_opacity
                marker_size = [0, 8, 0]
                marker_color = f"rgba({color_base}, {min(0.85, base_opacity + 0.3):.2f})"
            else:
                if sel_type == "node":
                    if sel_layer == "topic" and topic_id == sel_id:
                        is_active = True
                    elif sel_layer == "routing" and route_id == sel_id:
                        is_active = True
                    else:
                        is_active = False
                elif sel_type == "edge":
                    is_active = (sel_edge_type == "topic_routing" and topic_id == sel_src and route_id == sel_tgt)
                else:
                    is_active = False

                if is_active:
                    line_width = max(3.5, base_width * 1.6)
                    line_opacity = 0.90
                    marker_size = [0, 10, 0]
                    marker_color = f"rgb({color_base})"
                else:
                    line_width = 1.0
                    line_opacity = 0.05
                    marker_size = [0, 0, 0]
                    marker_color = f"rgba({color_base}, 0.05)"

            edge_cd = [
                ["edge", "topic_routing", topic_id, route_id, weight, route_id],
                ["edge", "topic_routing", topic_id, route_id, weight, route_id],
                ["edge", "topic_routing", topic_id, route_id, weight, route_id],
            ]

            figure.add_trace(
                go.Scatter(
                    x=[x0, xm, x1],
                    y=[y0, ym, y1],
                    mode="lines+markers",
                    line=dict(color=f"rgba({color_base}, {line_opacity:.2f})", width=line_width),
                    marker=dict(size=marker_size, color=marker_color, symbol="circle"),
                    hoverinfo="text",
                    hovertext=f"<b>Aliran: {topic_id} ➔ {route_id}</b><br>Volume: {weight:,} suara<br><i>Klik garis untuk investigasi aliran ini</i>",
                    customdata=edge_cd,
                    showlegend=False,
                )
            )

    annotations = []

    # 3. Draw Nodes (Layer by Layer)
    def render_layer(coords_dict, layer_name, layer_code, default_size=36):
        x_vals = []
        y_vals = []
        marker_texts = []
        hover_texts = []
        bg_colors = []
        border_colors = []
        border_widths = []
        sizes = []
        text_colors = []
        node_cds = []

        for node_id, (x, y, node) in coords_dict.items():
            x_vals.append(x)
            y_vals.append(y)

            is_sel_node = (has_selection and sel_type == "node" and sel_id == node["id"] and sel_layer == layer_code)
            is_edge_end = (has_selection and sel_type == "edge" and (node["id"] == sel_src or node["id"] == sel_tgt))

            # Node size dynamically scaled by volume
            size = default_size
            if layer_name == "Issue / Topic":
                size = max(32, min(58, 30 + int((node["count"] / max(1, topic_nodes[0]["count"])) * 24)))
            elif layer_name == "Routing / Priority":
                size = 56 if node["id"] == "IT" else 46

            if is_sel_node or is_edge_end:
                size += 6
                border_width = 4.0
                border_color = "#0F172A"
                bg_color = node["bg"]
                text_color = "#0F172A"
            else:
                border_width = 2.2
                border_color = node["border"]
                bg_color = node["bg"]
                text_color = node["border"]

            sizes.append(size)
            border_widths.append(border_width)
            border_colors.append(border_color)
            bg_colors.append(bg_color)
            text_colors.append(text_color)
            marker_texts.append(f"<b>{node['count']:,}</b>")
            hover_texts.append(
                f"<b>{node['name']}</b><br>Kategori: {layer_name}<br>Total Suara: {node['count']:,}<br><i>Klik node untuk investigasi mendalam</i>"
            )
            node_cds.append(["node", layer_code, node["id"], node["name"], node["count"]])

            # Node label annotation
            name_text = node["name"]
            display_name = name_text if len(name_text) <= 24 else name_text[:22] + "…"
            label_font_color = "#0F172A" if (is_sel_node or is_edge_end) else "#334155"
            annotations.append(
                dict(
                    x=x,
                    y=y + 0.038,
                    xref="x",
                    yref="y",
                    text=f"<b>{display_name}</b>",
                    showarrow=False,
                    font=dict(size=10, color=label_font_color, family="Inter, Arial"),
                    xanchor="center",
                    yanchor="bottom",
                )
            )

        figure.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=bg_colors,
                    line=dict(color=border_colors, width=border_widths),
                ),
                text=marker_texts,
                textposition="middle center",
                textfont=dict(color=text_colors, family="Inter, Arial", size=10),
                hoverinfo="text",
                hovertext=hover_texts,
                customdata=node_cds,
                showlegend=False,
            )
        )

    render_layer(source_coords, "Source", "source", default_size=36)
    render_layer(topic_coords, "Issue / Topic", "topic", default_size=40)
    render_layer(routing_coords, "Routing / Priority", "routing", default_size=48)

    # Column Headers
    annotations.extend(
        [
            dict(
                x=x_source,
                y=0.98,
                xref="x",
                yref="y",
                text="<b>SOURCE</b>",
                showarrow=False,
                font=dict(size=13, color="#008080", family="Inter, Arial"),
                xanchor="center",
            ),
            dict(
                x=x_topic,
                y=0.98,
                xref="x",
                yref="y",
                text="<b>ISSUE / TOPIC</b>",
                showarrow=False,
                font=dict(size=13, color="#E55300", family="Inter, Arial"),
                xanchor="center",
            ),
            dict(
                x=x_routing,
                y=0.98,
                xref="x",
                yref="y",
                text="<b>ROUTING / PRIORITY</b>",
                showarrow=False,
                font=dict(size=13, color="#5E4B8B", family="Inter, Arial"),
                xanchor="center",
            ),
        ]
    )

    figure.update_layout(
        template="plotly_white",
        height=actual_height,
        autosize=True,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=25, r=25, t=40, b=30),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.02, 1.05]),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#1E293B", bordercolor="#CBD5E1"),
        annotations=annotations,
    )

    return figure


def create_sankey_flow_figure(graph_data: dict, height: int = 680) -> go.Figure:
    """Build an interactive Plotly Sankey flow diagram (Source ➔ Topic ➔ Unit)."""
    source_nodes = graph_data["source_nodes"]
    topic_nodes = graph_data["topic_nodes"]
    routing_nodes = graph_data["routing_nodes"]
    src_topic_edges = graph_data["source_topic_edges"]
    topic_routing_edges = graph_data["topic_routing_edges"]

    if not source_nodes or not topic_nodes:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=height,
            annotations=[
                dict(text="Tidak ada data yang sesuai filter.", showarrow=False, font=dict(size=14, color="#64748B"))
            ],
        )
        return fig

    # Map labels and indices
    labels = []
    colors = []
    idx_map = {}

    for n in source_nodes:
        idx_map[("src", n["id"])] = len(labels)
        labels.append(f"{n['name']} ({n['count']:,})")
        colors.append("#008080")

    for n in topic_nodes:
        idx_map[("top", n["id"])] = len(labels)
        labels.append(f"{n['name']} ({n['count']:,})")
        colors.append("#E55300")

    for n in routing_nodes:
        idx_map[("route", n["id"])] = len(labels)
        labels.append(f"{n['name']} ({n['count']:,})")
        colors.append("#C0392B" if n["id"] == "Urgent" else "#5E4B8B")

    sources = []
    targets = []
    values = []
    link_colors = []

    # Source -> Topic links
    for src_id, topic_id, weight in src_topic_edges:
        src_k = ("src", src_id)
        top_k = ("top", topic_id)
        if src_k in idx_map and top_k in idx_map and weight > 0:
            sources.append(idx_map[src_k])
            targets.append(idx_map[top_k])
            values.append(weight)
            link_colors.append("rgba(0, 128, 128, 0.25)")

    # Topic -> Routing links
    for topic_id, route_id, weight in topic_routing_edges:
        top_k = ("top", topic_id)
        route_k = ("route", route_id)
        if top_k in idx_map and route_k in idx_map and weight > 0:
            sources.append(idx_map[top_k])
            targets.append(idx_map[route_k])
            values.append(weight)
            c = "rgba(192, 57, 43, 0.28)" if route_id == "Urgent" else "rgba(94, 75, 139, 0.25)"
            link_colors.append(c)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=22,
                    line=dict(color="#CBD5E1", width=1),
                    label=labels,
                    color=colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=25, b=25),
        font=dict(family="Inter, Arial", size=11, color="#1E293B"),
    )

    return fig


def build_quadrant_matrix_data(
    frame: pd.DataFrame,
    topic_mode: str = "modeled",
    min_volume: int = 5,
    top_n_topics: int = 20,
    urgent_threshold: float = 0.5,
    growth_threshold: float = 40.0,
) -> pd.DataFrame:
    """Build cluster-level metrics for the Priority/Action Quadrant Matrix from raw VoC data."""
    if frame.empty:
        return pd.DataFrame()

    data = frame.copy()
    if "date_posted" in data.columns:
        data["date"] = pd.to_datetime(data["date_posted"], errors="coerce")
        max_date = data["date"].max()
    else:
        data["date"] = pd.NaT
        max_date = None

    if topic_mode == "prototype":
        data["cluster_label"] = data.apply(
            lambda r: classify_voice_prototype(r.get("content"), r.get("topic_cluster")),
            axis=1,
        )
    else:
        data = data.loc[~data["topic_cluster"].isin(["Outlier", "Unlabeled"])].copy()
        data["cluster_label"] = data["topic_cluster"]

    vol_counts = data["cluster_label"].value_counts()
    eligible = vol_counts[vol_counts >= min_volume].index.tolist()
    if not eligible:
        eligible = vol_counts.head(top_n_topics).index.tolist()

    # Balanced cluster selection:
    # Top volume clusters, top urgent clusters, plus dominant ops and bisnis clusters
    top_by_vol = eligible[:top_n_topics]
    top_by_urg = (
        data.groupby("cluster_label")["is_Urgent"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
        if "is_Urgent" in data.columns
        else []
    )
    ops_col = "is_OPS" if "is_OPS" in data.columns else "is_Operasional"
    top_by_ops = (
        data[data[ops_col] == 1]["cluster_label"].value_counts().head(4).index.tolist()
        if ops_col in data.columns
        else []
    )
    top_by_bisnis = (
        data[data["is_Bisnis"] == 1]["cluster_label"].value_counts().head(4).index.tolist()
        if "is_Bisnis" in data.columns
        else []
    )

    selected_topics = list(dict.fromkeys(top_by_vol + top_by_urg + top_by_ops + top_by_bisnis))
    selected_topics = [t for t in selected_topics if t in eligible]

    records = []
    for t in selected_topics:
        g = data[data["cluster_label"] == t]
        vol = len(g)
        urg_cnt = int(g["is_Urgent"].sum()) if "is_Urgent" in g.columns else 0
        urg_ratio = urg_cnt / vol if vol > 0 else 0.0

        # Calculate momentum / velocity %
        if max_date is not None and not g["date"].isna().all():
            r90 = (g["date"] >= (max_date - pd.Timedelta(days=90))).sum()
            r180 = (g["date"] >= (max_date - pd.Timedelta(days=180))).sum()
            growth = min(100.0, max(0.0, (0.6 * (r90 / vol) + 0.4 * (r180 / vol)) * 100.0))
        else:
            growth = 20.0

        it_cnt = int(g["is_IT"].sum()) if "is_IT" in g.columns else 0
        ops_cnt = int(g[ops_col].sum()) if ops_col in g.columns else 0
        bisnis_cnt = int(g["is_Bisnis"].sum()) if "is_Bisnis" in g.columns else 0

        if ops_cnt > it_cnt and ops_cnt > bisnis_cnt:
            dom = "dominan Ops"
            color = "#D97706"
        elif bisnis_cnt > it_cnt and bisnis_cnt > ops_cnt:
            dom = "dominan Bisnis"
            color = "#10B981"
        else:
            dom = "dominan IT"
            color = "#3B82F6"

        platforms = int(g["source"].nunique()) if "source" in g.columns else 1
        platforms_list = list(g["source"].unique()) if "source" in g.columns else []

        # Determine Quadrant
        is_high_urg = urg_ratio >= urgent_threshold
        is_high_growth = growth >= growth_threshold

        if is_high_growth and is_high_urg:
            quad_key = "escalate_now"
            quad_name = "🚨 Eskalasi Sekarang"
        elif not is_high_growth and is_high_urg:
            quad_key = "urgent_scheduled"
            quad_name = "⏱ Urgent, Terjadwal"
        elif is_high_growth and not is_high_urg:
            quad_key = "fast_growth"
            quad_name = "📈 Tumbuh Cepat"
        else:
            quad_key = "routine_queue"
            quad_name = "📋 Antrian Rutin"

        # Truncate text nicely for on-bubble labels
        display_label = t if len(t) <= 24 else t[:22] + "…"

        records.append({
            "topic": t,
            "display_label": display_label,
            "vol": vol,
            "urg_cnt": urg_cnt,
            "urg_ratio": round(urg_ratio, 3),
            "growth": round(growth, 1),
            "dom": dom,
            "color": color,
            "platforms": platforms,
            "platforms_str": ", ".join(platforms_list),
            "quad_key": quad_key,
            "quad_name": quad_name,
        })

    return pd.DataFrame(records)


def create_quadrant_action_matrix(
    quadrant_df: pd.DataFrame,
    height: int = 500,
    show_labels: str = "priority",
    selected_quadrant: str = "Semua Kuadran",
    selected_unit: str = "Semua Unit",
    urgent_threshold: float = 0.5,
    growth_threshold: float = 40.0,
) -> go.Figure:
    """Build an interactive Plotly bubble chart reproducing Image 2 Action Matrix."""
    fig = go.Figure()

    if quadrant_df.empty:
        fig.update_layout(
            template="plotly_white",
            height=height,
            annotations=[
                dict(
                    text="Tidak ada klaster yang memenuhi kriteria filter saat ini.",
                    showarrow=False,
                    font=dict(size=14, color="#64748B"),
                )
            ],
        )
        return fig

    # 1. Shaded area for Top-Right Quadrant (Eskalasi Sekarang)
    fig.add_shape(
        type="rect",
        x0=growth_threshold,
        x1=102,
        y0=urgent_threshold,
        y1=1.02,
        fillcolor="rgba(255, 235, 230, 0.55)",
        line=dict(width=0),
        layer="below",
    )

    # 2. Dashed threshold lines
    fig.add_shape(
        type="line",
        x0=growth_threshold,
        x1=growth_threshold,
        y0=0,
        y1=1.02,
        line=dict(color="#94A3B8", width=1.5, dash="dash"),
        layer="below",
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=102,
        y0=urgent_threshold,
        y1=urgent_threshold,
        line=dict(color="#94A3B8", width=1.5, dash="dash"),
        layer="below",
    )

    # 3. Quadrant Annotations matching Image 2
    annotations = [
        dict(
            x=85,
            y=0.92,
            xref="x",
            yref="y",
            text="<b>ESKALASI SEKARANG</b>",
            showarrow=False,
            font=dict(size=14, color="#DC2626", family="Inter, Arial, sans-serif"),
            xanchor="center",
        ),
        dict(
            x=18,
            y=0.92,
            xref="x",
            yref="y",
            text="<b>URGENT, STABIL → tangani terjadwal</b>",
            showarrow=False,
            font=dict(size=11, color="#64748B", family="Inter, Arial, sans-serif"),
            xanchor="center",
        ),
        dict(
            x=14,
            y=0.035,
            xref="x",
            yref="y",
            text="<b>RUTIN → antrian biasa</b>",
            showarrow=False,
            font=dict(size=11, color="#64748B", family="Inter, Arial, sans-serif"),
            xanchor="center",
        ),
        dict(
            x=82,
            y=0.035,
            xref="x",
            yref="y",
            text="<b>TUMBUH CEPAT → pantau ketat</b>",
            showarrow=False,
            font=dict(size=11, color="#64748B", family="Inter, Arial, sans-serif"),
            xanchor="center",
        ),
    ]

    # Filter data if unit or quadrant filter active
    plot_df = quadrant_df.copy()
    if selected_quadrant and selected_quadrant != "Semua Kuadran":
        kw = selected_quadrant.split()[-1]
        plot_df = plot_df.loc[plot_df["quad_name"].str.contains(kw, case=False, na=False)]
    if selected_unit and selected_unit != "Semua Unit":
        plot_df = plot_df.loc[plot_df["dom"] == selected_unit]

    # Group by unit for trace styling
    unit_configs = [
        ("dominan IT", "#3B82F6"),
        ("dominan Ops", "#D97706"),
        ("dominan Bisnis", "#10B981"),
    ]

    for unit_name, unit_color in unit_configs:
        unit_sub = plot_df.loc[plot_df["dom"] == unit_name]
        if unit_sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=unit_name,
                    marker=dict(size=12, color=unit_color),
                    showlegend=True,
                )
            )
            continue

        # Compute marker sizes proportional to platforms count and volume
        marker_sizes = [
            max(20, min(48, int(18 + row["platforms"] * 4.0 + min(10, row["vol"] * 0.05))))
            for _, row in unit_sub.iterrows()
        ]

        # Determine text labels display
        labels = []
        for _, row in unit_sub.iterrows():
            if show_labels == "none":
                labels.append("")
            elif show_labels == "all":
                labels.append(row["display_label"])
            else:
                # "priority" mode: label clusters with high urgency, fast growth, or top volume
                if (
                    row["urg_ratio"] >= 0.20
                    or row["growth"] >= 40.0
                    or row["vol"] >= 30
                    or row["platforms"] >= 4
                ):
                    labels.append(row["display_label"])
                else:
                    labels.append("")

        custom_data = [
            (
                row["topic"],
                row["dom"],
                row["vol"],
                row["urg_cnt"],
                row["urg_ratio"],
                row["growth"],
                row["platforms"],
                row["platforms_str"],
            )
            for _, row in unit_sub.iterrows()
        ]

        hover_template = (
            "<b>%{customdata[0]}</b><br>"
            "Unit Terkait: <b>%{customdata[1]}</b><br>"
            "Volume Suara: <b>%{customdata[2]:,} suara</b><br>"
            "Keluhan Urgent: <b>%{customdata[3]:,}</b> (%{customdata[4]:.1%})<br>"
            "Pertumbuhan: <b>%{customdata[5]:.1f}% per hari</b><br>"
            "Jangkauan Platform: <b>%{customdata[6]} kanal</b> (%{customdata[7]})"
            "<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=unit_sub["growth"].tolist(),
                y=unit_sub["urg_ratio"].tolist(),
                mode="markers+text" if show_labels != "none" else "markers",
                name=unit_name,
                text=labels,
                textposition="top center",
                textfont=dict(family="Inter, Arial, sans-serif", size=10, color="#1E293B"),
                marker=dict(
                    size=marker_sizes,
                    color=unit_color,
                    opacity=0.88,
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                customdata=custom_data,
                hovertemplate=hover_template,
                showlegend=True,
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=45, r=30, t=30, b=135),
        xaxis=dict(
            title=dict(
                text="<b>Pertumbuhan volume (% per hari)</b>",
                font=dict(size=12, color="#1E293B"),
                standoff=12,
            ),
            range=[-2, 102],
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0", "20", "40", "60", "80", "100"],
            showline=True,
            linecolor="#1E293B",
            linewidth=1.5,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="<b>Rasio is_Urgent dalam klaster</b>",
                font=dict(size=12, color="#1E293B"),
            ),
            range=[-0.03, 1.02],
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
            showline=True,
            linecolor="#1E293B",
            linewidth=1.5,
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.26,
            xanchor="center",
            x=0.5,
            title=dict(
                text="warna = flag dominan · ukuran bubble = jangkauan (degree / jumlah platform)<br>",
                side="top",
                font=dict(family="Inter, Arial, sans-serif", size=11, color="#64748B"),
            ),
            font=dict(family="Inter, Arial, sans-serif", size=11, color="#1E293B"),
        ),
        annotations=annotations,
    )

    return fig
