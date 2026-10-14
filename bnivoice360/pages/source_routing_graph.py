"""Source–Issue–Routing Graph Analytics page, dynamically powered by VoC repository data."""

import json

import pandas as pd
import streamlit as st

from bnivoice360.container import AppServices
from bnivoice360.ui.components import header
from bnivoice360.ui.source_routing_graph import (
    build_quadrant_matrix_data,
    create_quadrant_action_matrix,
    create_sankey_flow_figure,
    create_source_routing_figure,
    prepare_graph_data,
)


def render(services: AppServices) -> None:
    # 1. Header component
    header(
        "Graph Analytics",
        "Analisis jaringan heterogen interaktif: menelusuri aliran suara nasabah dari kanal sumber, klaster topik hasil pemodelan, hingga unit eskalasi dan urgensi.",
        "BNIVoice360 · Graph Analytics",
    )

    # 2. Load identical data source as the Home page
    frame = services.repository.load()
    if frame.empty:
        st.info("Belum ada data VoC. Data Admin dapat mengimpor data di Data Ingestion.")
        return

    valid_dates = pd.to_datetime(frame["date_posted"], errors="coerce").dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
    else:
        min_date = None
        max_date = None

    topic_mode = "modeled"

    # 3. Interactive Controls Panel
    with st.expander("Filter & Pengaturan Pemetaan Jaringan", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            if min_date and max_date:
                selected_dates = st.date_input(
                    "Rentang Tanggal (Date Range)",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="sir_date_range",
                    help="Filter suara nasabah berdasarkan rentang tanggal publikasi.",
                )
            else:
                selected_dates = None

            channels = sorted(frame["source"].unique())
            selected_channels = st.multiselect(
                "Kanal Sumber (Source)",
                channels,
                default=channels,
                key="sir_channels",
            )

        with f_col2:
            sentiments = ["Negative", "Neutral", "Positive"]
            selected_sentiments = st.multiselect(
                "Sentimen",
                sentiments,
                default=sentiments,
                key="sir_sentiments",
            )
            statuses = ["Open", "Escalated", "Resolved"]
            selected_statuses = st.multiselect(
                "Status Eskalasi",
                statuses,
                default=statuses,
                key="sir_statuses",
            )
            urgent_only = st.checkbox("Hanya Suara Urgent", key="sir_urgent")

        with f_col3:
            top_n = st.slider(
                "Jumlah Topik Teratas Ditampilkan (Top N)",
                min_value=5,
                max_value=30,
                value=10,
                step=1,
                key="sir_top_n",
                help="Mengatur berapa banyak klaster hasil topic modeling yang ditampilkan pada kolom tengah.",
            )
            include_other = st.checkbox(
                "Gabungkan sisa topik di luar Top N ke 'Topik Lainnya'",
                value=True,
                key="sir_include_other",
            )
            include_outliers = st.checkbox(
                "Sertakan Outlier & Unlabeled",
                value=False,
                key="sir_include_outliers",
            )

    # 4. Filter the data frame
    filtered = frame.copy()
    if selected_dates:
        f_dt = pd.to_datetime(filtered["date_posted"], errors="coerce").dt.date
        if isinstance(selected_dates, (tuple, list)):
            if len(selected_dates) == 2:
                s_d, e_d = selected_dates
                if min_date and max_date and (s_d > min_date or e_d < max_date):
                    filtered = filtered.loc[f_dt.notna() & (f_dt >= s_d) & (f_dt <= e_d)]
            elif len(selected_dates) == 1:
                s_d = selected_dates[0]
                filtered = filtered.loc[f_dt.notna() & (f_dt >= s_d)]
        elif hasattr(selected_dates, "year"):
            filtered = filtered.loc[f_dt.notna() & (f_dt == selected_dates)]

    if selected_channels:
        filtered = filtered.loc[filtered["source"].isin(selected_channels)]
    if selected_sentiments:
        filtered = filtered.loc[filtered["sentiment"].isin(selected_sentiments)]
    if selected_statuses:
        filtered = filtered.loc[filtered["escalation_status"].isin(selected_statuses)]
    if urgent_only:
        filtered = filtered.loc[filtered["is_Urgent"].eq(1)]

    if filtered.empty:
        st.warning("Tidak ada suara nasabah yang sesuai dengan kriteria filter di atas. Coba longgarkan filter.")
        return

    # 5. Build Graph Data
    graph_data = prepare_graph_data(
        filtered,
        topic_mode=topic_mode,
        top_n_topics=top_n,
        include_other=include_other,
        include_outliers=include_outliers,
    )
    highlights = graph_data["highlights"]

    # 6. Interactive KPI Overview Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Suara dalam Graf", f"{len(filtered):,}", f"dari {len(frame):,} total")
    modeled_topic_count = len([t for t in graph_data["topic_nodes"] if t["id"] != "Topik Lainnya"])
    kpi2.metric(
        "Topik Terpetakan",
        modeled_topic_count,
        "Klaster Topic Modeling",
    )
    kpi3.metric("Kanal Terbesar", highlights["largest_source"], f"{highlights['largest_source_count']:,} suara")
    kpi4.metric(
        "Suara Urgent",
        highlights["urgent_count"],
        f"{highlights['urgent_share']} porsi",
        delta_color="inverse",
    )

    # 7. Main Visualizations (Tabs: Network Graph, Sankey Flow, Priority Action Matrix)
    st.markdown("---")
    tab_network, tab_sankey, tab_matrix = st.tabs(
        [
            "🌐 Jaringan Heterogen (Network Graph)",
            "🌊 Diagram Alir (Sankey Flow)",
            "🎯 Matriks Prioritas & Eskalasi (Action Matrix)",
        ]
    )

    with tab_network:
        active_sel = st.session_state.get("sir_active_selection")
        fig_network = create_source_routing_figure(
            graph_data,
            selected_item=active_sel,
        )
        chart_event = st.plotly_chart(
            fig_network,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            key="sir_hetero_graph_chart",
            on_select="rerun",
            selection_mode="points",
        )

        # Process node / edge click event
        if chart_event:
            points = chart_event.get("selection", {}).get("points", [])
            sig = json.dumps(points, sort_keys=True, default=str)
            if sig != st.session_state.get("sir_last_point_sig"):
                st.session_state["sir_last_point_sig"] = sig
                if points:
                    cd = points[0].get("customdata", [])
                    if isinstance(cd, (list, tuple)) and len(cd) >= 4:
                        if cd[0] == "node":
                            new_sel = {
                                "type": "node",
                                "layer": cd[1],
                                "id": cd[2],
                                "name": cd[3],
                                "count": cd[4],
                            }
                            st.session_state["sir_active_selection"] = new_sel
                            if cd[1] == "topic":
                                st.session_state["sir_selected_node"] = cd[2]
                            st.rerun()
                        elif cd[0] == "edge":
                            new_sel = {
                                "type": "edge",
                                "edge_type": cd[1],
                                "source": cd[2],
                                "target": cd[3],
                                "weight": cd[4],
                                "label": cd[5] if len(cd) > 5 else cd[2],
                            }
                            st.session_state["sir_active_selection"] = new_sel
                            st.rerun()

        st.markdown(
            ":blue[● **Source / Kanal**] &nbsp;&nbsp;&nbsp;&nbsp; "
            ":orange[● **Issue / Topik**] &nbsp;&nbsp;&nbsp;&nbsp; "
            ":violet[● **Unit / Routing**] &nbsp;&nbsp;&nbsp;&nbsp; "
            ":red[● **Urgent Priority**]"
        )

    with tab_sankey:
        fig_sankey = create_sankey_flow_figure(graph_data, height=660)
        st.plotly_chart(
            fig_sankey,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
        st.markdown(
            ":blue[● **Source / Kanal**] &nbsp;&nbsp;&nbsp;&nbsp; "
            ":orange[● **Issue / Topik**] &nbsp;&nbsp;&nbsp;&nbsp; "
            ":violet[● **Unit / Routing**] &nbsp;&nbsp;&nbsp;&nbsp; "
            ":red[● **Urgent Priority**]"
        )

    with tab_matrix:
        # Controls toolbar for Matrix
        matrix_ctrl1, matrix_ctrl2, matrix_ctrl3 = st.columns([1.5, 1.5, 1.2])
        with matrix_ctrl1:
            selected_quadrant = st.selectbox(
                "Filter Kuadran Prioritas",
                [
                    "Semua Kuadran",
                    "🚨 Eskalasi Sekarang",
                    "⏱ Urgent, Terjadwal",
                    "📈 Tumbuh Cepat",
                    "📋 Antrian Rutin",
                ],
                key="sir_filter_quadrant",
            )
        with matrix_ctrl2:
            selected_unit = st.selectbox(
                "Filter Unit Dominan",
                ["Semua Unit", "dominan IT", "dominan Ops", "dominan Bisnis"],
                key="sir_filter_unit",
            )
        with matrix_ctrl3:
            label_mode = st.selectbox(
                "Tampilan Label Klaster",
                ["Klaster Prioritas", "Semua Klaster", "Hanya Saat Hover"],
                key="sir_label_mode",
            )
            show_labels_val = (
                "priority"
                if label_mode == "Klaster Prioritas"
                else ("all" if label_mode == "Semua Klaster" else "none")
            )

        quadrant_df = build_quadrant_matrix_data(
            filtered,
            topic_mode="modeled",
            min_volume=5,
            top_n_topics=top_n,
        )

        if not quadrant_df.empty:
            q_esc = len(quadrant_df.loc[quadrant_df["quad_key"] == "escalate_now"])
            q_urg = len(quadrant_df.loc[quadrant_df["quad_key"] == "urgent_scheduled"])
            q_gro = len(quadrant_df.loc[quadrant_df["quad_key"] == "fast_growth"])
            q_rou = len(quadrant_df.loc[quadrant_df["quad_key"] == "routine_queue"])

            qc1, qc2, qc3, qc4 = st.columns(4)
            qc1.metric("🚨 Eskalasi Sekarang", f"{q_esc} Klaster", "Urgensi & Pertumbuhan Tinggi", delta_color="inverse")
            qc2.metric("⏱ Urgent, Terjadwal", f"{q_urg} Klaster", "Urgensi Tinggi, Stabil")
            qc3.metric("📈 Tumbuh Cepat", f"{q_gro} Klaster", "Pantau Laju Lonjakan")
            qc4.metric("📋 Antrian Rutin", f"{q_rou} Klaster", "Penanganan Reguler")

            fig_matrix = create_quadrant_action_matrix(
                quadrant_df,
                height=520,
                show_labels=show_labels_val,
                selected_quadrant=selected_quadrant,
                selected_unit=selected_unit,
            )
            st.plotly_chart(
                fig_matrix,
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )
        else:
            st.info("Tidak ada klaster data yang memenuhi kriteria filter matriks prioritas.")

    # 8. Interactive Node Deep-Dive & Voice Feed Explorer
    st.markdown("---")
    st.subheader("🔍 Investigasi Mendalam Node & Suara Nasabah Asli")

    topic_names = [t["name"] for t in graph_data["topic_nodes"]]
    proc_df = graph_data["processed_frame"]
    active_sel = st.session_state.get("sir_active_selection")

    if active_sel:
        inf_col, btn_col = st.columns([4, 1])
        with inf_col:
            if active_sel["type"] == "node":
                layer_lbl = {
                    "source": "Kanal Sumber",
                    "topic": "Klaster Topik",
                    "routing": "Unit / Routing",
                }.get(active_sel.get("layer"), "Node")
                node_name = active_sel.get("name", active_sel.get("id"))
                st.info(
                    f"🎯 **Node Terpilih dari Graf:** {layer_lbl} **'{node_name}'** ({active_sel.get('count', 0):,} suara) — "
                    "KPI dan daftar suara di bawah telah disesuaikan."
                )
            elif active_sel["type"] == "edge":
                st.info(
                    f"🌊 **Garis Aliran Terpilih dari Graf:** **'{active_sel.get('source')}' ➔ '{active_sel.get('target')}'** "
                    f"({active_sel.get('weight', 0):,} suara) — "
                    "KPI dan daftar suara di bawah telah disesuaikan."
                )
        with btn_col:
            if st.button("✖ Reset Pilihan", key="sir_btn_reset_sel"):
                st.session_state["sir_active_selection"] = None
                st.session_state["sir_last_point_sig"] = ""
                st.rerun()

        # Compute filtered node_voices based on active_sel
        if active_sel["type"] == "node":
            layer = active_sel.get("layer")
            node_id = active_sel.get("id")
            if layer == "source":
                scope_label = f"Kanal '{active_sel.get('name', node_id)}'"
                node_voices = proc_df.loc[proc_df["source"].eq(node_id)]
            elif layer == "routing":
                scope_label = f"Unit / Routing '{node_id}'"
                if node_id == "IT":
                    node_voices = proc_df.loc[proc_df["is_IT"].eq(1)]
                elif node_id in ("Operasional", "Ops"):
                    ops_col_name = "is_OPS" if "is_OPS" in proc_df.columns else "is_Operasional"
                    node_voices = proc_df.loc[proc_df[ops_col_name].eq(1)]
                elif node_id == "Bisnis":
                    node_voices = proc_df.loc[proc_df["is_Bisnis"].eq(1)]
                elif node_id == "Urgent":
                    node_voices = proc_df.loc[proc_df["is_Urgent"].eq(1)]
                else:
                    node_voices = proc_df
            else:  # topic
                scope_label = f"Klaster Topik '{node_id}'"
                node_voices = proc_df.loc[proc_df["computed_topic"].eq(node_id)]
        elif active_sel["type"] == "edge":
            src = active_sel.get("source")
            tgt = active_sel.get("target")
            edge_type = active_sel.get("edge_type")
            if edge_type == "source_topic":
                scope_label = f"Aliran '{src}' ➔ '{tgt}'"
                node_voices = proc_df.loc[proc_df["source"].eq(src) & proc_df["computed_topic"].eq(tgt)]
            else:  # topic_routing
                scope_label = f"Aliran '{src}' ➔ '{tgt}'"
                topic_match = proc_df["computed_topic"].eq(src)
                if tgt == "IT":
                    route_match = proc_df["is_IT"].eq(1)
                elif tgt in ("Operasional", "Ops"):
                    ops_col_name = "is_OPS" if "is_OPS" in proc_df.columns else "is_Operasional"
                    route_match = proc_df[ops_col_name].eq(1)
                elif tgt == "Bisnis":
                    route_match = proc_df["is_Bisnis"].eq(1)
                elif tgt == "Urgent":
                    route_match = proc_df["is_Urgent"].eq(1)
                else:
                    route_match = pd.Series(True, index=proc_df.index)
                node_voices = proc_df.loc[topic_match & route_match]
        else:
            scope_label = "Semua Suara"
            node_voices = proc_df
    else:
        # Default topic selection
        if topic_names:
            selected_node = st.selectbox(
                "Pilih Klaster Topik untuk Investigasi Mendalam:",
                topic_names,
                key="sir_selected_node",
            )
            scope_label = f"Klaster '{selected_node}'"
            node_voices = proc_df.loc[proc_df["computed_topic"].eq(selected_node)]
        else:
            selected_node = ""
            scope_label = "Klaster"
            node_voices = pd.DataFrame()

    # 5 KPIs ordered: Volume Suara Topik, Keterlibatan IT, Keterlibatan Operasional, Keterlibatan Bisnis, Keluhan Urgent
    d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns(5)
    d_col1.metric("Volume Suara Terpilih", f"{len(node_voices):,}")

    it_pct = (node_voices["is_IT"].eq(1).mean() * 100) if "is_IT" in node_voices.columns and len(node_voices) else 0.0
    d_col2.metric("Keterlibatan IT", f"{it_pct:.1f}%")

    ops_col = "is_OPS" if "is_OPS" in node_voices.columns else "is_Operasional"
    ops_pct = (node_voices[ops_col].eq(1).mean() * 100) if ops_col in node_voices.columns and len(node_voices) else 0.0
    d_col3.metric("Keterlibatan Operasional", f"{ops_pct:.1f}%")

    bisnis_pct = (node_voices["is_Bisnis"].eq(1).mean() * 100) if "is_Bisnis" in node_voices.columns and len(node_voices) else 0.0
    d_col4.metric("Keterlibatan Bisnis", f"{bisnis_pct:.1f}%")

    urg_cnt = int(node_voices["is_Urgent"].eq(1).sum()) if "is_Urgent" in node_voices.columns else 0
    d_col5.metric("Keluhan Urgent", urg_cnt)

    with st.expander(f"Daftar Suara Nasabah Terkait {scope_label} ({len(node_voices)} suara)", expanded=True):
        if len(node_voices) == 0:
            st.info(f"Tidak ada suara nasabah yang terkait dengan {scope_label}.")
        else:
            with st.container(height=450):
                for idx, (_, v_row) in enumerate(node_voices.iterrows(), start=1):
                    with st.container(border=True):
                        h_left, h_right = st.columns([4, 1])
                        with h_left:
                            meta_items = [
                                f"**Suara #{idx}**",
                                f"Kanal: **{v_row.get('source', '-')}**",
                                f"`{v_row.get('escalation_status', 'Open')}`",
                            ]
                            sentiment = v_row.get("sentiment")
                            if sentiment:
                                sent_color = (
                                    ":red"
                                    if sentiment == "Negative"
                                    else (":green" if sentiment == "Positive" else ":orange")
                                )
                                meta_items.append(f"Sentimen: {sent_color}[**{sentiment}**]")
                            if v_row.get("is_Urgent") == 1:
                                meta_items.append("🚨 **URGENT**")
                            st.markdown(" · ".join(meta_items))
                        with h_right:
                            date_str = str(v_row.get("date_posted", ""))[:10]
                            if date_str:
                                st.caption(f"📅 **{date_str}**")
                        content_text = str(v_row.get("content", ""))
                        st.write(f'"{content_text}"')

    # 9. Detailed Data Matrix Expander
    with st.expander("Detail Matriks Suara Nasabah (Kanal ➔ Topik ➔ Unit Terkait)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Distribusi Suara per Kanal (Source)**")
            src_df = pd.DataFrame(
                [{"Kanal": n["name"], "Volume Suara": n["count"]} for n in graph_data["source_nodes"]]
            )
            st.dataframe(src_df, width="stretch", hide_index=True)
        with c2:
            st.markdown("**Distribusi Suara per Unit Terkait & Urgensi**")
            routing_data = pd.DataFrame(
                [{"Unit / Kategori": n["name"], "Volume Suara": n["count"]} for n in graph_data["routing_nodes"]]
            )
            st.dataframe(routing_data, width="stretch", hide_index=True)
