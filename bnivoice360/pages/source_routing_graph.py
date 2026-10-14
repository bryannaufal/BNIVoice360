"""Source–Issue–Routing Graph Analytics page, dynamically powered by VoC repository data."""

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
        fig_network = create_source_routing_figure(graph_data)
        st.plotly_chart(
            fig_network,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
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
    if topic_names:
        selected_node = st.selectbox(
            "Pilih Klaster Topik untuk Investigasi Mendalam:",
            topic_names,
            key="sir_selected_node",
        )

        proc_df = graph_data["processed_frame"]
        node_voices = proc_df.loc[proc_df["computed_topic"].eq(selected_node)]

        # 5 KPIs ordered: Volume Suara Topik, Keterlibatan IT, Keterlibatan Operasional, Keterlibatan Bisnis, Keluhan Urgent
        d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns(5)
        d_col1.metric("Volume Suara Topik", f"{len(node_voices):,}")

        it_pct = (node_voices["is_IT"].eq(1).mean() * 100) if "is_IT" in node_voices.columns and len(node_voices) else 0.0
        d_col2.metric("Keterlibatan IT", f"{it_pct:.1f}%")

        ops_col = "is_OPS" if "is_OPS" in node_voices.columns else "is_Operasional"
        ops_pct = (node_voices[ops_col].eq(1).mean() * 100) if ops_col in node_voices.columns and len(node_voices) else 0.0
        d_col3.metric("Keterlibatan Operasional", f"{ops_pct:.1f}%")

        bisnis_pct = (node_voices["is_Bisnis"].eq(1).mean() * 100) if "is_Bisnis" in node_voices.columns and len(node_voices) else 0.0
        d_col4.metric("Keterlibatan Bisnis", f"{bisnis_pct:.1f}%")

        urg_cnt = int(node_voices["is_Urgent"].eq(1).sum()) if "is_Urgent" in node_voices.columns else 0
        d_col5.metric("Keluhan Urgent", urg_cnt)

        with st.expander(f"Daftar Suara Nasabah Terkait '{selected_node}' ({len(node_voices)} suara)", expanded=True):
            if len(node_voices) == 0:
                st.info(f"Tidak ada suara nasabah yang terkait dengan klaster '{selected_node}'.")
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
