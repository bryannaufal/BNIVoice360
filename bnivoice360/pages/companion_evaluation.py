"""Data Admin workspace for operational telemetry, reproducible evaluation and human review."""

import hashlib
import json
import sqlite3
from datetime import timedelta

import pandas as pd
import streamlit as st

from bnivoice360.container import AppServices
from bnivoice360.domain import Role
from bnivoice360.services.companion_evaluation import MODES
from bnivoice360.services.companion_workflow import CompanionConversation
from bnivoice360.ui.components import header
from bnivoice360.ui.credentials import groq_key


def local_time(value):
    return (
        pd.to_datetime(value, utc=True).tz_convert("Asia/Jakarta").strftime("%Y-%m-%d %H:%M:%S")
        if value
        else "—"
    )


def csv_bytes(frame):
    # Reviewer notes and legacy metadata may begin with spreadsheet formula characters.
    safe = frame.apply(
        lambda column: column.map(
            lambda value: (
                "'" + value
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r"))
                else value
            )
        )
    )
    return safe.to_csv(index=False).encode("utf-8-sig")


def event_frame(rows):
    return pd.DataFrame(
        [
            {
                "Event ID": row["event_id"],
                "Waktu (Jakarta)": local_time(row["timestamp"]),
                "Event": row["event"],
                "Status": row.get("status") or row["metadata"].get("status", "event"),
                "Sesi": row["session_id"],
                "Interaksi": row["metadata"].get("interaction_id", "—"),
                "Model": row["metadata"].get("model", "—"),
                "Tool": row["metadata"].get("tool", "—"),
                "Durasi (ms)": row["metadata"].get("duration_ms"),
                "Token": row["metadata"].get("tokens"),
                "Alasan": row["metadata"].get("reason") or row["metadata"].get("error_code", "—"),
            }
            for row in rows
        ]
    )


def overview(service, role, start, end):
    metrics = service.store.overview(role, start, end)
    columns = st.columns(4)
    columns[0].metric("Interaksi tercatat", f"{metrics['interactions']:,}")
    columns[1].metric(
        "Interaksi berstatus OK",
        f"{metrics['ok_percent']:.1f}%" if metrics["ok_percent"] is not None else "—",
    )
    columns[2].metric("Event fallback", f"{metrics['fallback_events']:,}")
    columns[3].metric(
        "Respons p95", f"{metrics['p95_ms'] / 1000:.2f} s" if metrics["p95_ms"] is not None else "—"
    )
    columns = st.columns(4)
    columns[0].metric(
        "Token model tercatat", f"{metrics['tokens']:,}" if metrics["tokens"] is not None else "—"
    )
    columns[1].metric(
        "Rata-rata panggilan model",
        f"{metrics['model_mean_ms'] / 1000:.2f} s" if metrics["model_mean_ms"] is not None else "—",
    )
    columns[2].metric(
        "Rata-rata tool analisis",
        f"{metrics['tool_mean_ms']:.0f} ms" if metrics["tool_mean_ms"] is not None else "—",
    )
    columns[3].metric("Event validasi ditolak", f"{metrics['blocked_events']:,}")
    st.caption(
        "Metrik ini mengukur proses teknis, bukan ketepatan jawaban AI. “—” berarti belum tersedia. Token hanya dijumlahkan dari event panggilan model; total analisis tidak dihitung ulang."
    )
    if not metrics["events"]:
        st.info("Belum ada log pada periode ini. Gunakan Companion atau ubah periode log.")
    elif not metrics["interactions"]:
        st.info(
            "Log historis tersedia, tetapi durasi per interaksi baru tercatat setelah pembaruan ini."
        )
    counts = metrics["event_counts"]
    if counts:
        left, right = st.columns(2)
        data = pd.DataFrame(counts)
        left.markdown("**Aktivitas Companion**")
        left.bar_chart(data.groupby("event")["n"].sum().sort_values(ascending=False).head(10))
        right.markdown("**Status event**")
        right.bar_chart(data.groupby("status")["n"].sum())
    runs = service.store.runs(role, 5)
    if runs:
        st.markdown("**Evaluasi terbaru**")
        st.dataframe(run_frame(runs), hide_index=True, width="stretch")


def logs(service, role, start, end):
    choices = service.store.log_options(role, start, end)
    a, b, c = st.columns(3)
    event = a.selectbox("Jenis event", ["Semua", *choices["events"]], key="eval_log_event")
    status = b.selectbox(
        "Status event",
        ["Semua", "ok", "failed", "fallback", "blocked", "refused", "clarification", "event"],
        key="eval_log_status",
    )
    session = c.text_input(
        "Session ID (opsional)", key="eval_log_session", placeholder="Filter ID sesi persis"
    )
    signature = repr((start, end, event, status, session))
    if st.session_state.get("eval_log_filter") != signature:
        st.session_state.eval_log_filter, st.session_state.eval_log_page = signature, 0
    page = st.session_state.get("eval_log_page", 0)
    result = service.store.logs(
        role,
        start,
        end,
        event=None if event == "Semua" else event,
        status=None if status == "Semua" else status,
        session=session.strip() or None,
        offset=page * 100,
        limit=100,
    )
    if not result["rows"]:
        st.info("Tidak ada event yang sesuai filter.")
        return
    frame = event_frame(result["rows"])
    st.caption(f"{result['total']:,} event cocok · 100 event per halaman · waktu Asia/Jakarta")
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "Unduh halaman log (.csv)",
        csv_bytes(frame),
        "companion-log.csv",
        "text/csv",
        key="eval_log_download",
    )
    previous, summary, next_page = st.columns([1, 2, 1])
    if previous.button("Sebelumnya", key="eval_log_prev", disabled=page == 0):
        st.session_state.eval_log_page = page - 1
        st.rerun()
    summary.caption(f"Halaman {page + 1} dari {max(1, (result['total'] + 99) // 100)}")
    if next_page.button(
        "Berikutnya", key="eval_log_next", disabled=(page + 1) * 100 >= result["total"]
    ):
        st.session_state.eval_log_page = page + 1
        st.rerun()
    rows = {row["event_id"]: row for row in result["rows"]}
    event_id = st.selectbox(
        "Detail event",
        list(rows),
        format_func=lambda key: (
            f"#{key} · {rows[key]['event']} · {local_time(rows[key]['timestamp'])}"
        ),
        key="eval_event_detail_" + hashlib.sha256(signature.encode()).hexdigest()[:12],
    )
    row = rows[event_id]
    with st.expander("Metadata event", expanded=False):
        st.json(row["metadata"])
    interaction_id = row["metadata"].get("interaction_id")
    if interaction_id:
        st.markdown("**Rangkaian proses dalam interaksi ini**")
        timeline = service.store.interaction_events(role, interaction_id)
        st.dataframe(event_frame(timeline), hide_index=True, width="stretch")
        st.caption(
            "Satu interaksi dapat mencakup intent, tool, model, draft, dan validasi. Sub-run analisis memiliki ID tersendiri di metadata."
        )
    else:
        st.caption(
            "Event ini belum memiliki ID interaksi. Log lama tetap dapat ditelusuri melalui Session ID."
        )


def run_frame(runs):
    return pd.DataFrame(
        [
            {
                "Run ID": run["run_id"],
                "Waktu (Jakarta)": local_time(run["started_at"]),
                "Mode": run["mode"],
                "Model": run["model"],
                "Status": run["status"],
                "Lulus": run["summary"].get("passed"),
                "Gagal": run["summary"].get("failed"),
                "Pass rate (%)": run["summary"].get("pass_percent"),
            }
            for run in runs
        ]
    )


def result_detail(service, role, run_id, results):
    by_id = {result["case_id"]: result for result in results}
    case_id = st.selectbox(
        "Detail skenario",
        list(by_id),
        format_func=lambda value: by_id[value]["label"],
        key="eval_result_case_" + run_id,
    )
    result = by_id[case_id]
    st.markdown(f"**{result['label']} · {result['status'].upper()}**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Pemeriksaan": check["check"],
                    "Expected": json.dumps(check["expected"], ensure_ascii=False),
                    "Actual": json.dumps(check["actual"], ensure_ascii=False),
                    "Lulus": check["passed"],
                }
                for check in result["checks"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Percakapan pada dataset sintetis", expanded=False):
        for message in result["transcript"]:
            st.caption(message["role"].upper())
            st.text(message["content"])
    if result["actual"].get("draft"):
        with st.expander("Draft hasil evaluasi"):
            st.code(result["actual"]["draft"], language=None, wrap_lines=True)
    with st.expander("Filter, metrik, dan trace evaluasi"):
        st.json(
            {key: value for key, value in result["actual"].items() if key != "draft"},
            expanded=False,
        )
        if result["trace"]:
            st.dataframe(event_frame(result["trace"]), hide_index=True, width="stretch")
    ratings = ["Sesuai", "Sebagian sesuai", "Tidak sesuai"]
    suffix = run_id + "_" + case_id
    with st.form("eval_review_form_" + suffix):
        st.markdown("**Review manual**")
        rating = st.selectbox(
            "Penilaian hasil",
            ratings,
            index=ratings.index(result["review_rating"])
            if result["review_rating"] in ratings
            else 0,
            key="eval_review_rating_" + suffix,
        )
        note = st.text_area(
            "Catatan reviewer",
            value=result["review_note"] or "",
            max_chars=2000,
            key="eval_review_note_" + suffix,
            placeholder="Jelaskan ketepatan, kekurangan, atau skenario regresi yang perlu ditambahkan.",
        )
        save = st.form_submit_button("Simpan review", key="eval_save_review")
    if result["reviewed_at"]:
        st.caption("Review terakhir: " + local_time(result["reviewed_at"]))
    if save:
        service.store.review(role, run_id, case_id, rating, note)
        st.session_state.eval_notice = (
            "Review tersimpan. Penilaian manual ditampilkan terpisah dari pemeriksaan otomatis."
        )
        st.rerun()


def testing(service, role):
    st.markdown("**Jalankan evaluasi terkontrol**")
    st.caption(
        "Evaluasi memakai 6 tiket sintetis pada database sementara. Setiap skenario dimulai dari data baru. Hasil tersimpan; membuka halaman tidak menjalankan pengujian."
    )
    scenarios = service.scenarios(role)
    labels = {case["case_id"]: f"{case['category']} · {case['label']}" for case in scenarios}
    mode = st.selectbox("Mode evaluasi", MODES, key="eval_mode")
    if mode == "Mock":
        st.info(
            "Mock menguji kontrak output, grounding, dan alur aplikasi. Hasilnya tidak mengukur kualitas model Groq live."
        )
    elif mode == "Live Groq":
        st.info(
            "Live Groq mengirim data sintetis ke model yang dikonfigurasi. Semua skenario dapat dipilih dan dijalankan berurutan; token aktual dicatat jika provider mengembalikannya."
        )
    else:
        st.caption(
            "Local menguji pencarian, konteks, draft dasar, dan persetujuan tanpa panggilan model."
        )
    select_all = st.checkbox("Pilih semua skenario", key="eval_all_" + mode)
    if select_all:
        st.session_state["eval_cases_" + mode] = list(labels)
    selected = st.multiselect(
        "Skenario evaluasi",
        list(labels),
        default=list(labels) if mode != "Live Groq" else ["draft"],
        format_func=lambda value: labels[value],
        key="eval_cases_" + mode,
        disabled=select_all,
    )
    if st.button(
        "Jalankan evaluasi",
        key="eval_run",
        type="primary",
        disabled=not selected,
    ):
        key = groq_key() if mode == "Live Groq" else ""
        if mode == "Live Groq" and not key:
            st.error("API key Groq belum tersedia. Pilih Local/Mock atau konfigurasikan Groq.")
        else:
            bar = st.progress(0, text="Menyiapkan dataset uji...")

            def progress(done, total, label):
                bar.progress(done / total, text=f"{done}/{total} · {label}")

            with st.spinner("Menjalankan skenario pada database uji..."):
                run_id = service.run(role, mode, selected, api_key=key, progress=progress)
            st.session_state.eval_selected_run = run_id
            st.session_state.eval_notice = "Evaluasi selesai dan hasil tersimpan."
            st.rerun()
    runs = service.store.runs(role)
    if not runs:
        st.info("Belum ada hasil evaluasi. Jalankan mode Local atau Mock untuk memulai.")
        return
    st.markdown("**Riwayat evaluasi**")
    st.caption(
        "Menampilkan 100 run terbaru. Pass rate adalah proporsi skenario yang memenuhi pemeriksaan terstruktur. Ketepatan semantik tetap memerlukan review manual. Riwayat ini tidak menggantikan hasil pytest aplikasi."
    )
    frame = run_frame(runs)
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "Unduh riwayat (.csv)",
        csv_bytes(frame),
        "companion-evaluation-runs.csv",
        "text/csv",
        key="eval_runs_download",
    )
    by_id = {run["run_id"]: run for run in runs}
    chosen = st.session_state.get("eval_selected_run")
    if chosen not in by_id:
        st.session_state.eval_selected_run = runs[0]["run_id"]
    run_id = st.selectbox(
        "Run untuk ditinjau",
        list(by_id),
        format_func=lambda value: (
            f"{local_time(by_id[value]['started_at'])} · {by_id[value]['mode']} · {value[:8]}"
        ),
        key="eval_selected_run",
    )
    run = by_id[run_id]
    if run["status"] == "running":
        st.info(
            "Run ini belum selesai atau terputus. Hanya hasil skenario yang sudah tersimpan ditampilkan."
        )
    elif run["status"] == "failed":
        st.warning("Eksekusi run gagal; hasil parsial tetap tersedia.")
    with st.expander("Versi dan konfigurasi run"):
        st.json(
            {
                key: run[key]
                for key in (
                    "mode",
                    "model",
                    "suite_version",
                    "code_digest",
                    "prompt_digest",
                    "dataset_digest",
                    "config",
                    "summary",
                )
            }
        )
    results = service.store.results(role, run_id)
    if results:
        frame = pd.DataFrame(
            [
                {
                    "Skenario": result["case_id"],
                    "Kategori": result["category"],
                    "Hasil": result["status"],
                    "Durasi (ms)": result["duration_ms"],
                    "Review": result["review_rating"] or "Belum direview",
                }
                for result in results
            ]
        )
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button(
            "Unduh hasil lengkap (.json)",
            json.dumps({"run": run, "results": results}, ensure_ascii=False, indent=2).encode(
                "utf-8"
            ),
            "companion-evaluation.json",
            "application/json",
            key="eval_results_download",
        )
        result_detail(service, role, run_id, results)


def comparison(service, role):
    runs = service.store.runs(role)
    completed = {run["run_id"]: run for run in runs if run["status"] == "completed"}
    if len(completed) < 2:
        st.info("Selesaikan minimal dua run untuk membandingkan hasil evaluasi.")
        return

    def label(value):
        run = completed[value]
        return f"{local_time(run['started_at'])} · {run['mode']} · {value[:8]}"

    a, b = st.columns(2)
    left = a.selectbox(
        "Run A · baseline", list(completed), index=1, format_func=label, key="eval_compare_left"
    )
    right = b.selectbox(
        "Run B · pembanding", list(completed), format_func=label, key="eval_compare_right"
    )
    if left == right:
        st.info("Pilih dua run berbeda.")
        return
    result = service.compare(role, left, right)
    if not result["comparable"]:
        st.warning(
            "Mode, dataset, versi suite, atau cakupan skenario berbeda. Perbandingan bersifat deskriptif; jangan menyimpulkan peningkatan kualitas dari pass rate saja."
        )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Konfigurasi": field,
                    "Run A": str(result["left"][field]),
                    "Run B": str(result["right"][field]),
                }
                for field in (
                    "mode",
                    "model",
                    "suite_version",
                    "code_digest",
                    "prompt_digest",
                    "dataset_digest",
                )
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    frame = pd.DataFrame(result["rows"])
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "Unduh perbandingan (.csv)",
        csv_bytes(frame),
        "companion-comparison.csv",
        "text/csv",
        key="eval_compare_download",
    )


def render(services: AppServices) -> None:
    role = st.session_state.get("selected_role", "")
    if role != Role.DATA_ADMIN.value:
        st.error("Companion Evaluation hanya tersedia untuk Data Admin.")
        return
    header(
        "Companion Evaluation",
        "Pantau proses Agentic AI, evaluasi skenario, dan tinjau kualitas hasil Companion.",
        "Data Admin Workspace",
    )
    notice = st.session_state.pop("eval_notice", "")
    if notice:
        st.success(notice)
    today = CompanionConversation.today()
    start_col, end_col, refresh = st.columns([2, 2, 1])
    start = start_col.date_input(
        "Log mulai", today - timedelta(days=29), max_value=today, key="eval_log_start"
    )
    end = end_col.date_input("Log hingga", today, max_value=today, key="eval_log_end")
    if refresh.button("Refresh", key="eval_refresh", width="stretch"):
        st.rerun()
    st.caption(
        "Filter tanggal berlaku untuk Overview dan Agent Logs. Riwayat evaluasi memiliki pilihan run sendiri."
    )
    tabs = st.tabs(["Overview", "Agent Logs", "Testing & Evaluation", "Run Comparison"])
    try:
        with tabs[0]:
            if start <= end:
                overview(services.evaluation, role, start, end)
            else:
                st.info("Pilih rentang tanggal log yang valid.")
        with tabs[1]:
            if start <= end:
                logs(services.evaluation, role, start, end)
        with tabs[2]:
            testing(services.evaluation, role)
        with tabs[3]:
            comparison(services.evaluation, role)
    except (ValueError, OSError, sqlite3.Error) as exc:
        st.error(
            "Evaluasi atau pembacaan log gagal. Hasil yang sudah tersimpan tetap tersedia; periksa database dan konfigurasi."
        )
        st.caption("Kode error: " + type(exc).__name__)
