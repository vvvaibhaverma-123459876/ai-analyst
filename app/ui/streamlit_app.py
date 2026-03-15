# DEPRECATED — This UI version is superseded by v06_app.py (v0.6+).
# It is retained for reference only and will be removed in v10.
# Do not add new functionality here.

"""
app/ui/streamlit_app.py
Main Streamlit application. Wires all modules together.
Run: streamlit run app/ui/streamlit_app.py
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from connectors.csv_connector import CSVConnector
from analysis.eda_engine import EDAEngine
from analysis.anomaly_detector import AnomalyDetector
from analysis.root_cause import RootCauseAnalyzer
from analysis.funnel_analyzer import FunnelAnalyzer
from analysis.cohort_analyzer import CohortAnalyzer
from analysis.statistics import (
    resample_timeseries, period_comparison, add_trend_line, rolling_stats
)
from insights.summary_builder import SummaryBuilder
from memory.history_store import HistoryStore
from charts.chart_builder import (
    trend_with_anomalies, driver_bar_chart, funnel_chart,
    cohort_heatmap, contribution_bar, kpi_comparison_bar,
)
from core.config import config
from core.constants import TIME_GRAINS

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="AI Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Minimal CSS
# ------------------------------------------------------------------

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
.section-header { font-size: 1.1rem; font-weight: 600;
                  border-bottom: 2px solid #3B82F6;
                  padding-bottom: 4px; margin-bottom: 12px; }
.followup-box { background: #F1F5F9; border-left: 4px solid #3B82F6;
                padding: 10px 14px; border-radius: 4px; margin: 6px 0;
                font-size: 0.9rem; cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Module instances (cached)
# ------------------------------------------------------------------

@st.cache_resource
def get_modules():
    return {
        "eda": EDAEngine(),
        "anomaly": AnomalyDetector(),
        "root_cause": RootCauseAnalyzer(),
        "funnel": FunnelAnalyzer(),
        "cohort": CohortAnalyzer(),
        "summary_builder": SummaryBuilder(),
        "history": HistoryStore(),
    }

mods = get_modules()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("📊 AI Analyst")
    st.caption("v0.2 — Modular Edition")
    st.divider()

    analysis_mode = st.radio(
        "Analysis Mode",
        ["CSV Upload", "Athena Query"],
        help="CSV for local files. Athena for warehouse queries.",
    )

    st.divider()
    st.markdown("**LLM Settings**")
    llm_provider = st.selectbox(
        "Provider",
        ["openai", "anthropic"],
        index=0 if config.LLM_PROVIDER == "openai" else 1,
        help="Set LLM_PROVIDER in .env to persist.",
    )
    llm_model = st.text_input("Model", value=config.LLM_MODEL)

    st.divider()
    st.markdown("**Session History**")
    history_records = mods["history"].recent(5)
    if history_records:
        for rec in history_records:
            st.caption(f"🕐 {rec['timestamp'][:16]}  
{rec['question'][:50]}")
    else:
        st.caption("No history yet.")

    if st.button("Clear History", use_container_width=True):
        mods["history"].clear()
        st.success("History cleared.")

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.title("📊 AI Analyst")
st.caption("Upload a CSV → Profile → Trend → Anomalies → Drivers → Root Cause → Funnel → Cohort → Summary")
st.divider()

# ------------------------------------------------------------------
# DATA LOAD
# ------------------------------------------------------------------

df = None
connector = CSVConnector()

if analysis_mode == "CSV Upload":
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = connector.load_from_uploaded_file(uploaded)

elif analysis_mode == "Athena Query":
    st.info("Configure Athena credentials in your `.env` file, then enter a query below.")
    athena_sql = st.text_area("Athena SQL", height=100,
                              placeholder="SELECT * FROM daily_metrics LIMIT 10000")
    if st.button("Run Athena Query"):
        try:
            from connectors.athena_connector import AthenaConnector
            ac = AthenaConnector()
            ac.connect()
            df = ac.execute(athena_sql)
            st.success(f"Query returned {len(df):,} rows.")
        except Exception as e:
            st.error(f"Athena error: {e}")

if df is None:
    st.info("⬆️ Upload a CSV file or run an Athena query to get started.")
    st.stop()

st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

# ------------------------------------------------------------------
# TAB LAYOUT
# ------------------------------------------------------------------

tab_profile, tab_trend, tab_anomaly, tab_drivers, tab_funnel, tab_cohort, tab_summary = st.tabs([
    "🔍 Profile",
    "📈 Trend",
    "⚠️ Anomalies",
    "🔎 Drivers",
    "🔽 Funnel",
    "👥 Cohort",
    "📝 Summary",
])

# ==================================================================
# TAB 1 — DATA PROFILE
# ==================================================================

with tab_profile:
    st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="section-header">Quality Report</div>', unsafe_allow_html=True)
    quality = mods["eda"].quality_report(df)
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Rows", f"{quality['total_rows']:,}")
    q2.metric("Columns", quality['total_columns'])
    q3.metric("Complete Rows", f"{quality['complete_rows']:,}")
    q4.metric("Completeness", f"{quality['completeness_pct']}%")
    q5.metric("Duplicates", quality['duplicate_rows'])

    st.markdown('<div class="section-header">Column Profile</div>', unsafe_allow_html=True)
    st.dataframe(mods["eda"].profile(df), use_container_width=True)

    inferred_kpis = mods["eda"].infer_kpis(df)
    inferred_dims = mods["eda"].infer_dimensions(df)
    col_a, col_b = st.columns(2)
    col_a.info(f"**Suggested KPIs:** {', '.join(inferred_kpis) or 'None detected'}")
    col_b.info(f"**Suggested Dimensions:** {', '.join(inferred_dims) or 'None detected'}")

# ==================================================================
# SHARED CONTROLS (used by Trend, Anomaly, Drivers tabs)
# ==================================================================

st.sidebar.divider()
st.sidebar.markdown("**Column Mappings**")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
date_guess = CSVConnector.detect_datetime_column(df)
all_cols = df.columns.tolist()

date_col = st.sidebar.selectbox(
    "Date column",
    all_cols,
    index=all_cols.index(date_guess) if date_guess else 0,
)
kpi_col = st.sidebar.selectbox(
    "KPI column",
    numeric_cols,
    index=0,
)
grain = st.sidebar.radio("Time grain", TIME_GRAINS, horizontal=True)

# Parse dates
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df_valid = df[df[date_col].notna()].copy()

# Build base time series
ts_base = resample_timeseries(df_valid, date_col, kpi_col, grain)
ts_base = rolling_stats(ts_base, kpi_col)
ts_base = add_trend_line(ts_base, date_col, kpi_col)

# Period comparisons
comparisons = []
for comp in ["DoD", "WoW", "MoM"]:
    try:
        comparisons.append(period_comparison(df_valid, date_col, kpi_col, comp))
    except Exception:
        pass

# ==================================================================
# TAB 2 — TREND
# ==================================================================

with tab_trend:
    st.markdown('<div class="section-header">KPI Trend</div>', unsafe_allow_html=True)

    show_trend = st.checkbox("Show trend line", value=True)
    show_bands = st.checkbox("Show ±2σ bands", value=True)

    fig_ts = ts_base.copy()
    if not show_trend and "trend" in fig_ts.columns:
        fig_ts = fig_ts.drop(columns=["trend"])
    if not show_bands and "upper_band" in fig_ts.columns:
        fig_ts = fig_ts.drop(columns=["upper_band", "lower_band"])

    st.plotly_chart(
        trend_with_anomalies(fig_ts, date_col, kpi_col, title=f"{kpi_col} — {grain}"),
        use_container_width=True,
    )

    if comparisons:
        st.markdown('<div class="section-header">Period-over-Period</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, comp in zip([c1, c2, c3], comparisons):
            arrow = "▲" if comp["pct_change"] >= 0 else "▼"
            col.metric(
                comp["comparison"],
                f"{comp['current']:,.0f}",
                f"{arrow} {comp['pct_change']:+.1f}%",
            )
        st.plotly_chart(
            kpi_comparison_bar(comparisons),
            use_container_width=True,
        )

# ==================================================================
# TAB 3 — ANOMALY DETECTION
# ==================================================================

with tab_anomaly:
    st.markdown('<div class="section-header">Anomaly Detection</div>', unsafe_allow_html=True)

    col_method, col_window, col_thresh = st.columns(3)
    with col_method:
        method = st.selectbox("Method", ["Z-Score", "IQR", "STL"])
    with col_window:
        window = st.slider("Rolling window", 3, 30, config.DEFAULT_ROLLING_WINDOW)
    with col_thresh:
        z_thresh = st.slider("Threshold", 1.0, 5.0, config.DEFAULT_Z_THRESHOLD)

    detector = AnomalyDetector(window=window, z_threshold=z_thresh)

    if method == "Z-Score":
        ts_anom = detector.detect_zscore(ts_base, date_col, kpi_col, window, z_thresh)
    elif method == "IQR":
        ts_anom = detector.detect_iqr(ts_base, kpi_col)
    else:
        ts_anom = detector.detect_stl(ts_base, date_col, kpi_col)

    anom_count = int(ts_anom["anomaly"].sum())
    st.metric("Anomalies detected", anom_count)

    st.plotly_chart(
        trend_with_anomalies(ts_anom, date_col, kpi_col,
                             title=f"{kpi_col} — {method} Anomaly Detection"),
        use_container_width=True,
    )

    if anom_count > 0:
        st.markdown("**Anomalous Points**")
        anom_rows = ts_anom[ts_anom["anomaly"] == True][[date_col, kpi_col, "zscore" if "zscore" in ts_anom.columns else kpi_col]]
        st.dataframe(anom_rows.reset_index(drop=True), use_container_width=True)

    st.session_state["ts_anom"] = ts_anom
    st.session_state["anom_count"] = anom_count

# ==================================================================
# TAB 4 — DRIVER ATTRIBUTION & ROOT CAUSE
# ==================================================================

with tab_drivers:
    st.markdown('<div class="section-header">Driver Attribution</div>', unsafe_allow_html=True)

    col_days, col_dim = st.columns(2)
    with col_days:
        driver_days = st.slider("Compare last N days", 2, 30, config.DEFAULT_DRIVER_DAYS)
    with col_dim:
        cat_cols = [c for c in df_valid.columns if df_valid[c].dtype == "object"]
        dim_filter = st.multiselect(
            "Focus dimensions (leave blank = all)",
            cat_cols,
        )

    result = mods["root_cause"].driver_attribution(df_valid, date_col, kpi_col, driver_days)
    delta = result["delta"]
    pct = result["pct_change"]
    drivers = result["drivers"]

    if dim_filter and not drivers.empty:
        drivers = drivers[drivers["dimension"].isin(dim_filter)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Period", f"{result['last_total']:,.0f}")
    m2.metric("Prior Period", f"{result['prior_total']:,.0f}")
    m3.metric("Delta", f"{delta:+,.0f}")
    m4.metric("% Change", f"{pct:+.1f}%")

    if not drivers.empty:
        st.plotly_chart(
            driver_bar_chart(drivers, title=f"Top Drivers for {kpi_col}"),
            use_container_width=True,
        )

        movers = mods["root_cause"].top_movers(drivers)

        col_neg, col_pos = st.columns(2)
        with col_neg:
            st.markdown("**Top Negative Drivers**")
            neg_df = pd.DataFrame(movers["negative"])
            if not neg_df.empty:
                st.dataframe(neg_df, use_container_width=True)

        with col_pos:
            st.markdown("**Top Positive Drivers**")
            pos_df = pd.DataFrame(movers["positive"])
            if not pos_df.empty:
                st.dataframe(pos_df, use_container_width=True)

    st.markdown('<div class="section-header">Contribution Analysis</div>', unsafe_allow_html=True)
    if cat_cols:
        contrib_dim = st.selectbox("Dimension for contribution", cat_cols)
        contrib_df = mods["root_cause"].contribution_analysis(
            df_valid, kpi_col, contrib_dim, date_col
        )
        st.plotly_chart(
            contribution_bar(contrib_df, contrib_dim,
                             title=f"{kpi_col} share by {contrib_dim}"),
            use_container_width=True,
        )
        st.dataframe(contrib_df, use_container_width=True)

    # Store for summary tab
    st.session_state["driver_result"] = result
    st.session_state["movers"] = movers

# ==================================================================
# TAB 5 — FUNNEL ANALYSIS
# ==================================================================

with tab_funnel:
    st.markdown('<div class="section-header">Funnel Analysis</div>', unsafe_allow_html=True)

    st.info(
        "Funnel analysis requires an **event-level** dataset with a stage/event column "
        "and a user ID column. Configure below."
    )

    f1, f2 = st.columns(2)
    with f1:
        stage_col = st.selectbox(
            "Stage / Event column",
            [c for c in df.columns if df[c].dtype == "object"],
            key="funnel_stage_col",
        )
    with f2:
        user_col_opts = [c for c in df.columns if "user" in c.lower() or "id" in c.lower()]
        user_col = st.selectbox(
            "User ID column",
            user_col_opts if user_col_opts else df.columns.tolist(),
            key="funnel_user_col",
        )

    detected_stages = df[stage_col].dropna().unique().tolist() if stage_col else []
    selected_stages = st.multiselect(
        "Select and order stages (top → bottom)",
        detected_stages,
        default=detected_stages[:6] if len(detected_stages) >= 6 else detected_stages,
    )

    if selected_stages and stage_col and user_col:
        funnel_df = mods["funnel"].compute_funnel(df, stage_col, user_col, selected_stages)

        st.plotly_chart(
            funnel_chart(funnel_df, title=f"Funnel: {stage_col}"),
            use_container_width=True,
        )
        st.dataframe(funnel_df, use_container_width=True)

        biggest = mods["funnel"].biggest_drop(funnel_df)
        if biggest:
            st.warning(
                f"📉 Biggest drop at **{biggest['stage']}** — "
                f"{biggest['drop_off_pct']:.1f}% drop-off "
                f"({biggest['users_lost']:,} users lost)"
            )

        seg_col_opts = [c for c in df.columns if df[c].dtype == "object" and c != stage_col]
        if seg_col_opts:
            seg_col = st.selectbox("Compare by segment", ["None"] + seg_col_opts)
            if seg_col != "None":
                pivot = mods["funnel"].compare_funnels(
                    df, stage_col, user_col, seg_col, selected_stages
                )
                st.markdown(f"**Conversion from top by {seg_col}**")
                st.dataframe(pivot.style.format("{:.1f}%").background_gradient(
                    cmap="Blues", axis=None
                ), use_container_width=True)
    else:
        st.warning("Select at least one stage to compute the funnel.")

# ==================================================================
# TAB 6 — COHORT ANALYSIS
# ==================================================================

with tab_cohort:
    st.markdown('<div class="section-header">Cohort Analysis</div>', unsafe_allow_html=True)

    st.info(
        "Cohort analysis requires a **user ID column** and an **activity date column**. "
        "Each row should represent one user-activity event."
    )

    coh1, coh2, coh3 = st.columns(3)
    with coh1:
        user_id_col = st.selectbox(
            "User ID column",
            [c for c in df.columns if "user" in c.lower() or "id" in c.lower()] or df.columns.tolist(),
            key="cohort_user_col",
        )
    with coh2:
        activity_date_col = st.selectbox(
            "Activity date column",
            [c for c in df.columns if "date" in c.lower() or "time" in c.lower()] or df.columns.tolist(),
            key="cohort_date_col",
        )
    with coh3:
        cohort_grain = st.selectbox("Cohort grain", ["M", "W", "D"], index=0)

    if st.button("Build Retention Matrix", use_container_width=True):
        try:
            cohort_df = df[[user_id_col, activity_date_col]].dropna()
            matrix = mods["cohort"].build_retention_matrix(
                cohort_df, user_id_col, activity_date_col, cohort_grain
            )
            if not matrix.empty:
                st.plotly_chart(
                    cohort_heatmap(matrix, title="User Retention Heatmap"),
                    use_container_width=True,
                )
                st.dataframe(
                    matrix.style.format("{:.1f}%", na_rep="-").background_gradient(
                        cmap="Blues", axis=None
                    ),
                    use_container_width=True,
                )
            else:
                st.warning("Not enough data to build a retention matrix.")
        except Exception as e:
            st.error(f"Cohort analysis failed: {e}")

    st.divider()
    st.markdown('<div class="section-header">Time to Convert</div>', unsafe_allow_html=True)

    event_col_opts = [c for c in df.columns if df[c].dtype == "object"]
    if event_col_opts:
        e1, e2, e3 = st.columns(3)
        with e1:
            event_col = st.selectbox("Event column", event_col_opts, key="ttc_event_col")
        with e2:
            event_vals = df[event_col].dropna().unique().tolist() if event_col else []
            signup_event = st.selectbox("Signup event", event_vals, key="ttc_signup")
        with e3:
            convert_event = st.selectbox("Conversion event", event_vals, key="ttc_convert")

        if st.button("Compute Time-to-Convert"):
            try:
                ttc_df = mods["cohort"].time_to_convert(
                    df, user_id_col, signup_event, convert_event, event_col, activity_date_col
                )
                if not ttc_df.empty:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Converted Users", f"{len(ttc_df):,}")
                    c2.metric("Median Days", f"{ttc_df['days_to_convert'].median():.1f}")
                    c3.metric("Avg Days", f"{ttc_df['days_to_convert'].mean():.1f}")
                    st.dataframe(ttc_df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Time-to-convert failed: {e}")

# ==================================================================
# TAB 7 — EXECUTIVE SUMMARY
# ==================================================================

with tab_summary:
    st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)

    # Retrieve results from other tabs if available
    driver_result = st.session_state.get("driver_result", {})
    movers = st.session_state.get("movers", {})
    ts_anom = st.session_state.get("ts_anom", ts_base)
    anom_count = st.session_state.get("anom_count", 0)

    delta = driver_result.get("delta", 0)
    pct = driver_result.get("pct_change", 0)
    anomaly_list = mods["anomaly"].summarise(ts_anom, date_col, kpi_col) if anom_count else []

    # Metric recap
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("KPI", kpi_col)
    r2.metric("Overall Delta", f"{delta:+,.0f}")
    r3.metric("% Change", f"{pct:+.1f}%")
    r4.metric("Anomalies", anom_count)

    st.divider()

    # LLM-powered sections
    api_key_set = bool(config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY)
    if not api_key_set:
        st.warning(
            "No LLM API key found. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your `.env` file "
            "to enable AI-generated summaries."
        )

    col_exec, col_followup = st.columns([3, 2])

    with col_exec:
        st.markdown("**Executive Summary**")
        if st.button("Generate Executive Summary", use_container_width=True, disabled=not api_key_set):
            payload = mods["summary_builder"].build_exec_payload(
                kpi_col=kpi_col,
                delta=delta,
                pct_change=pct,
                anomalies=anomaly_list,
                top_drivers=movers.get("negative", []) + movers.get("positive", []),
                period_last=driver_result.get("period_last"),
                period_prev=driver_result.get("period_prev"),
                comparisons={c["comparison"]: c for c in comparisons},
            )
            with st.spinner("Generating summary..."):
                try:
                    from llm.insight_generator import InsightGenerator
                    from llm.client import LLMClient
                    llm = LLMClient(provider=llm_provider, model=llm_model)
                    gen = InsightGenerator(llm)
                    summary = gen.executive_summary(payload)
                    st.markdown(summary)
                    st.session_state["exec_summary"] = summary
                except Exception as e:
                    st.error(f"Summary generation failed: {e}")

        if "exec_summary" in st.session_state:
            st.markdown(st.session_state["exec_summary"])

    with col_followup:
        st.markdown("**Follow-up Questions**")
        if st.button("Suggest Follow-ups", use_container_width=True, disabled=not api_key_set):
            analysis_summary = mods["summary_builder"].build_analysis_summary(
                kpi_col, ts_base, kpi_col, driver_result, anom_count
            )
            with st.spinner("Generating follow-up questions..."):
                try:
                    from llm.insight_generator import InsightGenerator
                    from llm.client import LLMClient
                    llm = LLMClient(provider=llm_provider, model=llm_model)
                    gen = InsightGenerator(llm)
                    questions = gen.follow_up_questions(
                        f"Analyse {kpi_col}", analysis_summary
                    )
                    st.session_state["followups"] = questions
                except Exception as e:
                    st.error(f"Follow-up generation failed: {e}")

        if "followups" in st.session_state:
            for q in st.session_state["followups"]:
                st.markdown(
                    f'<div class="followup-box">💡 {q}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()
    st.markdown("**Root Cause Narrative**")
    if st.button("Generate Root Cause Analysis", use_container_width=True, disabled=not api_key_set):
        with st.spinner("Analysing root cause..."):
            try:
                from llm.insight_generator import InsightGenerator
                from llm.client import LLMClient
                llm = LLMClient(provider=llm_provider, model=llm_model)
                gen = InsightGenerator(llm)
                narrative = gen.root_cause_narrative(
                    kpi=kpi_col,
                    delta=delta,
                    pct=pct,
                    drivers=movers.get("negative", [])[:5],
                    anomalies=anomaly_list[:5],
                )
                st.info(narrative)
            except Exception as e:
                st.error(f"Root cause narrative failed: {e}")

    # Save session to history
    if st.button("Save Session to History"):
        mods["history"].save(
            question=f"Analysis of {kpi_col}",
            kpi=kpi_col,
            delta=delta,
            pct_change=pct,
            summary=st.session_state.get("exec_summary", ""),
            followup_questions=st.session_state.get("followups", []),
        )
        st.success("Session saved to history.")
