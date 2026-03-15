"""
app/ui/agent_app.py
Agent-powered Streamlit UI.
Run: streamlit run app/ui/agent_app.py

Design:
  - Upload data → agents auto-run → live status panel → results per agent tab
  - No manual column selection required — agents figure it out
  - User can override date/KPI columns in sidebar if needed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import streamlit as st
import pandas as pd
import numpy as np

from agents.context import AnalysisContext, AgentResult
from agents.runner import AgentRunner
from connectors.csv_connector import CSVConnector
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
    page_title="AI Analyst — Agent Mode",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# CSS
# ------------------------------------------------------------------

st.markdown("""
<style>
.agent-card {
    border: 1px solid #E2E8F0; border-radius: 8px;
    padding: 10px 14px; margin: 6px 0; background: #F8FAFC;
}
.agent-running  { border-left: 4px solid #F59E0B; background: #FFFBEB; }
.agent-success  { border-left: 4px solid #22C55E; background: #F0FDF4; }
.agent-skipped  { border-left: 4px solid #94A3B8; background: #F8FAFC; }
.agent-error    { border-left: 4px solid #EF4444; background: #FEF2F2; }
.agent-pending  { border-left: 4px solid #CBD5E1; background: #F8FAFC; }
.badge-success  { color: #16A34A; font-weight: 600; }
.badge-skipped  { color: #64748B; }
.badge-error    { color: #DC2626; font-weight: 600; }
.badge-running  { color: #D97706; font-weight: 600; }
.followup-box {
    background: #EFF6FF; border-left: 4px solid #3B82F6;
    padding: 10px 14px; border-radius: 4px; margin: 6px 0; font-size: 0.9rem;
}
.brief-box {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    padding: 18px 22px; border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar — overrides
# ------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 AI Analyst")
    st.caption("Agent Mode — v0.3")
    st.divider()

    st.markdown("**Column Overrides**")
    st.caption("Leave blank to let agents auto-detect.")
    manual_date = st.text_input("Date column (optional)", placeholder="e.g. created_at")
    manual_kpi = st.text_input("KPI column (optional)", placeholder="e.g. revenue")
    grain = st.radio("Time grain", TIME_GRAINS, horizontal=True)

    st.divider()
    st.markdown("**LLM Settings**")
    llm_provider = st.selectbox("Provider", ["openai", "anthropic"])
    llm_model = st.text_input("Model", value=config.LLM_MODEL)
    api_key_set = bool(config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY)
    if api_key_set:
        st.success("API key detected ✓")
    else:
        st.warning("No API key — LLM steps will use rule-based fallback.")

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.title("🤖 AI Analyst — Agent Mode")
st.caption("Upload any dataset. Agents auto-profile, analyse, and brief you.")
st.divider()

# ------------------------------------------------------------------
# File upload
# ------------------------------------------------------------------

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("⬆️ Upload a CSV to start the agent pipeline.")
    st.stop()

connector = CSVConnector()
df = connector.load_from_uploaded_file(uploaded)
st.success(f"Loaded **{len(df):,} rows × {len(df.columns)} columns** — `{uploaded.name}`")

# ------------------------------------------------------------------
# Build context
# ------------------------------------------------------------------

context = AnalysisContext(
    df=df,
    grain=grain,
    filename=uploaded.name,
)
if manual_date:
    context.date_col = manual_date
if manual_kpi:
    context.kpi_col = manual_kpi

# ------------------------------------------------------------------
# Agent status panel
# ------------------------------------------------------------------

st.subheader("🔄 Agent Pipeline")

STATUS_ICONS = {
    "pending":  "⬜",
    "running":  "🟡",
    "success":  "🟢",
    "skipped":  "⚪",
    "error":    "🔴",
}

ALL_AGENT_NAMES = ["eda", "orchestrator", "trend", "anomaly",
                   "root_cause", "funnel", "cohort", "insight"]
AGENT_LABELS = {
    "eda":          "EDA — Profile & detect columns",
    "orchestrator": "Orchestrator — Plan agent pipeline",
    "trend":        "Trend — Time series & comparisons",
    "anomaly":      "Anomaly — Outlier detection",
    "root_cause":   "Root Cause — Driver attribution",
    "funnel":       "Funnel — Conversion analysis",
    "cohort":       "Cohort — Retention analysis",
    "insight":      "Insight — Brief & follow-ups",
}

# Initialise status state
if "agent_statuses" not in st.session_state:
    st.session_state.agent_statuses = {n: "pending" for n in ALL_AGENT_NAMES}
if "agent_results" not in st.session_state:
    st.session_state.agent_results = {}
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "context" not in st.session_state:
    st.session_state.context = None

status_placeholder = st.empty()


def render_status_panel(statuses: dict, results: dict):
    with status_placeholder.container():
        cols = st.columns(4)
        for i, name in enumerate(ALL_AGENT_NAMES):
            status = statuses.get(name, "pending")
            icon = STATUS_ICONS[status]
            result = results.get(name)
            duration = f" ({result.duration_sec}s)" if result and result.duration_sec else ""
            label = AGENT_LABELS.get(name, name)
            with cols[i % 4]:
                css_class = f"agent-card agent-{status}"
                badge_class = f"badge-{status}"
                summary_html = (
                    f'<div style="font-size:0.78rem;color:#64748B;margin-top:4px">'
                    f'{result.summary[:80]}{"..." if result and len(result.summary) > 80 else ""}'
                    f'</div>'
                ) if result else ""
                st.markdown(
                    f'<div class="{css_class}">'
                    f'{icon} <span class="{badge_class}">{status.upper()}</span>'
                    f'<span style="font-size:0.8rem;color:#475569"> {duration}</span><br>'
                    f'<strong style="font-size:0.85rem">{label}</strong>'
                    f'{summary_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


render_status_panel(st.session_state.agent_statuses, st.session_state.agent_results)

# ------------------------------------------------------------------
# Run pipeline button
# ------------------------------------------------------------------

run_col, reset_col = st.columns([2, 1])
with run_col:
    run_button = st.button(
        "🚀 Run Agent Pipeline",
        use_container_width=True,
        disabled=st.session_state.pipeline_done,
        type="primary",
    )
with reset_col:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.agent_statuses = {n: "pending" for n in ALL_AGENT_NAMES}
        st.session_state.agent_results = {}
        st.session_state.pipeline_done = False
        st.session_state.context = None
        st.rerun()

if run_button:
    # Reset statuses
    st.session_state.agent_statuses = {n: "pending" for n in ALL_AGENT_NAMES}
    st.session_state.agent_results = {}

    def on_start(name: str):
        st.session_state.agent_statuses[name] = "running"
        render_status_panel(st.session_state.agent_statuses, st.session_state.agent_results)

    def on_done(result: AgentResult):
        st.session_state.agent_statuses[result.agent] = result.status
        st.session_state.agent_results[result.agent] = result
        render_status_panel(st.session_state.agent_statuses, st.session_state.agent_results)

    runner = AgentRunner(max_workers=4)
    finished_context = runner.run(context, on_agent_start=on_start, on_agent_done=on_done)

    # Mark agents not in active plan as skipped
    for name in ALL_AGENT_NAMES:
        if name not in finished_context.results:
            st.session_state.agent_statuses[name] = "skipped"
            st.session_state.agent_results[name] = AgentResult(
                agent=name, status="skipped",
                summary="Not required for this dataset.", data={}
            )

    st.session_state.context = finished_context
    st.session_state.pipeline_done = True
    render_status_panel(st.session_state.agent_statuses, st.session_state.agent_results)

# ------------------------------------------------------------------
# Results (only shown after pipeline runs)
# ------------------------------------------------------------------

ctx: AnalysisContext = st.session_state.get("context")

if ctx is None:
    st.stop()

st.divider()
st.subheader("📋 Analysis Results")

tabs = st.tabs([
    "📝 Brief",
    "🔍 EDA",
    "📈 Trend",
    "⚠️ Anomalies",
    "🔎 Root Cause",
    "🔽 Funnel",
    "👥 Cohort",
])

# ==================================================================
# BRIEF TAB
# ==================================================================

with tabs[0]:
    insight = ctx.results.get("insight")
    orch = ctx.results.get("orchestrator")

    if orch and orch.status == "success":
        plan = orch.data.get("plan", [])
        st.info(f"**Agents activated:** {' → '.join(plan)}")

    if insight and insight.status == "success" and ctx.final_brief:
        st.markdown('<div class="brief-box">', unsafe_allow_html=True)
        st.markdown(ctx.final_brief)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Brief not available — run the pipeline first.")

    if ctx.follow_up_questions:
        st.markdown("### 💡 Follow-up Questions")
        for q in ctx.follow_up_questions:
            st.markdown(f'<div class="followup-box">💡 {q}</div>', unsafe_allow_html=True)

    # Timing summary
    total_time = sum(
        r.duration_sec for r in ctx.results.values() if r.duration_sec
    )
    st.caption(f"Pipeline completed in ~{total_time:.1f}s total agent time.")

# ==================================================================
# EDA TAB
# ==================================================================

with tabs[1]:
    eda = ctx.results.get("eda")
    if eda and eda.status == "success":
        q = eda.data["quality"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{q['total_rows']:,}")
        c2.metric("Columns", q["total_columns"])
        c3.metric("Completeness", f"{q['completeness_pct']}%")
        c4.metric("Duplicates", q["duplicate_rows"])
        c5.metric("Suggested KPIs", len(eda.data["kpis"]))

        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Detected KPIs:** {', '.join(eda.data['kpis']) or 'None'}")
            st.info(f"**Dimensions:** {', '.join(eda.data['dimensions']) or 'None'}")
        with col_b:
            st.info(f"**Date column:** {eda.data['date_col'] or 'Not detected'}")
            st.info(f"**Funnel signal:** {'Yes' if eda.data['has_funnel_signal'] else 'No'} | "
                    f"**Cohort signal:** {'Yes' if eda.data['has_cohort_signal'] else 'No'}")

        st.markdown("**Column Profile**")
        st.dataframe(eda.data["profile_df"], use_container_width=True)
    else:
        st.warning(f"EDA: {eda.summary if eda else 'Not run.'}")

# ==================================================================
# TREND TAB
# ==================================================================

with tabs[2]:
    trend = ctx.results.get("trend")
    if trend and trend.status == "success":
        ts = trend.data.get("ts", ctx.ts)
        date_col = ctx.date_col
        kpi_col = ctx.kpi_col
        comparisons = trend.data.get("comparisons", {})

        t1, t2, t3 = st.columns(3)
        t1.metric("Direction", trend.data["trend_direction"].title())
        t2.metric("Overall Trend", f"{trend.data['trend_pct']:+.1f}%")
        t3.metric("Latest Value", f"{trend.data['latest_value']:,.2f}" if trend.data.get("latest_value") else "—")

        if not ts.empty and date_col in ts.columns and kpi_col in ts.columns:
            st.plotly_chart(
                trend_with_anomalies(ts, date_col, kpi_col,
                                     title=f"{kpi_col} — {ctx.grain}"),
                use_container_width=True,
            )

        if comparisons:
            st.markdown("**Period Comparisons**")
            comp_cols = st.columns(len(comparisons))
            for col_widget, (comp_name, comp) in zip(comp_cols, comparisons.items()):
                arrow = "▲" if comp["pct_change"] >= 0 else "▼"
                col_widget.metric(comp_name, f"{comp['current']:,.0f}",
                                  f"{arrow} {comp['pct_change']:+.1f}%")
            st.plotly_chart(
                kpi_comparison_bar(list(comparisons.values())),
                use_container_width=True,
            )
    else:
        st.warning(f"Trend: {trend.summary if trend else 'Not run or skipped.'}")

# ==================================================================
# ANOMALY TAB
# ==================================================================

with tabs[3]:
    anom = ctx.results.get("anomaly")
    if anom and anom.status == "success":
        ts_a = anom.data.get("ts_with_anomalies", ctx.ts)
        date_col = ctx.date_col
        kpi_col = ctx.kpi_col
        sev = anom.data.get("severity_counts", {})

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Method", anom.data.get("method_used", "—"))
        a2.metric("Anomalies", anom.data.get("anomaly_count", 0))
        a3.metric("High Severity", sev.get("high", 0))
        a4.metric("Medium Severity", sev.get("medium", 0))

        if not ts_a.empty and date_col in ts_a.columns and kpi_col in ts_a.columns:
            st.plotly_chart(
                trend_with_anomalies(ts_a, date_col, kpi_col,
                                     title=f"{kpi_col} — {anom.data.get('method_used')} Anomalies"),
                use_container_width=True,
            )

        records = anom.data.get("anomaly_records", [])
        if records:
            st.markdown("**Flagged Points**")
            st.dataframe(pd.DataFrame(records), use_container_width=True)
    else:
        st.warning(f"Anomaly: {anom.summary if anom else 'Not run or skipped.'}")

# ==================================================================
# ROOT CAUSE TAB
# ==================================================================

with tabs[4]:
    rc = ctx.results.get("root_cause")
    if rc and rc.status == "success":
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Current Period", f"{rc.data['last_total']:,.0f}")
        r2.metric("Prior Period", f"{rc.data['prev_total']:,.0f}")
        r3.metric("Delta", f"{rc.data['delta']:+,.0f}")
        r4.metric("% Change", f"{rc.data['pct_change']:+.1f}%")

        drivers = rc.data.get("drivers", pd.DataFrame())
        if not drivers.empty:
            st.plotly_chart(
                driver_bar_chart(drivers, title=f"Top Drivers — {ctx.kpi_col}"),
                use_container_width=True,
            )
            movers = rc.data.get("movers", {})
            neg_col, pos_col = st.columns(2)
            with neg_col:
                st.markdown("**Top Negative Drivers**")
                neg = pd.DataFrame(movers.get("negative", []))
                if not neg.empty:
                    st.dataframe(neg, use_container_width=True)
            with pos_col:
                st.markdown("**Top Positive Drivers**")
                pos = pd.DataFrame(movers.get("positive", []))
                if not pos.empty:
                    st.dataframe(pos, use_container_width=True)

        contributions = rc.data.get("contributions", {})
        if contributions:
            st.markdown("**Contribution by Dimension**")
            for dim, contrib_df in contributions.items():
                with st.expander(f"{dim}"):
                    if not contrib_df.empty:
                        st.plotly_chart(
                            contribution_bar(contrib_df, dim, title=f"{ctx.kpi_col} share by {dim}"),
                            use_container_width=True,
                        )
    else:
        st.warning(f"Root Cause: {rc.summary if rc else 'Not run or skipped.'}")

# ==================================================================
# FUNNEL TAB
# ==================================================================

with tabs[5]:
    funnel = ctx.results.get("funnel")
    if funnel and funnel.status == "success":
        f1, f2, f3 = st.columns(3)
        f1.metric("Stages", len(funnel.data.get("stages", [])))
        f2.metric("Top of Funnel", f"{funnel.data.get('top_of_funnel_users', 0):,}")
        f3.metric("Overall Conversion", f"{funnel.data.get('overall_conversion_pct', 0):.1f}%")

        biggest = funnel.data.get("biggest_drop", {})
        if biggest:
            st.warning(
                f"📉 Biggest drop at **{biggest['stage']}** — "
                f"{biggest['drop_off_pct']:.1f}% drop-off "
                f"({biggest.get('users_lost', 0):,} users lost)"
            )

        funnel_df = funnel.data.get("funnel_df", pd.DataFrame())
        if not funnel_df.empty:
            st.plotly_chart(funnel_chart(funnel_df), use_container_width=True)
            st.dataframe(funnel_df, use_container_width=True)
    else:
        st.warning(f"Funnel: {funnel.summary if funnel else 'Not run or skipped.'}")

# ==================================================================
# COHORT TAB
# ==================================================================

with tabs[6]:
    cohort = ctx.results.get("cohort")
    if cohort and cohort.status == "success":
        c1, c2, c3 = st.columns(3)
        c1.metric("Unique Users", f"{cohort.data.get('n_users', 0):,}")
        c2.metric("Date Range", f"{cohort.data.get('date_range_days', 0)} days")
        c3.metric("Cohort Grain", cohort.data.get("grain", "—"))

        matrix = cohort.data.get("retention_matrix")
        if matrix is not None and not matrix.empty:
            st.plotly_chart(
                cohort_heatmap(matrix, title="User Retention Heatmap"),
                use_container_width=True,
            )
            st.dataframe(
                matrix.style.format("{:.1f}%", na_rep="—").background_gradient(
                    cmap="Blues", axis=None
                ),
                use_container_width=True,
            )
        else:
            st.info("Retention matrix could not be computed for this dataset.")
    else:
        st.warning(f"Cohort: {cohort.summary if cohort else 'Not run or skipped.'}")
