"""
app/ui/autonomous_app.py — v0.4
Fully autonomous AI analyst Streamlit UI.

Flow:
  1. Upload any file(s) → IngestionEngine normalises everything
  2. ContextEngine generates upfront questions → user answers
  3. Agent pipeline auto-runs with live status panel
  4. OutputRouter decides modes → brief / alert / conversation
  5. Results shown in tabs + conversational chat always available

Run:  streamlit run app/ui/autonomous_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import streamlit as st
import pandas as pd
import numpy as np

from ingestion.ingestion_engine import IngestionEngine
from context_engine.context_engine import ContextEngine
from context_engine.org_memory import OrgMemory
from agents.context import AnalysisContext, AgentResult
from agents.runner import AgentRunner, AGENT_REGISTRY
from output.output_router import OutputRouter
from output.alert_dispatcher import AlertDispatcher
from output.conversation_engine import ConversationEngine
from charts.chart_builder import (
    trend_with_anomalies, driver_bar_chart, funnel_chart,
    cohort_heatmap, contribution_bar, kpi_comparison_bar,
)
from core.config import config
from core.constants import TIME_GRAINS

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Analyst — Autonomous",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.agent-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:8px 0; }
.agent-card {
    border-radius:8px; padding:10px 12px;
    border-left:4px solid #CBD5E1; background:#F8FAFC;
    font-size:0.82rem; line-height:1.4;
}
.agent-success { border-left-color:#22C55E; background:#F0FDF4; }
.agent-running  { border-left-color:#F59E0B; background:#FFFBEB; }
.agent-skipped  { border-left-color:#94A3B8; background:#F8FAFC; }
.agent-error    { border-left-color:#EF4444; background:#FEF2F2; }
.agent-pending  { border-left-color:#CBD5E1; background:#F8FAFC; }
.urgency-critical { color:#DC2626; font-weight:600; }
.urgency-high     { color:#D97706; font-weight:600; }
.urgency-medium   { color:#2563EB; }
.urgency-low      { color:#16A34A; }
.followup { background:#EFF6FF; border-left:4px solid #3B82F6;
            padding:9px 13px; border-radius:4px; margin:4px 0;
            font-size:0.88rem; }
.brief-box { background:#F8FAFC; border:1px solid #E2E8F0;
             padding:18px 22px; border-radius:8px; }
.chat-user      { background:#EFF6FF; border-radius:8px; padding:8px 12px;
                  margin:4px 0; font-size:0.9rem; }
.chat-assistant { background:#F0FDF4; border-radius:8px; padding:8px 12px;
                  margin:4px 0; font-size:0.9rem; }
.debate-challenge { background:#FFF7ED; border-left:3px solid #F59E0B;
                    padding:8px 12px; border-radius:4px; margin:4px 0;
                    font-size:0.85rem; }
.red-flag { background:#FEF2F2; border-left:3px solid #EF4444;
            padding:8px 12px; border-radius:4px; margin:4px 0;
            font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─── Module instances ────────────────────────────────────────────────────────

@st.cache_resource
def get_engine():      return IngestionEngine()
@st.cache_resource
def get_org_memory():  return OrgMemory()
@st.cache_resource
def get_ctx_engine():  return ContextEngine(get_org_memory())

ingestion_engine = get_engine()
org_memory       = get_org_memory()
ctx_engine       = get_ctx_engine()

# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 AI Analyst")
    st.caption("v0.4 — Autonomous Mode")
    st.divider()

    st.markdown("**Overrides** *(optional)*")
    manual_date = st.text_input("Date column", placeholder="auto-detect")
    manual_kpi  = st.text_input("KPI column",  placeholder="auto-detect")
    grain       = st.radio("Time grain", TIME_GRAINS, horizontal=True)

    st.divider()
    st.markdown("**LLM**")
    api_ok = bool(config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY)
    if api_ok:
        st.success(f"{config.LLM_PROVIDER} — {config.LLM_MODEL} ✓")
    else:
        st.warning("No API key — rule-based fallback active")

    st.divider()
    st.markdown("**Org memory**")
    ctx_all = org_memory.get_all_context()
    if ctx_all:
        for k, v in list(ctx_all.items())[:4]:
            st.caption(f"{k}: {str(v)[:30]}")
    else:
        st.caption("No org context yet — answer upfront questions to build it.")
    if st.button("Clear org memory", use_container_width=True):
        org_memory.clear()
        st.success("Memory cleared.")

    st.divider()
    st.markdown("**Supported formats**")
    st.caption("CSV · Excel · JSON · PDF · Word · Images · SQL · Text · Streams")

# ─── Header ─────────────────────────────────────────────────────────────────

st.title("🧠 Autonomous AI Analyst")
st.caption("Drop any data. The system asks what it needs, analyses everything, and tells you what matters.")
st.divider()

# ─── Session state init ──────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "stage": "upload",           # upload → questions → running → results
        "documents": [],
        "context": None,
        "questions": [],
        "answers": [],
        "output_decision": None,
        "conversation": None,
        "agent_statuses": {},
        "agent_results": {},
        "chat_input_key": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── STAGE 1: UPLOAD ────────────────────────────────────────────────────────

if st.session_state.stage == "upload":
    st.subheader("Step 1 — Upload your data")

    col_upload, col_info = st.columns([2, 1])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Drop any file(s) here",
            type=None,
            accept_multiple_files=True,
            help="CSV, Excel, JSON, PDF, Word, images, plain text — all supported",
        )

    with col_info:
        st.markdown("**What can I upload?**")
        st.markdown("""
- CSV, TSV, Excel (all sheets)
- JSON, JSONL
- PDF (text + tables extracted)
- Word documents (.docx)
- Images (charts, screenshots, scans)
- Plain text, Markdown
- Multiple files at once
        """)

    if uploaded_files:
        docs = []
        with st.spinner("Parsing files..."):
            for f in uploaded_files:
                doc = ingestion_engine.ingest(f, filename=f.name)
                docs.append(doc)
                if doc.warnings:
                    for w in doc.warnings:
                        st.warning(f"{f.name}: {w}")

        st.session_state.documents = docs

        # Show parsed summary
        total_rows = sum(
            sum(len(d) for d in doc.dataframes)
            for doc in docs
        )
        total_chunks = sum(len(doc.text_chunks) for doc in docs)
        total_images = sum(len(doc.image_descriptions) for doc in docs)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Files parsed", len(docs))
        col2.metric("Data rows", f"{total_rows:,}")
        col3.metric("Text chunks", total_chunks)
        col4.metric("Images analysed", total_images)

        if st.button("Continue to context questions →", type="primary", use_container_width=True):
            # Merge all dataframes into primary df
            all_dfs = []
            for doc in docs:
                all_dfs.extend(doc.dataframes)
            primary_df = max(all_dfs, key=len) if all_dfs else pd.DataFrame()

            # Build a temporary context for EDA profiling
            from analysis.eda_engine import EDAEngine
            from connectors.csv_connector import CSVConnector
            eda = EDAEngine()
            profile = {}
            if not primary_df.empty:
                profile = eda.quality_report(primary_df)
                profile["kpis"] = eda.infer_kpis(primary_df)
                profile["dimensions"] = eda.infer_dimensions(primary_df)
                date_col = CSVConnector.detect_datetime_column(primary_df)
                profile["date_col"] = date_col
                profile["has_time_series"] = date_col is not None
                profile["has_funnel_signal"] = any(
                    any(kw in str(c).lower()
                        for kw in ["stage","event","step","funnel","status"])
                    for c in primary_df.columns
                )
                profile["has_cohort_signal"] = any(
                    any(kw in str(c).lower() for kw in ["user","customer","member"])
                    for c in primary_df.columns
                )

            doc_summary = " | ".join(doc.summary() for doc in docs)
            questions = ctx_engine.generate_questions(profile, doc_summary)
            st.session_state.questions = questions
            st.session_state["_primary_df"] = primary_df
            st.session_state["_profile"] = profile
            st.session_state.stage = "questions"
            st.rerun()

# ─── STAGE 2: UPFRONT QUESTIONS ─────────────────────────────────────────────

elif st.session_state.stage == "questions":
    st.subheader("Step 2 — Business context")
    st.caption("These questions help calibrate the analysis to your business. Skip any that don't apply.")

    questions = st.session_state.questions

    if not questions:
        st.success("Org memory already has sufficient context — skipping questions.")
        if st.button("Run analysis →", type="primary"):
            st.session_state.answers = []
            st.session_state.stage = "running"
            st.rerun()
    else:
        with st.form("context_form"):
            answers = []
            for i, q in enumerate(questions):
                ans = st.text_input(f"Q{i+1}: {q}", key=f"q_{i}")
                answers.append(ans)

            col_skip, col_run = st.columns([1, 2])
            with col_skip:
                skip = st.form_submit_button("Skip all questions")
            with col_run:
                submitted = st.form_submit_button("Save context + run analysis →", type="primary")

        if submitted or skip:
            if submitted and any(a.strip() for a in answers):
                enriched = ctx_engine.enrich_from_answers(questions, answers)
                st.session_state["_biz_context"] = enriched
            else:
                st.session_state["_biz_context"] = ctx_engine.load_context()
            st.session_state.answers = answers
            st.session_state.stage = "running"
            st.rerun()

# ─── STAGE 3: RUNNING ───────────────────────────────────────────────────────

elif st.session_state.stage == "running":
    st.subheader("Step 3 — Agent pipeline running")

    docs       = st.session_state.documents
    primary_df = st.session_state.get("_primary_df", pd.DataFrame())
    profile    = st.session_state.get("_profile", {})
    biz_ctx    = st.session_state.get("_biz_context", {})

    # Determine date + KPI columns
    date_col_detected = profile.get("date_col", "")
    kpi_candidates    = profile.get("kpis", [])
    date_col = manual_date if manual_date else date_col_detected
    kpi_col  = manual_kpi  if manual_kpi  else (kpi_candidates[0] if kpi_candidates else "")

    if date_col and not primary_df.empty and date_col in primary_df.columns:
        primary_df[date_col] = pd.to_datetime(primary_df[date_col], errors="coerce")

    # Build AnalysisContext
    context = AnalysisContext(
        df=primary_df,
        date_col=date_col or "",
        kpi_col=kpi_col or "",
        grain=grain,
        filename=docs[0].source_name if docs else "data",
        document=docs[0] if docs else None,
        business_context=biz_ctx,
        data_profile=profile,
    )

    ALL_NAMES = list(AGENT_REGISTRY.keys())
    statuses = {n: "pending" for n in ALL_NAMES}
    results_map = {}

    status_placeholder = st.empty()

    def render_status(statuses, results_map):
        cards_html = '<div class="agent-grid">'
        ICONS = {"pending":"⬜","running":"🟡","success":"🟢","skipped":"⚪","error":"🔴"}
        for name in ALL_NAMES:
            s = statuses.get(name, "pending")
            r = results_map.get(name)
            dur = f" ({r.duration_sec:.1f}s)" if r and r.duration_sec else ""
            snippet = (r.summary[:70] + "...") if r and len(r.summary) > 70 else (r.summary if r else "")
            cards_html += (
                f'<div class="agent-card agent-{s}">'
                f'{ICONS[s]} <strong>{name}</strong>{dur}<br>'
                f'<span style="color:#64748B;font-size:0.78rem">{snippet}</span>'
                f'</div>'
            )
        cards_html += '</div>'
        status_placeholder.markdown(cards_html, unsafe_allow_html=True)

    render_status(statuses, results_map)

    def on_start(name):
        statuses[name] = "running"
        render_status(statuses, results_map)

    def on_done(result: AgentResult):
        statuses[result.agent] = result.status
        results_map[result.agent] = result
        render_status(statuses, results_map)

    runner = AgentRunner(max_workers=6)
    t_start = time.time()
    finished_context = runner.run(context, on_agent_start=on_start, on_agent_done=on_done)
    elapsed = round(time.time() - t_start, 1)

    # Mark not-run as skipped
    for name in ALL_NAMES:
        if name not in finished_context.results:
            statuses[name] = "skipped"
            results_map[name] = AgentResult(
                agent=name, status="skipped",
                summary="Not required for this dataset.", data={}
            )
    render_status(statuses, results_map)

    # Route outputs
    router = OutputRouter()
    decision = router.decide(finished_context)

    # Fire alerts if needed
    if "alert" in decision.modes and decision.alert_channels != ["in_app"]:
        dispatcher = AlertDispatcher()
        alert_results = dispatcher.dispatch(
            decision.alert_channels, decision.alert_message, decision.urgency
        )
        finished_context.business_context["_alert_results"] = alert_results

    # Store everything
    st.session_state.context         = finished_context
    st.session_state.output_decision = decision
    st.session_state.agent_statuses  = statuses
    st.session_state.agent_results   = results_map
    st.session_state.conversation    = ConversationEngine(finished_context)

    st.success(f"Pipeline complete in {elapsed}s — {len(finished_context.results)} agents ran.")
    time.sleep(0.5)
    st.session_state.stage = "results"
    st.rerun()

# ─── STAGE 4: RESULTS ───────────────────────────────────────────────────────

elif st.session_state.stage == "results":
    ctx: AnalysisContext  = st.session_state.context
    decision              = st.session_state.output_decision
    conv_engine           = st.session_state.conversation

    # ── Top bar: urgency + output mode + reset ───────────────────────────
    top_col1, top_col2, top_col3 = st.columns([2, 3, 1])
    with top_col1:
        urgency_class = f"urgency-{decision.urgency}"
        st.markdown(
            f'<span class="{urgency_class}">Urgency: {decision.urgency.upper()}</span>  '
            f'<span style="color:#64748B;font-size:0.85rem">— {decision.reason[:80]}</span>',
            unsafe_allow_html=True,
        )
    with top_col2:
        modes_str = " · ".join(f"**{m}**" for m in decision.modes)
        st.markdown(f"Output modes: {modes_str}")
    with top_col3:
        if st.button("New analysis", use_container_width=True):
            for key in ["stage","documents","context","questions","answers",
                        "output_decision","conversation","agent_statuses",
                        "agent_results","_primary_df","_profile","_biz_context"]:
                st.session_state.pop(key, None)
            st.rerun()

    # Alert banner
    if "alert" in decision.modes:
        st.error(f"⚠️ Alert fired ({decision.urgency.upper()}): {decision.alert_message[:120]}")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📝 Brief",
        "🔍 EDA",
        "📈 Trend + Forecast",
        "⚠️ Anomalies",
        "🔎 Root Cause",
        "🔬 Experiment",
        "👥 Segments",
        "🔤 NLP + Vision",
        "🤔 Debate",
        "🔽 Funnel",
        "👤 Cohort",
        "💬 Chat",
    ])

    # ══════════════════════════════════════════════════════════════════
    # BRIEF TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[0]:
        orch = ctx.results.get("orchestrator")
        if orch:
            plan = orch.data.get("plan", [])
            biz  = ctx.business_context
            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"**Agents activated:** {' → '.join(plan)}")
            with col_b:
                audience = biz.get("audience", "—")
                goal     = biz.get("primary_goal", "—")
                st.info(f"**Audience:** {audience}  |  **Goal:** {goal}")

        if ctx.final_brief:
            st.markdown('<div class="brief-box">', unsafe_allow_html=True)
            st.markdown(ctx.final_brief)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No brief generated — ensure an LLM API key is set.")

        if ctx.follow_up_questions:
            st.markdown("### 💡 Suggested next questions")
            for q in ctx.follow_up_questions:
                if st.button(q, key=f"fq_{q[:20]}", use_container_width=True):
                    conv_engine.chat(q)
                    st.session_state.stage = "results"

        t_total = sum(r.duration_sec for r in ctx.results.values() if r.duration_sec)
        st.caption(f"Total agent time: {t_total:.1f}s  |  Pipeline elapsed: see running log")

    # ══════════════════════════════════════════════════════════════════
    # EDA TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[1]:
        eda = ctx.results.get("eda")
        if eda and eda.status == "success":
            q = eda.data["quality"]
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Rows",         f"{q['total_rows']:,}")
            c2.metric("Columns",      q["total_columns"])
            c3.metric("Completeness", f"{q['completeness_pct']}%")
            c4.metric("Duplicates",   q["duplicate_rows"])
            c5.metric("KPIs found",   len(eda.data["kpis"]))

            ca, cb = st.columns(2)
            with ca:
                st.info(f"**KPIs:** {', '.join(eda.data['kpis']) or 'None'}")
                st.info(f"**Dimensions:** {', '.join(eda.data['dimensions']) or 'None'}")
            with cb:
                st.info(f"**Date col:** {eda.data['date_col'] or 'Not detected'}")
                st.info(
                    f"**Funnel signal:** {'Yes' if eda.data['has_funnel_signal'] else 'No'}  |  "
                    f"**Cohort signal:** {'Yes' if eda.data['has_cohort_signal'] else 'No'}"
                )
            st.dataframe(eda.data["profile_df"], use_container_width=True)

            # Document summary if available
            doc = ctx.document
            if doc and (doc.text_chunks or doc.image_descriptions):
                st.markdown("**Document content parsed**")
                st.info(doc.summary())
        else:
            st.warning("EDA not run.")

    # ══════════════════════════════════════════════════════════════════
    # TREND + FORECAST TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[2]:
        trend = ctx.results.get("trend")
        fcst  = ctx.results.get("forecast")

        if trend and trend.status == "success":
            ts        = trend.data.get("ts", ctx.ts)
            date_col  = ctx.date_col
            kpi_col   = ctx.kpi_col
            comps     = trend.data.get("comparisons", {})

            t1,t2,t3,t4 = st.columns(4)
            t1.metric("Direction",     trend.data["trend_direction"].title())
            t2.metric("Overall trend", f"{trend.data['trend_pct']:+.1f}%")
            t3.metric("Latest value",  f"{trend.data.get('latest_value',0):,.2f}")
            t4.metric("Data points",   trend.data.get("data_points", 0))

            if not ts.empty and date_col in ts.columns and kpi_col in ts.columns:
                ts_for_chart = ts.copy()
                if "anomaly" in ts_for_chart.columns:
                    pass   # keep anomaly column if present
                st.plotly_chart(
                    trend_with_anomalies(ts_for_chart, date_col, kpi_col,
                                         title=f"{kpi_col} — {ctx.grain}"),
                    use_container_width=True,
                )
            if comps:
                cc = st.columns(len(comps))
                for widget, (cname, c) in zip(cc, comps.items()):
                    arrow = "▲" if c["pct_change"] >= 0 else "▼"
                    widget.metric(cname, f"{c['current']:,.0f}",
                                  f"{arrow} {c['pct_change']:+.1f}%")
        else:
            st.info("Trend analysis not run for this dataset.")

        st.divider()
        st.markdown("### Forecast")
        if fcst and fcst.status == "success":
            fc1,fc2,fc3 = st.columns(3)
            fc1.metric("Method",    fcst.data.get("method", "—"))
            fc2.metric("Horizon",   f"{fcst.data.get('horizon',0)} periods")
            fc3.metric("Direction", fcst.data.get("direction","—").title())

            import plotly.graph_objects as go
            fdf = fcst.data.get("forecast_df")
            if fdf is not None and not fdf.empty:
                fig = go.Figure()
                ts_plot = trend.data.get("ts", ctx.ts) if trend and trend.status == "success" else ctx.ts
                if not ts_plot.empty and ctx.date_col in ts_plot.columns:
                    fig.add_trace(go.Scatter(
                        x=ts_plot[ctx.date_col], y=ts_plot[ctx.kpi_col],
                        name="Actual", line=dict(color="#3B82F6", width=2),
                    ))
                fig.add_trace(go.Scatter(
                    x=fdf["ds"], y=fdf["yhat"],
                    name="Forecast", line=dict(color="#F59E0B", width=2, dash="dash"),
                ))
                if "yhat_lower" in fdf.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fdf["ds"], fdf["ds"][::-1]]),
                        y=pd.concat([fdf["yhat_upper"], fdf["yhat_lower"][::-1]]),
                        fill="toself", fillcolor="rgba(245,158,11,0.15)",
                        line=dict(color="rgba(0,0,0,0)"), name="80% CI",
                    ))
                fig.update_layout(height=380, hovermode="x unified",
                                  legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Forecasting not run — needs a time series with ≥10 data points.")

    # ══════════════════════════════════════════════════════════════════
    # ANOMALY TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[3]:
        anom = ctx.results.get("anomaly")
        if anom and anom.status == "success":
            ts_a     = anom.data.get("ts_with_anomalies", ctx.ts)
            date_col = ctx.date_col
            kpi_col  = ctx.kpi_col
            sev      = anom.data.get("severity_counts", {})

            a1,a2,a3,a4 = st.columns(4)
            a1.metric("Method",    anom.data.get("method_used","—"))
            a2.metric("Anomalies", anom.data.get("anomaly_count",0))
            a3.metric("High sev.", sev.get("high",0))
            a4.metric("Medium",    sev.get("medium",0))

            if not ts_a.empty and date_col in ts_a.columns and kpi_col in ts_a.columns:
                st.plotly_chart(
                    trend_with_anomalies(ts_a, date_col, kpi_col,
                                         title=f"{kpi_col} — {anom.data.get('method_used')} Detection"),
                    use_container_width=True,
                )
            records = anom.data.get("anomaly_records", [])
            if records:
                st.markdown("**Flagged points**")
                st.dataframe(pd.DataFrame(records), use_container_width=True)
        else:
            st.info("Anomaly detection not run.")

    # ══════════════════════════════════════════════════════════════════
    # ROOT CAUSE TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[4]:
        rc = ctx.results.get("root_cause")
        if rc and rc.status == "success":
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Current period",  f"{rc.data['last_total']:,.0f}")
            r2.metric("Prior period",    f"{rc.data['prev_total']:,.0f}")
            r3.metric("Delta",           f"{rc.data['delta']:+,.0f}")
            r4.metric("% Change",        f"{rc.data['pct_change']:+.1f}%")

            drivers = rc.data.get("drivers", pd.DataFrame())
            if not drivers.empty:
                st.plotly_chart(
                    driver_bar_chart(drivers, title=f"Top Drivers — {ctx.kpi_col}"),
                    use_container_width=True,
                )
                movers = rc.data.get("movers", {})
                nc, pc = st.columns(2)
                with nc:
                    st.markdown("**Top negative drivers**")
                    neg = pd.DataFrame(movers.get("negative", []))
                    if not neg.empty: st.dataframe(neg, use_container_width=True)
                with pc:
                    st.markdown("**Top positive drivers**")
                    pos = pd.DataFrame(movers.get("positive", []))
                    if not pos.empty: st.dataframe(pos, use_container_width=True)

            contribs = rc.data.get("contributions", {})
            if contribs:
                st.markdown("**Contribution by dimension**")
                for dim, cdf in contribs.items():
                    with st.expander(dim):
                        if not cdf.empty:
                            st.plotly_chart(
                                contribution_bar(cdf, dim, f"{ctx.kpi_col} share by {dim}"),
                                use_container_width=True,
                            )
        else:
            st.info("Root cause analysis not run.")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[5]:
        exp = ctx.results.get("experiment")
        if exp and exp.status == "success":
            tt = exp.data.get("results", {}).get("ttest", {})
            sig_label = "✅ Significant" if exp.data.get("significant") else "❌ Not significant"

            e1,e2,e3,e4 = st.columns(4)
            e1.metric("Result",     sig_label)
            e2.metric("Lift",       f"{exp.data.get('lift_pct',0):+.1f}%")
            e3.metric("p-value",    f"{exp.data.get('p_value',1):.4f}")
            e4.metric("Cohen's d",  f"{exp.data.get('cohens_d',0):.3f}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Group A ({tt.get('group_a','A')}):** mean = {tt.get('mean_a',0):,.4f}  (n={tt.get('n_a',0)})")
                st.markdown(f"**Group B ({tt.get('group_b','B')}):** mean = {tt.get('mean_b',0):,.4f}  (n={tt.get('n_b',0)})")
                st.markdown(f"**MDE:** ±{exp.data.get('results',{}).get('mde_pct',0):.2f}%")

            did = exp.data.get("results", {}).get("diff_in_diff")
            if did:
                with col_b:
                    st.markdown("**Diff-in-diff**")
                    st.dataframe(pd.DataFrame([did]), use_container_width=True)
        else:
            st.info("No A/B test or experiment column detected in this dataset.")

    # ══════════════════════════════════════════════════════════════════
    # SEGMENTS TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[6]:
        clst = ctx.results.get("ml_cluster")
        if clst and clst.status == "success":
            s1,s2,s3 = st.columns(3)
            s1.metric("Clusters",   clst.data.get("n_clusters",0))
            s2.metric("Silhouette", f"{clst.data.get('silhouette_score',0):.3f}")
            s3.metric("Features",   len(clst.data.get("feature_cols",[])))

            names = clst.data.get("cluster_names", [])
            profile_df = clst.data.get("profile_df")
            if profile_df is not None and not profile_df.empty:
                if names and len(names) == len(profile_df):
                    profile_df = profile_df.copy()
                    profile_df.insert(0, "name", names)
                st.markdown("**Cluster profiles**")
                st.dataframe(profile_df, use_container_width=True)

            umap_df = clst.data.get("umap_df")
            if umap_df is not None and not umap_df.empty:
                import plotly.express as px
                fig = px.scatter(
                    umap_df, x="x", y="y", color=umap_df["cluster"].astype(str),
                    title="UMAP 2D projection",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Clustering not run — needs multiple numeric columns and ≥20 rows.")

    # ══════════════════════════════════════════════════════════════════
    # NLP + VISION TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[7]:
        nlp_r    = ctx.results.get("nlp")
        vision_r = ctx.results.get("vision")

        if nlp_r and nlp_r.status == "success":
            st.markdown("### Text analysis")
            findings = nlp_r.data.get("findings", {})
            sent     = nlp_r.data.get("sentiment", {})
            kw       = nlp_r.data.get("top_keywords", [])

            if sent:
                n1,n2,n3 = st.columns(3)
                n1.metric("Positive", f"{sent.get('positive',0):.0%}")
                n2.metric("Negative", f"{sent.get('negative',0):.0%}")
                n3.metric("Neutral",  f"{sent.get('neutral',0):.0%}")

            if kw:
                st.markdown(f"**Top keywords:** {', '.join(kw[:15])}")

            llm_analysis = nlp_r.data.get("llm_analysis", "")
            if llm_analysis:
                with st.expander("LLM deep analysis"):
                    st.markdown(llm_analysis)

        if vision_r and vision_r.status == "success":
            st.markdown("### Visual content analysis")
            n_img  = vision_r.data.get("n_images", 0)
            types  = vision_r.data.get("image_types", [])
            nums   = vision_r.data.get("numbers_found", [])

            st.info(f"{n_img} image(s): {', '.join(set(types)) if types else 'unknown type'}")

            if nums:
                st.markdown("**Values extracted from images:**")
                st.dataframe(pd.DataFrame(nums), use_container_width=True)

            interp = vision_r.data.get("interpretation", "")
            if interp:
                with st.expander("Full image interpretation"):
                    st.markdown(interp)

            tbl = vision_r.data.get("table_from_vision")
            if tbl is not None and not tbl.empty:
                st.markdown("**Table extracted from image:**")
                st.dataframe(tbl, use_container_width=True)

        if not (nlp_r and nlp_r.status == "success") and not (vision_r and vision_r.status == "success"):
            st.info("NLP and Vision agents not activated for this dataset.")

    # ══════════════════════════════════════════════════════════════════
    # DEBATE TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[8]:
        dbte = ctx.results.get("debate")
        if dbte and dbte.status == "success":
            verdict = dbte.data.get("verdict", "medium")
            reason  = dbte.data.get("verdict_reason", "")
            flags   = dbte.data.get("red_flags", [])

            verdict_colors = {"high":"green","medium":"orange","low":"red"}
            st.markdown(
                f"**Overall narrative confidence:** "
                f"<span style='color:{verdict_colors.get(verdict,'gray')};font-weight:600'>"
                f"{verdict.upper()}</span>",
                unsafe_allow_html=True,
            )
            if reason:
                st.caption(reason)

            if flags:
                st.markdown("**Red flags**")
                for f in flags:
                    st.markdown(f'<div class="red-flag">⛳ {f}</div>', unsafe_allow_html=True)

            challenges = dbte.data.get("challenges", [])
            if challenges:
                st.markdown("**Challenges to findings**")
                for ch in challenges:
                    conf_col = {"high":"#16A34A","medium":"#D97706","low":"#DC2626"}
                    conf = ch.get("confidence_in_finding","medium")
                    html = (
                        f'<div class="debate-challenge">'
                        f'<strong>Finding:</strong> {ch.get("finding","")[:80]}<br>'
                        f'<strong>Challenge:</strong> {ch.get("challenge","")}<br>'
                        f'<strong>Alternative:</strong> {ch.get("alternative_explanation","")}<br>'
                        f'<strong style="color:{conf_col.get(conf,"#888")}">Confidence: {conf}</strong>'
                    )
                    if ch.get("data_quality_flag"):
                        html += f'<br><em style="color:#64748B">{ch["data_quality_flag"]}</em>'
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Debate review not run.")

    # ══════════════════════════════════════════════════════════════════
    # FUNNEL TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[9]:
        fn = ctx.results.get("funnel")
        if fn and fn.status == "success":
            f1,f2,f3 = st.columns(3)
            f1.metric("Stages",      len(fn.data.get("stages",[])))
            f2.metric("Top funnel",  f"{fn.data.get('top_of_funnel_users',0):,}")
            f3.metric("Conversion",  f"{fn.data.get('overall_conversion_pct',0):.1f}%")

            bd = fn.data.get("biggest_drop",{})
            if bd:
                st.warning(
                    f"📉 Biggest drop at **{bd['stage']}** — "
                    f"{bd['drop_off_pct']:.1f}% drop-off "
                    f"({bd.get('users_lost',0):,} users lost)"
                )
            fdf = fn.data.get("funnel_df", pd.DataFrame())
            if not fdf.empty:
                st.plotly_chart(funnel_chart(fdf), use_container_width=True)
                st.dataframe(fdf, use_container_width=True)
        else:
            st.info("Funnel analysis not activated — no stage/event column detected.")

    # ══════════════════════════════════════════════════════════════════
    # COHORT TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[10]:
        coh = ctx.results.get("cohort")
        if coh and coh.status == "success":
            c1,c2,c3 = st.columns(3)
            c1.metric("Unique users",  f"{coh.data.get('n_users',0):,}")
            c2.metric("Date range",    f"{coh.data.get('date_range_days',0)} days")
            c3.metric("Grain",         coh.data.get("grain","—"))

            matrix = coh.data.get("retention_matrix")
            if matrix is not None and not matrix.empty:
                st.plotly_chart(cohort_heatmap(matrix), use_container_width=True)
                st.dataframe(
                    matrix.style.format("{:.1f}%", na_rep="—")
                           .background_gradient(cmap="Blues", axis=None),
                    use_container_width=True,
                )
            else:
                st.info("Retention matrix could not be computed.")
        else:
            st.info("Cohort analysis not activated.")

    # ══════════════════════════════════════════════════════════════════
    # CHAT TAB
    # ══════════════════════════════════════════════════════════════════
    with tabs[11]:
        st.markdown("### Ask anything about this analysis")
        st.caption("Try: 'Why did it drop?', 'Show me the anomalies', 'What is the forecast?', 'How confident are you?'")

        # Display history
        for turn in conv_engine.history:
            if turn.role == "user":
                st.markdown(f'<div class="chat-user">🧑 {turn.content}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-assistant">🤖 {turn.content}</div>',
                            unsafe_allow_html=True)

        # Input
        with st.form(key=f"chat_form_{st.session_state.chat_input_key}",
                     clear_on_submit=True):
            user_msg = st.text_input(
                "Your question",
                placeholder="What caused the drop last week?",
                label_visibility="collapsed",
            )
            send = st.form_submit_button("Send", use_container_width=True)

        if send and user_msg.strip():
            with st.spinner("Thinking..."):
                response = conv_engine.chat(user_msg.strip())
            st.session_state.chat_input_key += 1
            st.rerun()

        if st.button("Clear chat history", use_container_width=True):
            conv_engine.reset()
            st.rerun()
