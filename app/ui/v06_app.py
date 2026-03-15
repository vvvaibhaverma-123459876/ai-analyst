"""
app/ui/v06_app.py — v0.6
AI Analyst — Theoretical Layer Edition.

New panels vs v0.5:
  - Connector hub (Postgres / Snowflake / BigQuery / Redshift / Athena / CSV)
  - Semantic memory panel (vector-powered prior insight retrieval)
  - Experiment designer panel (auto-spec from inconclusive hypotheses)
  - Schedule monitor panel (live scheduler job status + manual trigger)
  - dbt sync status (MetricRegistry populated from dbt manifest if present)
  - Multi-user session bar (who is logged in, team annotations)
  - Audit trail export (admin: download NDJSON / CSV)

Run: streamlit run app/ui/v06_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import uuid
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
from security.security_shell import SecurityShell
from security.audit_logger import AuditLogger
from security.policy_store import PolicyStore
from ground_truth.recorder import GroundTruthRecorder, Finding, Outcome
from charts.chart_builder import (
    trend_with_anomalies, driver_bar_chart, funnel_chart,
    cohort_heatmap, contribution_bar, kpi_comparison_bar,
)
from core.config import config
from core.constants import TIME_GRAINS

st.set_page_config(
    page_title="AI Analyst v0.6",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
.security-bar {
    background:#F0FDF4; border:1px solid #86EFAC;
    border-radius:6px; padding:6px 14px; font-size:.82rem; margin-bottom:8px;
    display:flex; gap:20px; align-items:center;
}
.security-bar.warn { background:#FFFBEB; border-color:#FCD34D; }
.security-bar.off  { background:#F8FAFC; border-color:#CBD5E1; }
.session-bar {
    background:#EFF6FF; border:1px solid #93C5FD;
    border-radius:6px; padding:5px 14px; font-size:.80rem;
    margin-bottom:8px; display:flex; gap:16px; align-items:center;
}
.connector-card {
    border-radius:8px; padding:10px 14px; margin:4px 0;
    font-size:.83rem; line-height:1.5;
    border-left:4px solid #CBD5E1; background:#F8FAFC;
}
.connector-ok   { border-left-color:#22C55E; background:#F0FDF4; }
.connector-err  { border-left-color:#EF4444; background:#FEF2F2; }
.connector-na   { border-left-color:#94A3B8; }
.agent-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:7px; margin:8px 0; }
.agent-card {
    border-radius:7px; padding:9px 11px;
    border-left:4px solid #CBD5E1; background:#F8FAFC;
    font-size:.80rem; line-height:1.4;
}
.agent-success  { border-left-color:#22C55E; background:#F0FDF4; }
.agent-running  { border-left-color:#F59E0B; background:#FFFBEB; }
.agent-skipped  { border-left-color:#94A3B8; }
.agent-error    { border-left-color:#EF4444; background:#FEF2F2; }
.hypothesis-card {
    border-radius:7px; padding:10px 14px; margin:5px 0;
    font-size:.85rem; line-height:1.5;
}
.h-testable     { background:#EFF6FF; border-left:4px solid #3B82F6; border-radius:0 7px 7px 0; }
.h-confirmed    { background:#F0FDF4; border-left:4px solid #22C55E; border-radius:0 7px 7px 0; }
.h-rejected     { background:#FEF2F2; border-left:4px solid #EF4444; border-radius:0 7px 7px 0; }
.h-inconclusive { background:#FFFBEB; border-left:4px solid #F59E0B; border-radius:0 7px 7px 0; }
.exp-card {
    background:#F8FAFC; border:1px solid #E2E8F0;
    border-radius:8px; padding:14px 18px; margin:8px 0;
    font-size:.84rem; line-height:1.6;
}
.exp-card h4 { margin:0 0 8px; font-size:.92rem; color:#1E40AF; }
.mem-hit {
    background:#F0FDF4; border-left:3px solid #22C55E;
    border-radius:0 6px 6px 0; padding:7px 12px; margin:4px 0;
    font-size:.82rem; line-height:1.45;
}
.sched-card {
    background:#F8FAFC; border:1px solid #E2E8F0;
    border-radius:7px; padding:9px 13px; margin:4px 0;
    font-size:.82rem;
}
.sched-ok   { border-left:3px solid #22C55E; }
.sched-err  { border-left:3px solid #EF4444; }
.sched-pend { border-left:3px solid #94A3B8; }
.brief-box  { background:#F8FAFC; border:1px solid #E2E8F0; padding:18px 22px; border-radius:8px; }
.chat-user      { background:#EFF6FF; border-radius:8px; padding:8px 12px; margin:4px 0; }
.chat-assistant { background:#F0FDF4; border-radius:8px; padding:8px 12px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Cached resources ─────────────────────────────────────────────────

@st.cache_resource
def get_ingestion():   return IngestionEngine()
@st.cache_resource
def get_org_memory():  return OrgMemory()
@st.cache_resource
def get_ctx_engine():  return ContextEngine(get_org_memory())
@st.cache_resource
def get_gt_recorder(): return GroundTruthRecorder()

# ── Session (v0.6: multi-user identity) ─────────────────────────────

def get_user_session():
    try:
        from api.session_manager import streamlit_session
        return streamlit_session(
            tenant_id=st.session_state.get("tenant_id", "default"),
            role=st.session_state.get("user_role", "analyst"),
            display_name=st.session_state.get("display_name", ""),
        )
    except Exception:
        return None

# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔬 AI Analyst v0.6")
    st.caption("Theoretical layer — all gaps bridged")

    # User identity
    st.markdown("---")
    st.markdown("**Session**")
    display_name = st.text_input("Display name", value=st.session_state.get("display_name",""), key="display_name")
    role = st.selectbox("Role", ["analyst","viewer","admin"], key="user_role")
    tenant_id = st.text_input("Tenant", value=st.session_state.get("tenant_id","default"), key="tenant_id")

    # Connector hub
    st.markdown("---")
    st.markdown("**Connector Hub**")
    connector_type = st.selectbox(
        "Active connector",
        ["csv_upload", "postgres", "snowflake", "bigquery", "redshift", "athena"],
        key="connector_type",
    )

    st.markdown("---")
    st.markdown("**Analysis config**")
    grain = st.selectbox("Time grain", TIME_GRAINS, index=0, key="grain")

    # dbt sync
    st.markdown("---")
    st.markdown("**dbt / Metrics**")
    dbt_manifest = st.text_input("dbt manifest.json path (optional)", key="dbt_manifest")
    if st.button("Sync dbt metrics"):
        try:
            from semantic.dbt_adapter import DbtAdapter
            adapter = DbtAdapter(manifest_path=dbt_manifest or None)
            metrics = adapter.load()
            if metrics:
                st.success(f"✓ {len(metrics)} metrics synced from dbt")
            else:
                st.info("No dbt manifest found — using configs/metrics.yaml")
        except Exception as e:
            st.error(f"dbt sync failed: {e}")

    # Scheduler status
    st.markdown("---")
    st.markdown("**Schedule Monitor**")
    if st.button("Refresh scheduler status"):
        st.session_state["refresh_sched"] = True

# ── Main layout ──────────────────────────────────────────────────────

# Session bar
user_session = get_user_session()
if user_session:
    st.markdown(
        f'<div class="session-bar">'
        f'<span>👤 <b>{user_session.display_name}</b></span>'
        f'<span>Role: {user_session.role}</span>'
        f'<span>Tenant: {user_session.tenant_id}</span>'
        f'<span>Session: {user_session.session_id[:8]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Security bar
try:
    shell = SecurityShell()
    policy = PolicyStore()
    pii_on = policy.get("pii_masking_enabled", True)
    internet_off = policy.get("internet_off_mode", False)
    bar_class = "security-bar" if pii_on else "security-bar warn"
    st.markdown(
        f'<div class="{bar_class}">'
        f'<span>🔒 PII masking: {"ON" if pii_on else "OFF"}</span>'
        f'<span>🌐 Internet: {"OFF (local)" if internet_off else "ON"}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
except Exception:
    shell = None

# ── Tabs ─────────────────────────────────────────────────────────────

tab_analysis, tab_connectors, tab_memory, tab_schedule, tab_audit = st.tabs([
    "🔬 Analysis",
    "🔌 Connectors",
    "🧠 Semantic Memory",
    "⏱ Scheduler",
    "📋 Audit",
])

# ────────────────────────────────────────────────────────────────────
# TAB 1 — ANALYSIS
# ────────────────────────────────────────────────────────────────────

with tab_analysis:
    st.subheader("Analysis")

    df_loaded: pd.DataFrame | None = None
    connector_type_val = st.session_state.get("connector_type", "csv_upload")

    if connector_type_val == "csv_upload":
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_file")
        if uploaded:
            df_loaded = pd.read_csv(uploaded)
            st.success(f"✓ {len(df_loaded):,} rows × {len(df_loaded.columns)} cols")
    else:
        st.info(f"Active connector: **{connector_type_val}**  — configure env vars in `.env`, then click Run Analysis.")
        if st.button("Test connection"):
            try:
                from connectors.registry import ConnectorRegistry
                registry = ConnectorRegistry()
                health = registry.health_check()
                for name, ok in health.items():
                    st.write(f"{'✓' if ok else '✗'} {name}")
            except Exception as e:
                st.error(f"Connection test failed: {e}")

        sql_query = st.text_area("SQL query", value="SELECT * FROM your_table LIMIT 10000", key="sql_query")
        if st.button("Load from connector"):
            try:
                from connectors.registry import ConnectorRegistry
                registry = ConnectorRegistry()
                if connector_type_val in registry.available():
                    registry.set_active(connector_type_val)
                df_loaded = registry.execute(sql_query)
                st.success(f"✓ {len(df_loaded):,} rows loaded")
            except Exception as e:
                st.error(f"Query failed: {e}")

    if df_loaded is not None:
        st.dataframe(df_loaded.head(5), use_container_width=True)

        cols = list(df_loaded.columns)
        col1, col2, col3 = st.columns(3)
        with col1:
            kpi_col  = st.selectbox("KPI column", cols, key="kpi_col")
        with col2:
            date_col = st.selectbox("Date column", ["(none)"] + cols, key="date_col")
        with col3:
            grain_val = st.session_state.get("grain", "Daily")

        # Business context
        with st.expander("Business context (optional — enriches hypotheses)"):
            biz_company  = st.text_input("Company name", key="biz_company")
            biz_industry = st.text_input("Industry", key="biz_industry")
            biz_goal     = st.text_input("Primary goal", key="biz_goal")

        if st.button("🚀 Run Analysis", type="primary"):
            context = AnalysisContext(
                df=df_loaded,
                kpi_col=kpi_col,
                date_col="" if date_col == "(none)" else date_col,
                grain=grain_val,
                filename="uploaded",
                run_id=str(uuid.uuid4()),
                tenant_id=st.session_state.get("tenant_id", "default"),
                user_id=user_session.user_id if user_session else "anonymous",
                business_context={
                    "company":       st.session_state.get("biz_company", ""),
                    "industry":      st.session_state.get("biz_industry", ""),
                    "primary_goal":  st.session_state.get("biz_goal", ""),
                },
                security_shell=shell,
            )

            # Agent status grid
            st.markdown("**Agent pipeline**")
            agent_status: dict[str, str] = {}
            status_placeholder = st.empty()

            def render_agent_grid():
                html_parts = ['<div class="agent-grid">']
                for name, st_val in agent_status.items():
                    css = {"running": "agent-running", "success": "agent-success",
                           "error": "agent-error", "skipped": "agent-skipped"}.get(st_val, "")
                    icon = {"running": "⏳", "success": "✓", "error": "✗", "skipped": "—"}.get(st_val, "")
                    html_parts.append(
                        f'<div class="agent-card {css}">{icon} {name}</div>'
                    )
                html_parts.append("</div>")
                status_placeholder.markdown("".join(html_parts), unsafe_allow_html=True)

            def on_start(name):
                agent_status[name] = "running"
                render_agent_grid()

            def on_done(result: AgentResult):
                agent_status[result.agent] = result.status
                render_agent_grid()

            with st.spinner("Running pipeline…"):
                runner = AgentRunner()
                context = runner.run(context, on_agent_start=on_start, on_agent_done=on_done)

            st.session_state["last_context"] = context
            st.success(f"✓ Analysis complete in {context.elapsed():.1f}s")

    # Results panels (shown after analysis runs)
    ctx: AnalysisContext | None = st.session_state.get("last_context")
    if ctx:

        # Brief
        if ctx.final_brief:
            st.markdown("---")
            st.markdown("**Executive brief**")
            st.markdown(f'<div class="brief-box">{ctx.final_brief}</div>', unsafe_allow_html=True)

        # Hypothesis + conclusion panel
        plan = ctx.research_plan
        if plan and plan.hypotheses:
            st.markdown("---")
            with st.expander("🔬 Scientific reasoning — hypotheses & verdicts", expanded=True):
                st.caption(f"{len(plan.hypotheses)} hypotheses | primary: {plan.primary_conclusion[:80] if plan.primary_conclusion else '—'}")
                for h in plan.hypotheses:
                    css = {
                        "confirmed": "h-confirmed", "rejected": "h-rejected",
                        "inconclusive": "h-inconclusive",
                    }.get(str(h.status).lower(), "h-testable")
                    verdict_text = h.verdict or str(h.status).upper()
                    st.markdown(
                        f'<div class="hypothesis-card {css}">'
                        f'<b>{verdict_text}</b> (conf={h.confidence:.0%})'
                        f' <span style="font-size:.78rem;color:#64748B">[{h.source}]</span><br>'
                        f'{h.statement}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # v0.6: Experiment designer panel
            if plan.experiment_spec:
                st.markdown("---")
                with st.expander("🧪 Experiment design (auto-proposed)", expanded=False):
                    spec = plan.experiment_spec
                    st.markdown(f'<div class="exp-card"><h4>Proposed experiment — {spec.spec_id}</h4>'
                                f'<b>Hypothesis:</b> {spec.hypothesis[:100]}<br>'
                                f'<b>Metric:</b> {spec.metric} &nbsp;|&nbsp; '
                                f'<b>Expected lift:</b> {spec.expected_lift_pct:.1f}%<br>'
                                f'<b>Required sample/variant:</b> {spec.required_sample_per_variant:,}<br>'
                                f'<b>Estimated duration:</b> {spec.estimated_duration_days} days '
                                f'(at {spec.traffic_per_day:,}/day)<br>'
                                f'<b>α</b>={spec.alpha} &nbsp; <b>Power</b>={spec.power}'
                                f'</div>', unsafe_allow_html=True)
                    if spec.design_notes:
                        st.markdown("**Design notes**")
                        st.markdown(spec.design_notes)

        # Charts
        trend_result = ctx.results.get("trend")
        if trend_result and trend_result.status == "success" and not ctx.ts.empty:
            st.markdown("---")
            st.markdown("**Trend**")
            anomaly_dates = []
            anom = ctx.results.get("anomaly")
            if anom and anom.status == "success":
                anomaly_dates = [r.get("date") for r in anom.data.get("anomaly_records", [])]
            fig = trend_with_anomalies(ctx.ts, ctx.kpi_col, ctx.date_col, anomaly_dates)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        rc = ctx.results.get("root_cause")
        if rc and rc.status == "success":
            st.markdown("**Root cause — top drivers**")
            fig = driver_bar_chart(rc.data.get("movers", {}), ctx.kpi_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Team annotation panel (v0.6)
        st.markdown("---")
        with st.expander("🗒 Team annotations", expanded=False):
            st.caption("Annotate findings for the whole team to see.")
            ann_key = st.selectbox("Finding to annotate", ["overall"] + list(ctx.results.keys()), key="ann_key")
            ann_verdict = st.selectbox("Verdict", ["correct","incorrect","disputed","note"], key="ann_verdict")
            ann_comment = st.text_input("Comment", key="ann_comment")
            if st.button("Save annotation") and user_session:
                try:
                    from api.session_manager import SessionStore
                    store = SessionStore()
                    store.annotate(
                        run_id=ctx.run_id,
                        finding_key=ann_key,
                        user_id=user_session.user_id,
                        tenant_id=user_session.tenant_id,
                        verdict=ann_verdict,
                        comment=ann_comment,
                    )
                    st.success("Annotation saved and pushed to ground truth.")
                except Exception as e:
                    st.error(f"Annotation failed: {e}")

        # Conversation
        st.markdown("---")
        st.markdown("**Follow-up questions**")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        for msg in st.session_state["chat_history"]:
            css = "chat-user" if msg["role"] == "user" else "chat-assistant"
            st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)
        user_q = st.text_input("Ask a follow-up…", key="follow_up_input")
        if user_q and st.button("Send"):
            st.session_state["chat_history"].append({"role":"user","content":user_q})
            try:
                engine = ConversationEngine()
                answer = engine.answer(user_q, ctx)
            except Exception:
                answer = "Follow-up engine unavailable."
            st.session_state["chat_history"].append({"role":"assistant","content":answer})
            st.rerun()

# ────────────────────────────────────────────────────────────────────
# TAB 2 — CONNECTORS
# ────────────────────────────────────────────────────────────────────

with tab_connectors:
    st.subheader("Connector Hub")
    st.caption("Configure data sources. The registry auto-probes based on env vars in `.env`.")

    if st.button("🔍 Probe all connectors"):
        try:
            from connectors.registry import ConnectorRegistry
            registry = ConnectorRegistry()
            health = registry.health_check()
            active = registry.active()
            for name in ["postgres","snowflake","bigquery","redshift","athena"]:
                if name in health:
                    ok = health[name]
                    css = "connector-ok" if ok else "connector-err"
                    icon = "✓" if ok else "✗"
                    badge = " (active)" if name == active else ""
                    st.markdown(
                        f'<div class="connector-card {css}">'
                        f'<b>{icon} {name.upper()}{badge}</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="connector-card connector-na">'
                        f'<b>— {name.upper()}</b>  <span style="color:#94A3B8">not configured</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"Registry probe failed: {e}")

    st.markdown("---")
    st.markdown("**Required env vars per connector**")
    connector_docs = {
        "Postgres":   "`DATABASE_URL` or `POSTGRES_HOST / DB / USER / PASS`",
        "Snowflake":  "`SNOWFLAKE_ACCOUNT / USER / PASSWORD / DATABASE / SCHEMA / WAREHOUSE`",
        "BigQuery":   "`BQ_PROJECT_ID` + `GOOGLE_APPLICATION_CREDENTIALS`",
        "Redshift":   "`REDSHIFT_HOST / PORT / DB / USER / PASSWORD`",
        "Athena":     "`ATHENA_REGION / S3_STAGING_DIR / DATABASE`",
    }
    for name, vars_text in connector_docs.items():
        st.markdown(f"**{name}:** {vars_text}")

    st.markdown("---")
    st.markdown("**dbt Metrics Sync**")
    st.caption("Point to a dbt `manifest.json` or project dir — metrics auto-populate MetricRegistry.")
    dbt_path = st.text_input("Path to manifest.json or dbt project dir", key="dbt_path_tab2")
    if st.button("Load dbt metrics", key="dbt_load_tab2"):
        try:
            from semantic.dbt_adapter import DbtAdapter
            adapter = DbtAdapter(
                manifest_path=dbt_path if dbt_path.endswith(".json") else None,
                project_dir=dbt_path if not dbt_path.endswith(".json") else None,
            )
            metrics = adapter.load()
            if metrics:
                st.success(f"✓ {len(metrics)} metrics loaded from dbt")
                st.json({k: {"description": v.get("description",""), "formula": v.get("aggregation","")} for k,v in list(metrics.items())[:10]})
            else:
                st.info("No dbt manifest found. Using configs/metrics.yaml.")
        except Exception as e:
            st.error(f"{e}")

# ────────────────────────────────────────────────────────────────────
# TAB 3 — SEMANTIC MEMORY
# ────────────────────────────────────────────────────────────────────

with tab_memory:
    st.subheader("Semantic Memory")
    st.caption("Vector-powered retrieval across all org knowledge. Replaces keyword search.")

    mem = get_org_memory()
    query = st.text_input("Search org knowledge semantically", placeholder="e.g. CAC increase last quarter", key="mem_query")
    if query:
        with st.spinner("Retrieving…"):
            results = mem.semantic_search_context(query, n=8)
        if results:
            st.caption(f"{len(results)} results")
            for r in results:
                score = r.get("score", 0)
                col   = r.get("collection", "?")
                text  = r.get("text", "")
                st.markdown(
                    f'<div class="mem-hit">'
                    f'<span style="font-weight:600;color:#166534">{col}</span> '
                    f'<span style="color:#94A3B8">score={score:.2f}</span><br>'
                    f'{text[:180]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No semantic hits. Run some analyses first to build the memory.")

    st.markdown("---")
    st.markdown("**Manually save insight**")
    ins_kpi     = st.text_input("KPI", key="ins_kpi")
    ins_finding = st.text_area("Finding", key="ins_finding")
    if st.button("Save insight") and ins_finding:
        mem.save_insight(kpi=ins_kpi, finding=ins_finding)
        st.success("Saved to SQLite + vector index.")

    st.markdown("---")
    st.markdown("**All stored insights (SQLite)**")
    all_insights = []
    try:
        import sqlite3
        from pathlib import Path as P
        db = P(mem._db)
        if db.exists():
            with sqlite3.connect(str(db)) as conn:
                rows = conn.execute("SELECT kpi, finding, created_at FROM insight_history ORDER BY id DESC LIMIT 50").fetchall()
                all_insights = [{"kpi":r[0],"finding":r[1],"at":r[2]} for r in rows]
    except Exception:
        pass
    if all_insights:
        st.dataframe(pd.DataFrame(all_insights), use_container_width=True)
    else:
        st.caption("No insights stored yet.")

# ────────────────────────────────────────────────────────────────────
# TAB 4 — SCHEDULER
# ────────────────────────────────────────────────────────────────────

with tab_schedule:
    st.subheader("Proactive Monitor — Schedule")
    st.caption("Set `ENABLE_SCHEDULER=true` in `.env` and edit `configs/schedule.yaml` to add jobs.")

    try:
        from scheduler.monitor import load_schedule, MonitorRunner, AnalyticsScheduler
        jobs = load_schedule()
        if not jobs:
            st.info("No scheduled jobs found. Create `configs/schedule.yaml` with job definitions.")
            with st.expander("Example schedule.yaml"):
                st.code("""
jobs:
  - id: daily_signups
    name: Daily signups anomaly monitor
    connector: postgres
    query: |
      SELECT date_trunc('day', created_at)::date AS date,
             COUNT(*) AS signups
      FROM users
      WHERE created_at >= NOW() - INTERVAL '90 days'
      GROUP BY 1 ORDER BY 1
    kpi_col: signups
    date_col: date
    cron: "0 8 * * *"
    alert_channels: [slack, in_app]
    enabled: true
""", language="yaml")
        else:
            runner = MonitorRunner()
            for job in jobs:
                css = {"ok": "sched-ok", "error": "sched-err"}.get(job.last_status, "sched-pend")
                st.markdown(
                    f'<div class="sched-card {css}">'
                    f'<b>{job.name}</b> &nbsp; <span style="color:#64748B">{job.cron}</span><br>'
                    f'Connector: {job.connector} | KPI: {job.kpi_col} | Last: {job.last_run or "never"} | Status: {job.last_status}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"▶ Run now — {job.name}", key=f"run_{job.job_id}"):
                    with st.spinner(f"Running {job.name}…"):
                        result = runner.run_job(job)
                    if result.get("anomaly_count", 0) > 0:
                        st.warning(f"⚠ {result['anomaly_count']} anomalies detected! Alert {'sent' if result.get('alert_sent') else 'not sent'}.")
                    else:
                        st.success(f"✓ No anomalies. Status: {result.get('status')}")
    except Exception as e:
        st.error(f"Scheduler module error: {e}")

    st.markdown("---")
    st.markdown("**Scheduler status (if running)**")
    try:
        import os
        enabled = os.getenv("ENABLE_SCHEDULER", "false").lower() == "true"
        st.markdown(f"ENABLE_SCHEDULER: **{'true — background scheduler active' if enabled else 'false'}**")
    except Exception:
        pass

# ────────────────────────────────────────────────────────────────────
# TAB 5 — AUDIT
# ────────────────────────────────────────────────────────────────────

with tab_audit:
    st.subheader("Audit Trail")
    st.caption("Immutable log of all external calls and pipeline events. Admin role required for export.")

    is_admin = st.session_state.get("user_role", "analyst") == "admin"
    if not is_admin:
        st.warning("⚠ Admin role required to export audit log. Change role in sidebar.")
    else:
        from api.audit_export import AuditExporter
        exporter = AuditExporter()

        col1, col2, col3 = st.columns(3)
        with col1:
            export_tenant = st.text_input("Tenant (blank=all)", key="audit_tenant")
        with col2:
            export_days = st.number_input("Last N days", min_value=1, max_value=365, value=30, key="audit_days")
        with col3:
            export_fmt = st.selectbox("Format", ["ndjson", "csv"], key="audit_fmt")

        summary = exporter.summary(
            tenant_id=export_tenant or None,
            days=int(export_days),
        )
        st.metric("Total records", summary.get("total_records", 0))

        calls_per_day = summary.get("calls_per_day", {})
        if calls_per_day:
            df_cpd = pd.DataFrame(
                sorted(calls_per_day.items()),
                columns=["date","calls"]
            )
            st.line_chart(df_cpd.set_index("date"))

        from datetime import datetime, timedelta
        since = (datetime.now() - timedelta(days=int(export_days))).isoformat()
        if export_fmt == "csv":
            content = exporter.to_csv(tenant_id=export_tenant or None, since=since, limit=5000)
            mime = "text/csv"
            filename = "audit_export.csv"
        else:
            content = exporter.to_ndjson(tenant_id=export_tenant or None, since=since, limit=5000)
            mime = "application/x-ndjson"
            filename = "audit_export.ndjson"

        st.download_button(
            label=f"⬇ Download {export_fmt.upper()}",
            data=content,
            file_name=filename,
            mime=mime,
        )

        # Team annotations view
        st.markdown("---")
        st.markdown("**Team annotations**")
        ctx_now: AnalysisContext | None = st.session_state.get("last_context")
        if ctx_now:
            try:
                from api.session_manager import SessionStore
                store = SessionStore()
                anns = store.get_annotations(
                    run_id=ctx_now.run_id,
                    tenant_id=st.session_state.get("tenant_id","default"),
                )
                if anns:
                    df_ann = pd.DataFrame([
                        {"finding": a.finding_key, "verdict": a.verdict,
                         "comment": a.comment, "by": a.user_id, "at": a.created_at}
                        for a in anns
                    ])
                    st.dataframe(df_ann, use_container_width=True)
                else:
                    st.caption("No annotations for current run.")
            except Exception as e:
                st.warning(f"Annotations unavailable: {e}")
        else:
            st.caption("Run an analysis first to see its annotations here.")
