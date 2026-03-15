# DEPRECATED — This UI version is superseded by v06_app.py (v0.6+).
# It is retained for reference only and will be removed in v10.
# Do not add new functionality here.

"""
app/ui/v05_app.py — v0.5
Fully autonomous AI analyst — enterprise edition.

New panels vs v0.4:
  - Security status bar (PII masking, policy mode, audit count)
  - Hypothesis panel (research plan, testable vs not-testable, data gaps)
  - Guardian dashboard (agent scores, contradictions, policy blocks)
  - Ground truth verification (thumbs up/down on each finding)
  - Jury panel (consensus level, minority views for anomaly + forecast)
  - Learning status (what each layer has learned so far)
  - External enrichment panel (confirming vs opposing evidence)

Run: streamlit run app/ui/v05_app.py
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

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Analyst v0.5",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
.security-bar {
    background: #F0FDF4; border: 1px solid #86EFAC;
    border-radius: 6px; padding: 6px 14px;
    font-size: 0.82rem; margin-bottom: 8px;
    display: flex; gap: 20px; align-items: center;
}
.security-bar.warn { background: #FFFBEB; border-color: #FCD34D; }
.security-bar.off  { background: #F8FAFC; border-color: #CBD5E1; }
.agent-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:7px; margin:8px 0; }
.agent-card {
    border-radius:7px; padding:9px 11px;
    border-left:4px solid #CBD5E1; background:#F8FAFC;
    font-size:0.80rem; line-height:1.4;
}
.agent-success  { border-left-color:#22C55E; background:#F0FDF4; }
.agent-running  { border-left-color:#F59E0B; background:#FFFBEB; }
.agent-skipped  { border-left-color:#94A3B8; }
.agent-error    { border-left-color:#EF4444; background:#FEF2F2; }
.agent-pending  { border-left-color:#CBD5E1; }
.hypothesis-card {
    border-radius:7px; padding:10px 14px; margin:5px 0;
    font-size:0.85rem; line-height:1.5;
}
.h-testable     { background:#EFF6FF; border-left:4px solid #3B82F6; border-radius:0 7px 7px 0; }
.h-not-testable { background:#F8FAFC; border-left:4px solid #CBD5E1; border-radius:0 7px 7px 0; }
.h-confirmed    { background:#F0FDF4; border-left:4px solid #22C55E; border-radius:0 7px 7px 0; }
.h-rejected     { background:#FEF2F2; border-left:4px solid #EF4444; border-radius:0 7px 7px 0; }
.h-inconclusive { background:#FFFBEB; border-left:4px solid #F59E0B; border-radius:0 7px 7px 0; }
.jury-card {
    border-radius:7px; padding:10px 14px; margin:4px 0;
    background:#F8FAFC; border:1px solid #E2E8F0;
    font-size:0.83rem;
}
.consensus-unanimous { border-left:4px solid #22C55E; }
.consensus-majority  { border-left:4px solid #3B82F6; }
.consensus-split     { border-left:4px solid #F59E0B; }
.consensus-none      { border-left:4px solid #EF4444; }
.gt-card {
    border-radius:7px; padding:10px 14px; margin:5px 0;
    background:#F8FAFC; border:1px solid #E2E8F0;
    font-size:0.83rem;
}
.chat-user      { background:#EFF6FF; border-radius:8px; padding:8px 12px; margin:4px 0; }
.chat-assistant { background:#F0FDF4; border-radius:8px; padding:8px 12px; margin:4px 0; }
.followup       { background:#EFF6FF; border-left:4px solid #3B82F6;
                  padding:9px 13px; border-radius:0 7px 7px 0; margin:4px 0; }
.brief-box      { background:#F8FAFC; border:1px solid #E2E8F0;
                  padding:18px 22px; border-radius:8px; }
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
@st.cache_resource
def get_policy():      return PolicyStore()
@st.cache_resource
def get_audit():       return AuditLogger()

ingestion_engine = get_ingestion()
org_memory       = get_org_memory()
ctx_engine       = get_ctx_engine()
gt_recorder      = get_gt_recorder()
policy           = get_policy()
audit            = get_audit()

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 AI Analyst v0.5")
    st.caption("Autonomous · Confidential · Scientific")
    st.divider()

    # Security settings
    st.markdown("**Security mode**")
    internet_off = policy.get("internet_off_mode", False)
    local_llm    = policy.get("local_llm_mode", False)
    api_ok       = bool(config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY)

    if internet_off:
        st.error("Internet-off mode active")
    elif local_llm:
        st.warning("Local LLM mode active (Ollama)")
    elif api_ok:
        st.success(f"{config.LLM_PROVIDER} / {config.LLM_MODEL}")
    else:
        st.warning("No API key — rule-based fallback")

    st.divider()
    st.markdown("**Column overrides** *(optional)*")
    manual_date = st.text_input("Date column", placeholder="auto-detect")
    manual_kpi  = st.text_input("KPI column",  placeholder="auto-detect")
    grain       = st.radio("Time grain", TIME_GRAINS, horizontal=True)

    st.divider()
    st.markdown("**Tenant / User**")
    tenant_id = st.text_input("Tenant", value="default")
    user_id   = st.text_input("User",   value="analyst")

    st.divider()
    st.markdown("**Org memory**")
    ctx_all = org_memory.get_all_context()
    if ctx_all:
        for k, v in list(ctx_all.items())[:4]:
            st.caption(f"{k}: {str(v)[:28]}")
    if st.button("Clear org memory", use_container_width=True):
        org_memory.clear(); st.success("Cleared.")

    st.divider()
    audit_count = audit.external_call_count(days=7)
    st.caption(f"External calls this week: {audit_count}")
    st.caption("Supported: CSV·Excel·JSON·PDF·Word·Images·SQL·Streams")

# ── Header + security bar ────────────────────────────────────────────

st.title("🧠 AI Analyst v0.5")
st.caption("Drop any data. Ask upfront questions. Get scientific, evidence-ranked, confidentiality-first analysis.")

# Security status bar
pii_mode  = "active" if not policy.get("allow_raw_data_in_prompts", False) else "off"
bar_class = "security-bar" if not internet_off else "security-bar warn"
st.markdown(
    f'<div class="{bar_class}">'
    f'<span>PII masking: <strong>{pii_mode}</strong></span>'
    f'<span>Internet: <strong>{"off" if internet_off else "on"}</strong></span>'
    f'<span>LLM: <strong>{"local" if local_llm else ("external" if api_ok else "none")}</strong></span>'
    f'<span>Policy: <strong>{len(policy.all_checks())} rules active</strong></span>'
    f'<span>Audit log: <strong>{audit_count} external calls / 7d</strong></span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── Session state ─────────────────────────────────────────────────────

def _init():
    defaults = {
        "stage": "upload",
        "documents": [],
        "context": None,
        "questions": [],
        "answers": [],
        "output_decision": None,
        "conversation": None,
        "agent_statuses": {},
        "agent_results_display": {},
        "pipeline_done": False,
        "chat_key": 0,
        "run_id": str(uuid.uuid4()),
        "sec_report": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ════════════════════════════════════════════════════════════════════
# STAGE 1: UPLOAD
# ════════════════════════════════════════════════════════════════════

if st.session_state.stage == "upload":
    st.subheader("Step 1 — Upload your data")
    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded_files = st.file_uploader(
            "Drop any file(s)",
            type=None, accept_multiple_files=True,
            help="Any format supported",
        )
    with col_info:
        st.markdown("""
**Supported formats**
CSV · TSV · Excel (all sheets)
JSON · JSONL · PDF · Word
Images (OCR + vision)
Plain text · Markdown
SQL database files
        """)

    if uploaded_files:
        shell = SecurityShell(tenant_id=tenant_id, user_id=user_id)
        docs, all_rows, all_chunks, all_images = [], 0, 0, 0
        with st.spinner("Parsing and classifying..."):
            for f in uploaded_files:
                doc = ingestion_engine.ingest(f, filename=f.name)
                # Process through security shell
                if doc.has_structured_data:
                    safe_df, sec_report = shell.process_dataframe(
                        doc.primary_df, run_id=st.session_state.run_id
                    )
                    st.session_state.sec_report = sec_report
                    audit.log_ingestion(
                        f.name, doc.source_type, len(doc.primary_df),
                        sec_report.get("summary", ""),
                        doc.warnings, st.session_state.run_id, tenant_id,
                    )
                docs.append(doc)
                all_rows   += sum(len(d) for d in doc.dataframes)
                all_chunks += len(doc.text_chunks)
                all_images += len(doc.image_descriptions)

        st.session_state.documents = docs

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files",        len(docs))
        c2.metric("Data rows",    f"{all_rows:,}")
        c3.metric("Text chunks",  all_chunks)
        c4.metric("Images",       all_images)

        # Show classification summary
        sec = st.session_state.sec_report
        if sec.get("summary"):
            st.info(f"Data classification: {sec['summary']}")
        masked = sec.get("mask_report", {}).get("masked_columns", [])
        if masked:
            st.warning(f"PII masked: {', '.join(masked)}")

        if st.button("Continue →", type="primary", use_container_width=True):
            all_dfs = []
            for doc in docs:
                all_dfs.extend(doc.dataframes)
            primary_df = max(all_dfs, key=len) if all_dfs else pd.DataFrame()

            from analysis.eda_engine import EDAEngine
            from connectors.csv_connector import CSVConnector
            eda = EDAEngine()
            profile = {}
            if not primary_df.empty:
                q = eda.quality_report(primary_df)
                profile = {**q,
                    "kpis": eda.infer_kpis(primary_df),
                    "dimensions": eda.infer_dimensions(primary_df),
                }
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

            doc_summary = " | ".join(d.summary() for d in docs)
            questions = ctx_engine.generate_questions(profile, doc_summary)
            st.session_state.questions = questions
            st.session_state["_primary_df"] = primary_df
            st.session_state["_profile"]    = profile
            st.session_state.stage = "questions"
            st.rerun()

# ════════════════════════════════════════════════════════════════════
# STAGE 2: UPFRONT QUESTIONS
# ════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "questions":
    st.subheader("Step 2 — Business context")
    st.caption("Answers are stored in org memory and improve every future run.")

    questions = st.session_state.questions
    if not questions:
        st.success("Org memory is sufficient — skipping questions.")
        if st.button("Run analysis →", type="primary"):
            st.session_state["_biz_context"] = ctx_engine.load_context()
            st.session_state.stage = "running"
            st.rerun()
    else:
        with st.form("ctx_form"):
            answers = [
                st.text_input(f"Q{i+1}: {q}", key=f"q_{i}")
                for i, q in enumerate(questions)
            ]
            col_s, col_r = st.columns([1, 2])
            with col_s:
                skip = st.form_submit_button("Skip")
            with col_r:
                submitted = st.form_submit_button(
                    "Save + run analysis →", type="primary"
                )

        if submitted or skip:
            if submitted and any(a.strip() for a in answers):
                biz = ctx_engine.enrich_from_answers(questions, answers)
            else:
                biz = ctx_engine.load_context()
            st.session_state["_biz_context"]     = biz
            st.session_state["_questions_asked"] = questions
            st.session_state.stage = "running"
            st.rerun()

# ════════════════════════════════════════════════════════════════════
# STAGE 3: RUNNING
# ════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "running":
    st.subheader("Step 3 — Agent pipeline")

    docs       = st.session_state.documents
    primary_df = st.session_state.get("_primary_df", pd.DataFrame())
    profile    = st.session_state.get("_profile", {})
    biz_ctx    = st.session_state.get("_biz_context", {})

    date_col = manual_date or profile.get("date_col", "")
    kpi_col  = manual_kpi  or (profile.get("kpis") or [""])[0]

    if date_col and not primary_df.empty and date_col in primary_df.columns:
        primary_df[date_col] = pd.to_datetime(primary_df[date_col], errors="coerce")

    shell = SecurityShell(tenant_id=tenant_id, user_id=user_id)

    context = AnalysisContext(
        df=primary_df,
        date_col=date_col or "",
        kpi_col=kpi_col  or "",
        grain=grain,
        filename=docs[0].source_name if docs else "data",
        document=docs[0] if docs else None,
        business_context=biz_ctx,
        data_profile=profile,
        security_shell=shell,
        run_id=st.session_state.run_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    context._questions_asked = st.session_state.get("_questions_asked", [])

    ALL_NAMES = [
        "eda","orchestrator","hypothesis","feasibility",
        "trend","anomaly","root_cause","funnel","cohort",
        "forecast","experiment","ml_cluster","nlp","vision",
        "debate","guardian","insight",
    ]
    statuses = {n: "pending" for n in ALL_NAMES}
    results_map = {}

    status_ph = st.empty()

    def render_status(st_map, rm):
        ICONS = {"pending":"⬜","running":"🟡","success":"🟢",
                 "skipped":"⚪","error":"🔴"}
        html = '<div class="agent-grid">'
        for name in ALL_NAMES:
            s = st_map.get(name, "pending")
            r = rm.get(name)
            dur = f" ({r.duration_sec:.1f}s)" if r and r.duration_sec else ""
            snip = (r.summary[:65]+"…") if r and len(r.summary)>65 else (r.summary if r else "")
            html += (
                f'<div class="agent-card agent-{s}">'
                f'{ICONS[s]} <strong>{name}</strong>{dur}<br>'
                f'<span style="color:#64748B;font-size:0.77rem">{snip}</span>'
                f'</div>'
            )
        html += '</div>'
        status_ph.markdown(html, unsafe_allow_html=True)

    render_status(statuses, results_map)

    def on_start(name):
        statuses[name] = "running"
        render_status(statuses, results_map)

    def on_done(result: AgentResult):
        statuses[result.agent] = result.status
        results_map[result.agent] = result
        render_status(statuses, results_map)

    t0 = time.time()
    runner = AgentRunner(max_workers=6)
    finished = runner.run(context, on_agent_start=on_start, on_agent_done=on_done)
    elapsed = round(time.time() - t0, 1)

    for name in ALL_NAMES:
        if name not in finished.results:
            statuses[name] = "skipped"
            results_map[name] = AgentResult(
                agent=name, status="skipped",
                summary="Not required.", data={}
            )
    render_status(statuses, results_map)

    # Output routing
    router   = OutputRouter()
    decision = router.decide(finished)
    finished._output_decision = decision

    if "alert" in decision.modes and decision.alert_channels != ["in_app"]:
        AlertDispatcher().dispatch(
            decision.alert_channels, decision.alert_message, decision.urgency
        )

    st.session_state.context              = finished
    st.session_state.output_decision      = decision
    st.session_state.agent_statuses       = statuses
    st.session_state.agent_results_display = results_map
    st.session_state.conversation         = ConversationEngine(finished)
    st.session_state.pipeline_done        = True

    st.success(f"Pipeline complete in {elapsed}s")
    time.sleep(0.4)
    st.session_state.stage = "results"
    st.rerun()

# ════════════════════════════════════════════════════════════════════
# STAGE 4: RESULTS
# ════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "results":
    ctx: AnalysisContext = st.session_state.context
    decision             = st.session_state.output_decision
    conv                 = st.session_state.conversation

    # Top bar
    top1, top2, top3 = st.columns([3, 3, 1])
    with top1:
        urgency_color = {"low":"green","medium":"blue","high":"orange","critical":"red"}
        uc = urgency_color.get(decision.urgency, "gray")
        st.markdown(
            f'<span style="color:{uc};font-weight:600">Urgency: {decision.urgency.upper()}</span>'
            f' — {decision.reason[:80]}',
            unsafe_allow_html=True,
        )
    with top2:
        st.markdown("Modes: " + " · ".join(f"**{m}**" for m in decision.modes))
    with top3:
        if st.button("New analysis"):
            for k in ["stage","documents","context","questions","answers",
                      "output_decision","conversation","agent_statuses",
                      "agent_results_display","_primary_df","_profile",
                      "_biz_context","pipeline_done","run_id"]:
                st.session_state.pop(k, None)
            st.session_state.run_id = str(uuid.uuid4())
            st.rerun()

    if "alert" in decision.modes:
        st.error(f"Alert fired: {decision.alert_message[:120]}")

    st.divider()

    tabs = st.tabs([
        "📝 Brief",
        "🔬 Hypotheses",
        "📈 Trend + Forecast",
        "⚠️ Anomalies",
        "🔎 Root Cause",
        "🔬 Experiment",
        "👥 Segments",
        "🔤 NLP + Vision",
        "🤔 Debate",
        "⚖️ Guardian",
        "✅ Verify findings",
        "🧠 Learning",
        "🌐 Enrichment",
        "💬 Chat",
    ])

    # ── BRIEF ──────────────────────────────────────────────────────
    with tabs[0]:
        orch = ctx.results.get("orchestrator")
        if orch:
            st.info(f"**Agents activated:** {' → '.join(orch.data.get('plan', []))}")

        if ctx.final_brief:
            st.markdown('<div class="brief-box">', unsafe_allow_html=True)
            st.markdown(ctx.final_brief)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No brief generated.")

        # Research plan primary conclusion
        plan = ctx.research_plan
        if plan and plan.primary_conclusion:
            st.markdown("---")
            st.markdown("**Scientific conclusion**")
            st.info(plan.primary_conclusion)

        if ctx.follow_up_questions:
            st.markdown("### Suggested next questions")
            for q in ctx.follow_up_questions:
                if st.button(q, key=f"fq_{q[:15]}", use_container_width=True):
                    conv.chat(q)
                    st.rerun()

        t_total = sum(r.duration_sec for r in ctx.results.values() if r.duration_sec)
        st.caption(f"Total agent time: {t_total:.1f}s  ·  Run ID: {ctx.run_id}")

    # ── HYPOTHESES ─────────────────────────────────────────────────
    with tabs[1]:
        plan = ctx.research_plan
        if plan and plan.hypotheses:
            from science.research_plan import HypothesisStatus

            h_agent = ctx.results.get("hypothesis")
            f_agent = ctx.results.get("feasibility")

            if h_agent:
                h1, h2 = st.columns(2)
                h1.metric("Hypotheses generated", len(plan.hypotheses))
                testable_n = sum(1 for h in plan.hypotheses if h.testable)
                h2.metric("Testable", testable_n)

            if f_agent and f_agent.data.get("data_gaps"):
                st.warning(
                    "**Data gaps** — these columns/signals would unlock more hypotheses:
"
                    + "
".join(f"- {g}" for g in f_agent.data["data_gaps"][:5])
                )

            st.markdown("**Hypothesis verdicts**")
            status_class = {
                HypothesisStatus.TESTABLE:      "h-testable",
                HypothesisStatus.NOT_TESTABLE:  "h-not-testable",
                HypothesisStatus.CONFIRMED:     "h-confirmed",
                HypothesisStatus.REJECTED:      "h-rejected",
                HypothesisStatus.INCONCLUSIVE:  "h-inconclusive",
            }
            status_label = {
                HypothesisStatus.TESTABLE:      "TESTABLE",
                HypothesisStatus.NOT_TESTABLE:  "NOT TESTABLE",
                HypothesisStatus.CONFIRMED:     "CONFIRMED",
                HypothesisStatus.REJECTED:      "REJECTED",
                HypothesisStatus.INCONCLUSIVE:  "INCONCLUSIVE",
            }
            for h in plan.ranked_hypotheses() + [
                hh for hh in plan.hypotheses
                if hh.status == HypothesisStatus.TESTABLE
            ]:
                css = status_class.get(h.status, "h-not-testable")
                label = status_label.get(h.status, h.status.value.upper())
                conf_str = f" · conf={h.confidence:.0%}" if h.confidence > 0 else ""
                agents_str = f" · agents: {', '.join(h.assigned_agents)}" if h.assigned_agents else ""
                verdict_str = f"<br><em>{h.verdict}</em>" if h.verdict else ""
                missing_str = ""
                if h.missing_data:
                    missing_str = f"<br><small style='color:#94A3B8'>Missing: {', '.join(h.missing_data[:2])}</small>"
                st.markdown(
                    f'<div class="hypothesis-card {css}">'
                    f'<strong>[{label}]</strong>{conf_str}{agents_str} '
                    f'(source: {h.source}, novelty: {h.novelty_score:.2f})<br>'
                    f'{h.statement}'
                    f'{verdict_str}{missing_str}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Hypothesis pipeline not run or no hypotheses generated.")

    # ── TREND + FORECAST ───────────────────────────────────────────
    with tabs[2]:
        trend = ctx.results.get("trend")
        fcst  = ctx.results.get("forecast")

        if trend and trend.status == "success":
            ts = trend.data.get("ts", ctx.ts)
            comps = trend.data.get("comparisons", {})
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Direction",     trend.data["trend_direction"].title())
            c2.metric("Overall trend", f"{trend.data['trend_pct']:+.1f}%")
            c3.metric("Latest value",  f"{trend.data.get('latest_value',0):,.2f}")
            c4.metric("Data points",   trend.data.get("data_points",0))
            if not ts.empty and ctx.date_col in ts.columns and ctx.kpi_col in ts.columns:
                st.plotly_chart(
                    trend_with_anomalies(ts, ctx.date_col, ctx.kpi_col,
                                         title=f"{ctx.kpi_col} — {ctx.grain}"),
                    use_container_width=True,
                )
            if comps:
                cc = st.columns(len(comps))
                for w, (cn, c) in zip(cc, comps.items()):
                    arrow = "▲" if c["pct_change"] >= 0 else "▼"
                    w.metric(cn, f"{c['current']:,.0f}", f"{arrow}{c['pct_change']:+.1f}%")

        st.divider()
        st.markdown("### Forecast jury")
        if fcst and fcst.status == "success":
            # Jury consensus display
            consensus = fcst.data.get("consensus", "unknown")
            conf = fcst.data.get("confidence", 0)
            mape = fcst.data.get("best_holdout_mape")

            css_c = f"jury-card consensus-{consensus}"
            mape_str = f" · holdout MAPE={mape:.3f}" if mape else ""
            st.markdown(
                f'<div class="{css_c}"><strong>Jury consensus: {consensus.upper()}</strong>'
                f' · confidence={conf:.2f}{mape_str}</div>',
                unsafe_allow_html=True,
            )

            import plotly.graph_objects as go
            fdf = fcst.data.get("forecast_df")
            if fdf is not None and not fdf.empty and ctx.date_col in (ts.columns if trend and trend.status == "success" else []):
                fig = go.Figure()
                ts_plot = trend.data.get("ts", ctx.ts) if trend and trend.status == "success" else ctx.ts
                if not ts_plot.empty:
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
                        fill="toself", fillcolor="rgba(245,158,11,0.12)",
                        line=dict(color="rgba(0,0,0,0)"), name="80% CI",
                    ))
                fig.update_layout(height=380, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Forecast not run for this dataset.")

    # ── ANOMALIES ──────────────────────────────────────────────────
    with tabs[3]:
        anom = ctx.results.get("anomaly")
        if anom and anom.status == "success":
            consensus = anom.data.get("consensus", "unknown")
            conf      = anom.data.get("confidence", 0)
            n         = anom.data.get("anomaly_count", 0)

            css_c = f"jury-card consensus-{consensus}"
            st.markdown(
                f'<div class="{css_c}"><strong>Anomaly jury: {consensus.upper()}</strong>'
                f' · {n} anomalies · confidence={conf:.2f}</div>',
                unsafe_allow_html=True,
            )

            ts_a = ctx.ts
            if not ts_a.empty and ctx.date_col in ts_a.columns and ctx.kpi_col in ts_a.columns:
                st.plotly_chart(
                    trend_with_anomalies(ts_a, ctx.date_col, ctx.kpi_col,
                                         title=f"{ctx.kpi_col} — jury anomaly detection"),
                    use_container_width=True,
                )
            records = anom.data.get("anomaly_records", [])
            if records:
                st.dataframe(pd.DataFrame(records), use_container_width=True)

            # Minority view
            minority = anom.data.get("minority_finding")
            if minority and consensus in ("split", "majority"):
                st.markdown("**Minority view**")
                st.caption(f"Minority jurors: {minority.get('_jurors', [])} "
                           f"(confidence={minority.get('_consensus_confidence', 0):.2f})")
        else:
            st.info("Anomaly detection not run.")

    # ── ROOT CAUSE ─────────────────────────────────────────────────
    with tabs[4]:
        rc = ctx.results.get("root_cause")
        if rc and rc.status == "success":
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Current",  f"{rc.data['last_total']:,.0f}")
            r2.metric("Prior",    f"{rc.data['prev_total']:,.0f}")
            r3.metric("Delta",    f"{rc.data['delta']:+,.0f}")
            r4.metric("% Change", f"{rc.data['pct_change']:+.1f}%")
            drivers = rc.data.get("drivers", pd.DataFrame())
            if not drivers.empty:
                st.plotly_chart(
                    driver_bar_chart(drivers, title=f"Drivers — {ctx.kpi_col}"),
                    use_container_width=True,
                )
            movers = rc.data.get("movers", {})
            nc, pc = st.columns(2)
            with nc:
                st.markdown("**Negative drivers**")
                neg = pd.DataFrame(movers.get("negative", []))
                if not neg.empty: st.dataframe(neg, use_container_width=True)
            with pc:
                st.markdown("**Positive drivers**")
                pos = pd.DataFrame(movers.get("positive", []))
                if not pos.empty: st.dataframe(pos, use_container_width=True)
        else:
            st.info("Root cause not run.")

    # ── EXPERIMENT ─────────────────────────────────────────────────
    with tabs[5]:
        exp = ctx.results.get("experiment")
        if exp and exp.status == "success":
            tt = exp.data.get("results", {}).get("ttest", {})
            sig = "✅ Significant" if exp.data.get("significant") else "❌ Not significant"
            e1,e2,e3,e4 = st.columns(4)
            e1.metric("Result",    sig)
            e2.metric("Lift",      f"{exp.data.get('lift_pct',0):+.1f}%")
            e3.metric("p-value",   f"{exp.data.get('p_value',1):.4f}")
            e4.metric("Cohen's d", f"{exp.data.get('cohens_d',0):.3f}")
        else:
            st.info("No A/B test column detected.")

    # ── SEGMENTS ───────────────────────────────────────────────────
    with tabs[6]:
        clst = ctx.results.get("ml_cluster")
        if clst and clst.status == "success":
            c1,c2,c3 = st.columns(3)
            c1.metric("Clusters",   clst.data.get("n_clusters",0))
            c2.metric("Silhouette", f"{clst.data.get('silhouette_score',0):.3f}")
            c3.metric("Features",   len(clst.data.get("feature_cols",[])))
            pdf = clst.data.get("profile_df")
            if pdf is not None and not pdf.empty:
                names = clst.data.get("cluster_names", [])
                if names and len(names) == len(pdf):
                    pdf = pdf.copy(); pdf.insert(0, "name", names)
                st.dataframe(pdf, use_container_width=True)
            umap_df = clst.data.get("umap_df")
            if umap_df is not None and not umap_df.empty:
                import plotly.express as px
                fig = px.scatter(umap_df, x="x", y="y",
                                 color=umap_df["cluster"].astype(str),
                                 title="UMAP 2D projection",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Clustering not run.")

    # ── NLP + VISION ───────────────────────────────────────────────
    with tabs[7]:
        nlp_r = ctx.results.get("nlp")
        vis_r = ctx.results.get("vision")
        if nlp_r and nlp_r.status == "success":
            st.markdown("**Text analysis**")
            sent = nlp_r.data.get("sentiment", {})
            kw   = nlp_r.data.get("top_keywords", [])
            if sent:
                n1,n2,n3 = st.columns(3)
                n1.metric("Positive", f"{sent.get('positive',0):.0%}")
                n2.metric("Negative", f"{sent.get('negative',0):.0%}")
                n3.metric("Neutral",  f"{sent.get('neutral',0):.0%}")
            if kw:
                st.markdown(f"**Top keywords:** {', '.join(kw[:15])}")
            if nlp_r.data.get("llm_analysis"):
                with st.expander("LLM analysis"):
                    st.markdown(nlp_r.data["llm_analysis"])
        if vis_r and vis_r.status == "success":
            st.markdown("**Visual analysis**")
            st.info(f"{vis_r.data.get('n_images',0)} image(s): "
                    f"{', '.join(set(vis_r.data.get('image_types',[])))}")
            nums = vis_r.data.get("numbers_found", [])
            if nums:
                st.dataframe(pd.DataFrame(nums), use_container_width=True)
        if not (nlp_r and nlp_r.status == "success") and not (vis_r and vis_r.status == "success"):
            st.info("NLP and Vision not activated.")

    # ── DEBATE ─────────────────────────────────────────────────────
    with tabs[8]:
        dbte = ctx.results.get("debate")
        if dbte and dbte.status == "success":
            verdict = dbte.data.get("verdict","medium")
            vc = {"high":"green","medium":"orange","low":"red"}.get(verdict,"gray")
            st.markdown(
                f"**Confidence:** <span style='color:{vc};font-weight:600'>{verdict.upper()}</span>",
                unsafe_allow_html=True,
            )
            for flag in dbte.data.get("red_flags", []):
                st.error(f"⛳ {flag}")
            for ch in dbte.data.get("challenges", []):
                with st.expander(ch.get("finding","")[:70]):
                    st.write(f"**Challenge:** {ch.get('challenge','')}")
                    st.write(f"**Alternative:** {ch.get('alternative_explanation','')}")
                    cf = ch.get("confidence_in_finding","medium")
                    cf_c = {"high":"green","medium":"orange","low":"red"}.get(cf,"gray")
                    st.markdown(
                        f"**Confidence in finding:** <span style='color:{cf_c}'>{cf}</span>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Debate not run.")

    # ── GUARDIAN ───────────────────────────────────────────────────
    with tabs[9]:
        guard = ctx.results.get("guardian")
        if guard and guard.status == "success":
            v = guard.data.get("verdict")
            st.markdown(f"**Guardian verdict:** {'✅ Approved' if v and v.approved else '❌ Blocked'}")

            blocked = guard.data.get("blocked_findings", [])
            if blocked:
                st.markdown("**Policy blocks**")
                for b in blocked:
                    icon = "🔴" if b.get("blocking") else "🟡"
                    st.markdown(f"{icon} **{b.get('rule')}**: {b.get('reason')}")

            warnings = guard.data.get("warnings", [])
            for w in warnings:
                st.warning(w)

            contras = guard.data.get("contradictions", [])
            if contras:
                st.markdown("**Contradictions with prior runs**")
                for c in contras:
                    st.error(
                        f"**{c.get('finding_type')}** — "
                        f"Current: _{c.get('current')}_  vs  "
                        f"Prior: _{c.get('prior')}_"
                    )

            rewrites = guard.data.get("prompt_rewrites", {})
            if rewrites:
                st.markdown("**Prompt improvements suggested**")
                for agent_name, suggestion in rewrites.items():
                    with st.expander(f"{agent_name}"):
                        st.write(suggestion)

            # Agent reliability scores
            st.markdown("**Agent reliability (decay-weighted)**")
            from guardian.guardian_agent import GuardianAgent
            ga = GuardianAgent()
            scores_data = []
            for name in ctx.active_agents:
                rel = ga.get_agent_reliability(name)
                if rel["score"] is not None:
                    scores_data.append({
                        "agent": name,
                        "reliability_score": rel["score"],
                        "n_observations": rel["n"],
                    })
            if scores_data:
                df_scores = pd.DataFrame(scores_data).sort_values(
                    "reliability_score", ascending=False
                )
                st.dataframe(df_scores, use_container_width=True)
            else:
                st.caption("No reliability data yet — builds up over multiple runs.")
        else:
            st.info("Guardian not run.")

    # ── VERIFY FINDINGS ────────────────────────────────────────────
    with tabs[10]:
        st.markdown("**Record ground truth outcomes**")
        st.caption(
            "Thumbs up/down on each finding. "
            "This data trains every learning agent and the Guardian's reliability scores."
        )

        pending = gt_recorder.pending_verification(n=10)
        if not pending:
            st.success("No pending verifications — all findings have outcomes recorded.")
        else:
            for item in pending:
                with st.container():
                    col_desc, col_up, col_down = st.columns([6, 1, 1])
                    with col_desc:
                        st.markdown(
                            f'<div class="gt-card">'
                            f'<strong>{item["agent"]}</strong> · '
                            f'{item["finding_type"]} · '
                            f'conf={item["confidence"]:.2f}<br>'
                            f'{item["finding_summary"][:100]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with col_up:
                        if st.button("✓", key=f"gt_up_{item['finding_id']}",
                                     help="This finding was correct"):
                            gt_recorder.record_outcome(Outcome(
                                finding_id=item["finding_id"],
                                outcome_actual="verified correct",
                                outcome_correct=True,
                                verified_by=user_id,
                            ))
                            audit.log_user_action(
                                "VERIFY", item["finding_id"],
                                {"correct": True}, user_id=user_id,
                            )
                            st.rerun()
                    with col_down:
                        if st.button("✗", key=f"gt_dn_{item['finding_id']}",
                                     help="This finding was incorrect"):
                            gt_recorder.record_outcome(Outcome(
                                finding_id=item["finding_id"],
                                outcome_actual="verified incorrect",
                                outcome_correct=False,
                                verified_by=user_id,
                            ))
                            audit.log_user_action(
                                "VERIFY", item["finding_id"],
                                {"correct": False}, user_id=user_id,
                            )
                            st.rerun()

        # Run quality rating
        st.divider()
        st.markdown("**Rate this analysis run** (1–5)")
        quality = st.slider("Quality", 1, 5, 3, key="run_quality_slider")
        if st.button("Save run rating", use_container_width=True):
            ctx._run_quality = quality
            gt_recorder.update_run_quality(ctx.run_id, quality)
            audit.log_user_action(
                "RATE_RUN", ctx.run_id, {"quality": quality}, user_id=user_id
            )
            st.success(f"Run quality {quality}/5 saved.")

    # ── LEARNING STATUS ────────────────────────────────────────────
    with tabs[11]:
        st.markdown("**Learning layer status — what the system has learned so far**")
        from learning.layer_adapters import (
            IngestionLearner, OrchestratorLearner, AnalysisLearner,
            HypothesisLearner, InsightLearner, OutputRouterLearner
        )
        learners = [
            ("Ingestion", IngestionLearner()),
            ("Orchestrator", OrchestratorLearner()),
            ("Analysis", AnalysisLearner()),
            ("Hypothesis", HypothesisLearner()),
            ("Insight", InsightLearner()),
            ("Output router", OutputRouterLearner()),
        ]
        for layer_name, learner in learners:
            beliefs = learner.all_beliefs()
            if beliefs:
                with st.expander(f"{layer_name} — {len(beliefs)} beliefs"):
                    for key, meta in beliefs.items():
                        st.markdown(
                            f"**{key}**: `{str(meta['value'])[:60]}` "
                            f"(conf={meta['confidence']:.2f}, n={meta['n']})"
                        )
            else:
                st.caption(f"{layer_name}: no beliefs yet — builds up over multiple runs.")

    # ── ENRICHMENT ─────────────────────────────────────────────────
    with tabs[12]:
        enrich = ctx.enrichment_context
        if enrich and enrich.findings:
            n_conf = enrich.confirming_count
            n_opp  = enrich.opposing_count
            signal = enrich.net_signal

            sig_color = {"confirms":"green","contradicts":"red","neutral":"gray"}.get(signal,"gray")
            e1,e2,e3 = st.columns(3)
            e1.metric("Confirming sources", n_conf)
            e2.metric("Opposing sources",   n_opp)
            e3.markdown(
                f"**Net signal:** <span style='color:{sig_color};font-weight:600'>"
                f"{signal.upper()}</span>",
                unsafe_allow_html=True,
            )

            st.markdown("**Sources — confirming**")
            for r in [f for f in enrich.findings if f.direction == "confirming"][:5]:
                st.markdown(
                    f"[{r.title[:60]}]({r.source}) "
                    f"· reliability={r.reliability_score:.2f} "
                    f"· _{r.snippet[:80]}_"
                )

            st.markdown("**Sources — opposing**")
            for r in [f for f in enrich.findings if f.direction == "opposing"][:5]:
                st.markdown(
                    f"[{r.title[:60]}]({r.source}) "
                    f"· reliability={r.reliability_score:.2f} "
                    f"· _{r.snippet[:80]}_"
                )
        else:
            st.info(
                "No external enrichment for this run. "
                "Enrichment triggers automatically when anomalies are detected "
                "and internet access is enabled."
            )

    # ── CHAT ───────────────────────────────────────────────────────
    with tabs[13]:
        st.markdown("**Ask anything about this analysis**")
        st.caption("Try: 'Show me the anomalies', 'What hypothesis was confirmed?', "
                   "'How confident is the forecast?', 'What did the debate flag?'")

        for turn in conv.history:
            css = "chat-user" if turn.role == "user" else "chat-assistant"
            icon = "🧑" if turn.role == "user" else "🤖"
            st.markdown(
                f'<div class="{css}">{icon} {turn.content}</div>',
                unsafe_allow_html=True,
            )

        with st.form(key=f"chat_{st.session_state.chat_key}", clear_on_submit=True):
            msg = st.text_input("Question", placeholder="What caused the anomaly?",
                                label_visibility="collapsed")
            send = st.form_submit_button("Send", use_container_width=True)

        if send and msg.strip():
            with st.spinner("Thinking..."):
                conv.chat(msg.strip())
            st.session_state.chat_key += 1
            st.rerun()

        if st.button("Clear chat", use_container_width=True):
            conv.reset(); st.rerun()
