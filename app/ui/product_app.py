"""
AI Analyst Product App
======================

A production-oriented Streamlit surface that presents the platform as a modern
analytics application rather than a collection of experiments.

Run:
    streamlit run app/ui/product_app.py

Design goals:
- One guided workflow: connect data -> validate schema -> run analysis -> act.
- Make hidden capabilities visible: governed pipeline, security shell, semantic
  layer, anomaly jury, root-cause engine, forecast jury, memory and audit.
- Work offline by default using deterministic/rule-based analysis.
"""

from __future__ import annotations

import io
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from agents.context import AnalysisContext
from agents.runner import AgentRunner, AGENT_REGISTRY
from charts.chart_builder import trend_with_anomalies, driver_bar_chart, funnel_chart, cohort_heatmap, kpi_comparison_bar
from ingestion.ingestion_engine import IngestionEngine
from output.conversation_engine import ConversationEngine
from security.security_shell import SecurityShell

APP_VERSION = "production-ui-1.0"
ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_GROWTH = ROOT / "sample_growth_data.csv"
SAMPLE_FINTECH = ROOT / "sample_fintech_onboarding.csv"


@dataclass
class ColumnGuess:
    date_col: str | None
    kpi_col: str | None
    dimensions: list[str]
    user_col: str | None
    stage_col: str | None


# -----------------------------------------------------------------------------
# Page setup + styles
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Analyst Command Center",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1020;
            --panel: rgba(255, 255, 255, 0.06);
            --panel-2: rgba(255, 255, 255, 0.09);
            --border: rgba(255, 255, 255, 0.14);
            --text-muted: rgba(255, 255, 255, 0.68);
            --accent: #7c3aed;
            --accent-2: #06b6d4;
            --success: #10b981;
            --warn: #f59e0b;
            --danger: #ef4444;
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 18%, rgba(124,58,237,.22), transparent 28%),
                radial-gradient(circle at 80% 4%, rgba(6,182,212,.18), transparent 24%),
                linear-gradient(135deg, #070b16 0%, #101827 46%, #0b1020 100%);
            color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background: rgba(4, 8, 20, 0.92);
            border-right: 1px solid rgba(255,255,255,.10);
        }
        [data-testid="stSidebar"] * { color: #e5e7eb; }
        .block-container { padding-top: 1.2rem; max-width: 1500px; }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.13);
            padding: 1rem 1.05rem;
            border-radius: 18px;
            box-shadow: 0 18px 44px rgba(0,0,0,.24);
        }
        div[data-testid="stMetric"] label { color: rgba(255,255,255,.70) !important; }
        .hero {
            border: 1px solid rgba(255,255,255,.14);
            border-radius: 28px;
            padding: 28px 30px;
            margin-bottom: 18px;
            background:
                linear-gradient(135deg, rgba(124,58,237,.24), rgba(6,182,212,.09)),
                rgba(255,255,255,.055);
            box-shadow: 0 24px 80px rgba(0,0,0,.34);
        }
        .hero-kicker {
            color: #a5b4fc;
            text-transform: uppercase;
            letter-spacing: .14em;
            font-size: .76rem;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.55rem;
            letter-spacing: -0.05em;
            line-height: 1.05;
        }
        .hero p { color: rgba(255,255,255,.74); max-width: 850px; font-size: 1.02rem; }
        .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
        .pill {
            border: 1px solid rgba(255,255,255,.14);
            background: rgba(255,255,255,.075);
            color: rgba(255,255,255,.84);
            border-radius: 999px;
            padding: 7px 11px;
            font-size: .78rem;
            font-weight: 700;
        }
        .surface {
            border: 1px solid rgba(255,255,255,.13);
            background: rgba(255,255,255,.065);
            border-radius: 22px;
            padding: 18px 18px;
            box-shadow: 0 16px 48px rgba(0,0,0,.22);
            margin-bottom: 14px;
        }
        .surface h3 { margin-top: 0; letter-spacing: -.02em; }
        .soft { color: rgba(255,255,255,.66); }
        .capability-grid {
            display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin: 14px 0 10px;
        }
        .cap-card {
            background: rgba(255,255,255,.065);
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 18px;
            padding: 15px 15px 14px;
            min-height: 112px;
        }
        .cap-card b { display:block; font-size: .94rem; margin-bottom: 6px; }
        .cap-card span { color: rgba(255,255,255,.65); font-size: .82rem; line-height: 1.42; }
        .status-dot { display:inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }
        .ok { background: #10b981; box-shadow: 0 0 0 4px rgba(16,185,129,.14); }
        .warn { background: #f59e0b; box-shadow: 0 0 0 4px rgba(245,158,11,.14); }
        .bad { background: #ef4444; box-shadow: 0 0 0 4px rgba(239,68,68,.14); }
        .agent-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 9px; }
        .agent-card {
            border: 1px solid rgba(255,255,255,.12);
            background: rgba(255,255,255,.06);
            border-radius: 14px;
            padding: 11px 12px;
            font-size: .82rem;
        }
        .agent-card small { display:block; color: rgba(255,255,255,.6); margin-top: 2px; }
        .brief {
            background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.045));
            border: 1px solid rgba(255,255,255,.14);
            border-radius: 20px;
            padding: 20px 22px;
            font-size: .96rem;
            line-height: 1.62;
            white-space: pre-wrap;
        }
        .mini-table {
            border: 1px solid rgba(255,255,255,.12);
            background: rgba(255,255,255,.05);
            border-radius: 16px;
            padding: 10px;
        }
        .step {
            padding: 10px 11px;
            border-radius: 14px;
            background: rgba(255,255,255,.055);
            border: 1px solid rgba(255,255,255,.10);
            margin-bottom: 8px;
        }
        .step b { font-size:.86rem; }
        .step span { color: rgba(255,255,255,.62); font-size:.78rem; }
        .stButton > button {
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,.16) !important;
            background: linear-gradient(135deg, #7c3aed, #0891b2) !important;
            color: white !important;
            font-weight: 800 !important;
            min-height: 42px;
            box-shadow: 0 12px 32px rgba(8,145,178,.18);
        }
        .stDownloadButton > button {
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,.16) !important;
            background: rgba(255,255,255,.08) !important;
            color: white !important;
            font-weight: 700 !important;
        }
        div[data-baseweb="select"] > div, input, textarea {
            background-color: rgba(255,255,255,.08) !important;
            color: #f8fafc !important;
            border-color: rgba(255,255,255,.14) !important;
            border-radius: 14px !important;
        }
        [data-testid="stTabs"] button { color: rgba(255,255,255,.78); }
        .footer-note { color: rgba(255,255,255,.52); font-size: .78rem; margin-top: 12px; }
        @media (max-width: 1100px) {
            .capability-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
            .agent-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def pct_delta(value: float | None) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):+.1f}%"
    except Exception:
        return None


def metric_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


@st.cache_data(show_spinner=False)
def load_sample(kind: str) -> pd.DataFrame:
    path = SAMPLE_FINTECH if kind == "Fintech onboarding demo" and SAMPLE_FINTECH.exists() else SAMPLE_GROWTH
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def ingestion_engine() -> IngestionEngine:
    return IngestionEngine()


def read_uploaded_file(uploaded) -> tuple[pd.DataFrame, Any, list[str]]:
    if uploaded is None:
        return pd.DataFrame(), None, []
    content = uploaded.getvalue()
    doc = ingestion_engine().ingest(io.BytesIO(content), filename=uploaded.name, mime_type=getattr(uploaded, "type", ""))
    df = doc.primary_df
    return df, doc, doc.warnings


def guess_columns(df: pd.DataFrame) -> ColumnGuess:
    if df is None or df.empty:
        return ColumnGuess(None, None, [], None, None)

    cols = list(df.columns)
    lower = {c: c.lower().strip() for c in cols}

    date_scores: list[tuple[int, str]] = []
    for c in cols:
        score = 0
        name = lower[c]
        if any(k in name for k in ["date", "time", "created", "signup", "event"]):
            score += 3
        sample = pd.to_datetime(df[c].dropna().head(30), errors="coerce")
        if len(sample) and sample.notna().mean() >= 0.65:
            score += 3
        if score:
            date_scores.append((score, c))
    date_col = sorted(date_scores, reverse=True)[0][1] if date_scores else None

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    kpi_preference = ["revenue", "conversion", "success", "active", "orders", "signups", "users", "amount", "value", "kpi", "metric"]
    kpi_col = None
    if numeric_cols:
        scored: list[tuple[int, float, str]] = []
        for c in numeric_cols:
            name = lower[c]
            score = sum(4 - min(i, 3) for i, k in enumerate(kpi_preference) if k in name)
            if any(k in name for k in ["id", "pin", "phone", "mobile"]):
                score -= 5
            variance = float(pd.to_numeric(df[c], errors="coerce").std() or 0)
            scored.append((score, variance, c))
        scored = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)
        kpi_col = scored[0][2]

    dims = []
    for c in cols:
        if c in {date_col, kpi_col}:
            continue
        nunique = df[c].nunique(dropna=True)
        if df[c].dtype == "object" or nunique <= 25:
            if 1 < nunique <= max(25, len(df) // 2):
                dims.append(c)

    user_col = None
    for c in cols:
        if any(k in lower[c] for k in ["user_id", "userid", "customer_id", "member_id", "account_id", "user", "customer"]):
            user_col = c
            break

    stage_col = None
    for c in cols:
        if any(k in lower[c] for k in ["stage", "step", "event", "funnel", "status"]):
            stage_col = c
            break

    return ColumnGuess(date_col, kpi_col, dims[:8], user_col, stage_col)


def build_report(context: AnalysisContext) -> str:
    lines = [
        "# AI Analyst Report",
        "",
        f"Dataset: `{context.filename}`",
        f"KPI: `{context.kpi_col}`",
        f"Date column: `{context.date_col}`",
        f"Grain: `{context.grain}`",
        f"Run ID: `{context.run_id}`",
        "",
        "## Executive brief",
        context.final_brief or "No brief generated.",
        "",
        "## Agent findings",
    ]
    for name, result in context.results.items():
        lines.extend([
            f"### {name}",
            f"Status: `{result.status}`",
            result.summary or "No summary.",
            "",
        ])
    if context.follow_up_questions:
        lines.extend(["## Suggested follow-up questions", ""])
        lines.extend([f"- {q}" for q in context.follow_up_questions])
    return "\n".join(lines)


def run_analysis(df: pd.DataFrame, filename: str, date_col: str, kpi_col: str, grain: str, business_context: dict) -> AnalysisContext:
    shell = SecurityShell(tenant_id=business_context.get("tenant_id", "default"), user_id=business_context.get("user_id", "streamlit"))
    safe_df, security_report = shell.process_dataframe(df.copy(), run_id="streamlit-preview")
    context = AnalysisContext(
        df=safe_df,
        date_col=date_col,
        kpi_col=kpi_col,
        grain=grain,
        filename=filename,
        business_context=business_context | {"security_report": security_report},
        tenant_id=business_context.get("tenant_id", "default"),
        user_id=business_context.get("user_id", "streamlit"),
        security_shell=shell,
        output_mode="business",
    )

    progress = st.progress(0, text="Starting governed pipeline…")
    status_box = st.empty()
    total_agents = max(len(AGENT_REGISTRY), 1)
    state = {"done": 0, "started": []}

    def on_start(name: str) -> None:
        state["started"].append(name)
        progress.progress(min(0.92, state["done"] / total_agents), text=f"Running {name}…")
        status_box.markdown(render_agent_grid(state["started"], context.results), unsafe_allow_html=True)

    def on_done(result) -> None:
        state["done"] += 1
        progress.progress(min(0.96, state["done"] / total_agents), text=f"Completed {result.agent}")
        status_box.markdown(render_agent_grid(state["started"], context.results), unsafe_allow_html=True)

    finished = AgentRunner().run(context, on_agent_start=on_start, on_agent_done=on_done)
    progress.progress(1.0, text="Analysis complete")
    return finished


def render_agent_grid(started: list[str], results: dict[str, Any]) -> str:
    ordered = ["eda", "orchestrator", "hypothesis", "feasibility", "trend", "anomaly", "root_cause", "funnel", "cohort", "forecast", "ml_cluster", "debate", "guardian", "insight"]
    cards = []
    for name in ordered:
        result = results.get(name)
        if result:
            cls = "ok" if result.status == "success" else ("warn" if result.status == "skipped" else "bad")
            status = result.status
        elif name in started:
            cls = "warn"
            status = "running"
        else:
            cls = ""
            status = "waiting"
        cards.append(f"<div class='agent-card'><span class='status-dot {cls}'></span><b>{name}</b><small>{status}</small></div>")
    return "<div class='agent-grid'>" + "".join(cards) + "</div>"


# -----------------------------------------------------------------------------
# Renderers
# -----------------------------------------------------------------------------


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">AI Analyst Command Center · production UI</div>
            <h1>One app for governed, explainable business analysis.</h1>
            <p>
                Upload data, validate schema, run the governed agent pipeline, inspect root causes,
                forecast what happens next, and ask grounded follow-up questions — without exposing
                users to the system's internal complexity.
            </p>
            <div class="pill-row">
                <span class="pill">Semantic Layer</span>
                <span class="pill">Security Shell</span>
                <span class="pill">Anomaly Jury</span>
                <span class="pill">Root Cause Engine</span>
                <span class="pill">Forecast Jury</span>
                <span class="pill">Guardian Review</span>
                <span class="pill">Memory + Audit</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_capability_map() -> None:
    st.markdown(
        """
        <div class="capability-grid">
            <div class="cap-card"><b>Governed pipeline</b><span>Every answer flows through data quality, semantic resolution, agents, debate, guardian review, and security publishing.</span></div>
            <div class="cap-card"><b>Analyst intelligence</b><span>Trend, anomaly, root cause, funnel, cohort, forecast, clustering, recommendations, and follow-up Q&A.</span></div>
            <div class="cap-card"><b>Trust controls</b><span>PII masking, tenant-aware access checks, output classification, audit logs, replay manifests, and evidence grading.</span></div>
            <div class="cap-card"><b>Product workflow</b><span>Guided setup, schema validation, analysis console, executive brief, deep dives, exportable reports, and chat.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, str, str, str, dict]:
    with st.sidebar:
        st.markdown("## ◈ AI Analyst")
        st.caption(f"{APP_VERSION}")
        st.markdown("---")
        st.markdown("<div class='step'><b>1. Connect data</b><br><span>CSV, Excel, JSON, or demo dataset.</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='step'><b>2. Validate schema</b><br><span>Select KPI, date grain, and context.</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='step'><b>3. Run agents</b><br><span>Governed pipeline with security and guardian.</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='step'><b>4. Act</b><br><span>Brief, root cause, forecasts, Q&A, export.</span></div>", unsafe_allow_html=True)
        st.markdown("---")
        tenant_id = st.text_input("Tenant", value=st.session_state.get("tenant_id", "default"), key="tenant_id")
        user_id = st.text_input("User", value=st.session_state.get("user_id", "analyst"), key="user_id")
        audience = st.selectbox("Output audience", ["Leadership", "Product/Growth", "Data team", "Operations"], index=1)
        goal = st.text_area("Business question", value=st.session_state.get("business_goal", "Find what changed, why it changed, and what action to take next."), key="business_goal", height=90)
        return tenant_id, user_id, audience, goal, {"tenant_id": tenant_id, "user_id": user_id, "audience": audience, "primary_goal": goal}


def render_data_connector() -> tuple[pd.DataFrame, str, Any, list[str]]:
    st.markdown("<div class='surface'><h3>Connect data</h3><p class='soft'>Start with a demo dataset or upload your own CSV, Excel, or JSON file. The app uses the same ingestion engine as the backend.</p></div>", unsafe_allow_html=True)
    source = st.radio("Data source", ["Fintech onboarding demo", "Growth KPI demo", "Upload file"], horizontal=True, label_visibility="collapsed")
    filename = ""
    doc = None
    warnings: list[str] = []

    if source == "Upload file":
        uploaded = st.file_uploader("Drop a CSV, Excel, or JSON file", type=["csv", "tsv", "xlsx", "xls", "json", "jsonl", "ndjson"])
        if uploaded is not None:
            df, doc, warnings = read_uploaded_file(uploaded)
            filename = uploaded.name
        else:
            df = pd.DataFrame()
    else:
        df = load_sample(source)
        filename = SAMPLE_FINTECH.name if source == "Fintech onboarding demo" and SAMPLE_FINTECH.exists() else SAMPLE_GROWTH.name

    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        with st.expander("Preview data and schema", expanded=False):
            st.dataframe(df.head(50), use_container_width=True)
            schema = pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns], "nulls": [int(df[c].isna().sum()) for c in df.columns], "unique": [int(df[c].nunique(dropna=True)) for c in df.columns]})
            st.dataframe(schema, use_container_width=True)
    elif source == "Upload file":
        st.info("Upload a file to continue.")

    for warning in warnings:
        st.warning(warning)

    return df, filename, doc, warnings


def render_schema_controls(df: pd.DataFrame) -> tuple[str | None, str | None, str]:
    if df.empty:
        return None, None, "Daily"
    guess = guess_columns(df)
    st.markdown("<div class='surface'><h3>Validate analysis setup</h3><p class='soft'>The app makes smart guesses, but the user stays in control before agents run.</p></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.1, 1.1, .8])
    with c1:
        date_index = list(df.columns).index(guess.date_col) if guess.date_col in df.columns else 0
        date_col = st.selectbox("Date column", list(df.columns), index=date_index)
    with c2:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        kpi_options = numeric_cols or list(df.columns)
        kpi_index = kpi_options.index(guess.kpi_col) if guess.kpi_col in kpi_options else 0
        kpi_col = st.selectbox("KPI column", kpi_options, index=kpi_index)
    with c3:
        grain = st.selectbox("Time grain", ["Daily", "Weekly", "Monthly"], index=0)

    cols = st.columns(4)
    cols[0].metric("Auto dimensions", len(guess.dimensions))
    cols[1].metric("User ID", guess.user_col or "Not found")
    cols[2].metric("Funnel stage", guess.stage_col or "Not found")
    cols[3].metric("Numeric cols", len(numeric_cols))
    return date_col, kpi_col, grain


def render_results(context: AnalysisContext) -> None:
    st.markdown("---")
    st.markdown("## Command center")
    trend = context.results.get("trend")
    anomaly = context.results.get("anomaly")
    root = context.results.get("root_cause")
    forecast = context.results.get("forecast")
    guardian = context.results.get("guardian")

    c1, c2, c3, c4, c5 = st.columns(5)
    latest = trend.data.get("latest_value") if trend and trend.status == "success" else None
    trend_pct = trend.data.get("trend_pct") if trend and trend.status == "success" else None
    c1.metric("Latest KPI", metric_value(latest), delta=pct_delta(trend_pct))
    c2.metric("Rows analyzed", f"{len(context.df):,}")
    c3.metric("Agents active", len(context.active_agents))
    c4.metric("Anomalies", anomaly.data.get("anomaly_count", 0) if anomaly and anomaly.status == "success" else "—")
    c5.metric("Trust review", guardian.status if guardian else "pending")

    overview, deep_dive, agents_tab, chat_tab, export_tab = st.tabs(["Executive brief", "Deep dive", "Agent console", "Ask follow-up", "Export"])

    with overview:
        st.markdown("### Executive brief")
        st.markdown(f"<div class='brief'>{context.final_brief or 'No final brief generated.'}</div>", unsafe_allow_html=True)
        if context.follow_up_questions:
            st.markdown("### Suggested next questions")
            for q in context.follow_up_questions[:5]:
                st.info(q)
        render_capability_map()

    with deep_dive:
        c_left, c_right = st.columns([1.2, 1])
        with c_left:
            st.markdown("### KPI trend and anomalies")
            if not context.ts.empty and context.date_col in context.ts.columns and context.kpi_col in context.ts.columns:
                try:
                    st.plotly_chart(trend_with_anomalies(context.ts, context.date_col, context.kpi_col, f"{context.kpi_col} trend"), use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render trend chart: {e}")
            else:
                st.info("No time-series chart available.")
        with c_right:
            st.markdown("### Period comparison")
            if trend and trend.status == "success":
                comps = trend.data.get("comparisons", {})
                comp_rows = [{"comparison": k, "pct_change": v.get("pct_change", 0)} for k, v in comps.items() if isinstance(v, dict)]
                if comp_rows:
                    st.plotly_chart(kpi_comparison_bar(comp_rows), use_container_width=True)
                else:
                    st.info(trend.summary)
            else:
                st.info("Trend agent did not produce comparisons.")

        st.markdown("### Root cause and drivers")
        if root and root.status == "success":
            st.write(root.summary)
            drivers = root.data.get("drivers")
            if isinstance(drivers, pd.DataFrame) and not drivers.empty:
                try:
                    st.plotly_chart(driver_bar_chart(drivers, title="Top drivers"), use_container_width=True)
                    st.dataframe(drivers.head(25), use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render driver chart: {e}")
        else:
            st.info("Root cause analysis was skipped or unavailable.")

        col_funnel, col_cohort = st.columns(2)
        with col_funnel:
            st.markdown("### Funnel")
            funnel = context.results.get("funnel")
            funnel_df = funnel.data.get("funnel_df") if funnel and funnel.status == "success" else None
            if isinstance(funnel_df, pd.DataFrame) and not funnel_df.empty:
                st.plotly_chart(funnel_chart(funnel_df), use_container_width=True)
                st.dataframe(funnel_df, use_container_width=True)
            elif funnel:
                st.info(funnel.summary)
            else:
                st.info("No funnel output.")
        with col_cohort:
            st.markdown("### Cohort / retention")
            cohort = context.results.get("cohort")
            matrix = cohort.data.get("retention_matrix") if cohort and cohort.status == "success" else None
            if isinstance(matrix, pd.DataFrame) and not matrix.empty:
                st.plotly_chart(cohort_heatmap(matrix), use_container_width=True)
                st.dataframe(matrix, use_container_width=True)
            elif cohort:
                st.info(cohort.summary)
            else:
                st.info("No cohort output.")

        if forecast:
            st.markdown("### Forecast")
            st.info(forecast.summary)
            fdf = forecast.data.get("forecast_df") if forecast.status == "success" else None
            if isinstance(fdf, pd.DataFrame) and not fdf.empty:
                st.dataframe(fdf.head(30), use_container_width=True)

    with agents_tab:
        st.markdown(render_agent_grid(context.active_agents, context.results), unsafe_allow_html=True)
        st.markdown("### Raw agent summaries")
        for name, result in context.results.items():
            with st.expander(f"{name} · {result.status}", expanded=name in {"anomaly", "root_cause", "guardian", "insight"}):
                st.write(result.summary)
                if result.error:
                    st.error(result.error)
                compact = {}
                for k, v in result.data.items():
                    if isinstance(v, pd.DataFrame):
                        compact[k] = f"DataFrame{v.shape}"
                    else:
                        compact[k] = v
                st.json(compact, expanded=False)

    with chat_tab:
        st.markdown("### Ask the analysis")
        st.caption("Grounded in the completed run. It will not invent metrics outside this context.")
        if "chat_context_id" not in st.session_state or st.session_state.get("chat_context_id") != context.run_id:
            st.session_state["chat_context_id"] = context.run_id
            st.session_state["chat_engine"] = ConversationEngine(context)
            st.session_state["chat_messages"] = []

        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask: what changed, why, which segment, how reliable, what next…")
        if prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            answer = st.session_state["chat_engine"].chat(prompt)
            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
            st.rerun()

    with export_tab:
        report = build_report(context)
        st.download_button("Download Markdown report", report, file_name="ai_analyst_report.md", mime="text/markdown")
        payload = {
            "run_id": context.run_id,
            "filename": context.filename,
            "kpi_col": context.kpi_col,
            "date_col": context.date_col,
            "grain": context.grain,
            "brief": context.final_brief,
            "agents": {name: {"status": r.status, "summary": r.summary} for name, r in context.results.items()},
            "follow_up_questions": context.follow_up_questions,
            "run_manifest": context.run_manifest,
        }
        st.download_button("Download JSON payload", json.dumps(payload, indent=2, default=str), file_name="ai_analyst_run.json", mime="application/json")
        st.markdown("<p class='footer-note'>Production note: API, memory, audit, and security surfaces are available for backend deployment. This UI is the operator-facing command center.</p>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    inject_css()
    tenant_id, user_id, audience, goal, business_context = render_sidebar()
    render_hero()

    df, filename, doc, warnings = render_data_connector()
    date_col, kpi_col, grain = render_schema_controls(df)

    if df.empty:
        render_capability_map()
        return

    ready = bool(date_col and kpi_col and date_col in df.columns and kpi_col in df.columns)
    run_col, note_col = st.columns([.35, .65])
    with run_col:
        run_clicked = st.button("Run governed analysis", type="primary", disabled=not ready, use_container_width=True)
    with note_col:
        st.caption("The run will execute the canonical pipeline: EDA → DQ → semantic resolution → agents → debate → guardian → insight → security publish.")

    if run_clicked:
        with st.spinner("Running the full analysis stack…"):
            try:
                context = run_analysis(df, filename, date_col, kpi_col, grain, business_context)
                st.session_state["last_context"] = context
                st.session_state["last_filename"] = filename
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

    context = st.session_state.get("last_context")
    if isinstance(context, AnalysisContext):
        render_results(context)
    else:
        st.markdown("<div class='surface'><h3>Before you run</h3><p class='soft'>This product surface is designed to reveal the platform's true capability: governed analysis, trust controls, agent reasoning, visual deep dives, and grounded conversation in one seamless workflow.</p></div>", unsafe_allow_html=True)
        render_capability_map()


if __name__ == "__main__":
    main()
