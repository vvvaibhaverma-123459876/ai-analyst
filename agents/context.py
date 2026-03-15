"""
agents/context.py — v0.5
AnalysisContext with all v0.5 fields:
  - security_shell reference
  - research_plan (scientific reasoning)
  - enrichment_context (web enrichment)
  - learning adaptations
  - run quality score
  - output decision
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class AgentResult:
    agent: str
    status: str
    summary: str
    data: dict
    error: str = ""
    duration_sec: float = 0.0


@dataclass
class AnalysisContext:
    # --- Core inputs ---
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    ts: pd.DataFrame = field(default_factory=pd.DataFrame)
    date_col: str = ""
    kpi_col: str = ""
    grain: str = "Daily"
    filename: str = "uploaded_file"

    # --- Document (from ingestion) ---
    document: Any = None

    # --- Business context ---
    business_context: dict = field(default_factory=dict)

    # --- Security ---
    security_shell: Any = None          # SecurityShell instance
    run_id: str = ""
    tenant_id: str = "default"
    user_id: str = "system"

    # --- Scientific reasoning ---
    research_plan: Any = None           # ResearchPlan instance

    # --- External enrichment ---
    enrichment_context: Any = None      # EnrichmentContext instance

    # --- Learning adaptations (set by learning layer before pipeline) ---
    learning_adaptations: dict = field(default_factory=dict)

    # --- Orchestrator ---
    active_agents: list[str] = field(default_factory=list)
    data_profile: dict = field(default_factory=dict)

    # --- Agent outputs ---
    results: dict[str, AgentResult] = field(default_factory=dict)

    # --- Final outputs ---
    final_brief: str = ""
    follow_up_questions: list[str] = field(default_factory=list)
    text_summaries: list[str] = field(default_factory=list)

    # --- Quality + feedback (set post-run) ---
    _run_quality: int = 3               # 1-5, set by user or guardian
    _output_decision: Any = None
    _questions_asked: list = field(default_factory=list)
    _alert_acted_on: bool | None = None

    # --- Timing ---
    started_at: float = field(default_factory=time.time)

    def write_result(self, result: AgentResult):
        self.results[result.agent] = result

    def get_summaries(self) -> dict[str, str]:
        return {
            name: r.summary
            for name, r in self.results.items()
            if r.status == "success"
        }

    def get_data(self, agent: str, key: str, default=None) -> Any:
        result = self.results.get(agent)
        if result is None or result.status != "success":
            return default
        return result.data.get(key, default)

    def elapsed(self) -> float:
        return round(time.time() - self.started_at, 1)
