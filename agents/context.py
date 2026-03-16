"""
agents/context.py — v10
AnalysisContext with phase-barrier support (v10 gap closure).

v10 additions:
  - freeze() returns a ContextSnapshot — a shallow copy of all inputs
    and completed results, safe to read from parallel agents without
    race conditions on the live context.
  - Parallel agents receive the snapshot; results still write back to
    the live context via write_result() which is thread-safe (GIL-protected
    dict assignment).
"""

from __future__ import annotations
import copy
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

    # --- Truth / trust runtime controls ---
    data_quality_report: dict = field(default_factory=dict)
    approval_log: list[dict] = field(default_factory=list)
    run_manifest: dict = field(default_factory=dict)
    output_mode: str = 'business'
    recommendation_candidates: list[dict] = field(default_factory=list)

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


    def freeze(self) -> "ContextSnapshot":
        """
        Returns a ContextSnapshot: a frozen, shallow copy of this context
        suitable for read-only access by parallel agents.

        - DataFrames are NOT copied (read-only in parallel agents, ~zero cost).
        - results dict IS copied so agents see only completed results at
          the moment the snapshot was taken, not mid-run writes.
        - security_shell, research_plan, etc. are passed by reference
          (they are either immutable or thread-safe internally).
        """
        return ContextSnapshot(
            df=self.df,
            ts=self.ts,
            date_col=self.date_col,
            kpi_col=self.kpi_col,
            grain=self.grain,
            filename=self.filename,
            business_context=dict(self.business_context),
            data_profile=dict(self.data_profile),
            active_agents=list(self.active_agents),
            results=dict(self.results),          # snapshot of completed results
            security_shell=self.security_shell,
            research_plan=self.research_plan,
            run_id=self.run_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            learning_adaptations=dict(self.learning_adaptations),
            data_quality_report=dict(self.data_quality_report),
        )

    def write_result(self, result: "AgentResult"):
        """Thread-safe: dict assignment is atomic under the GIL."""
        self.results[result.agent] = result

    def elapsed(self) -> float:
        return round(time.time() - self.started_at, 1)


@dataclass
class ContextSnapshot:
    """
    Frozen read-only view of AnalysisContext at a point in time.
    Passed to parallel agents so they read a consistent state.
    Agents MUST NOT write to the snapshot — write back to the live context.
    """
    df:                  pd.DataFrame
    ts:                  pd.DataFrame = field(default_factory=pd.DataFrame)
    date_col:            str = ""
    kpi_col:             str = ""
    grain:               str = "Daily"
    filename:            str = ""
    business_context:    dict = field(default_factory=dict)
    data_profile:        dict = field(default_factory=dict)
    active_agents:       list = field(default_factory=list)
    results:             dict = field(default_factory=dict)
    security_shell:      Any  = None
    research_plan:       Any  = None
    run_id:              str  = ""
    tenant_id:           str  = "default"
    user_id:             str  = "system"
    learning_adaptations:dict = field(default_factory=dict)
    data_quality_report: dict = field(default_factory=dict)

    def get_data(self, agent: str, key: str, default=None) -> Any:
        result = self.results.get(agent)
        if result is None or result.status != "success":
            return default
        return result.data.get(key, default)
