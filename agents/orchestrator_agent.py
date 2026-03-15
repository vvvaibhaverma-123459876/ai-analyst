"""
agents/orchestrator_agent.py
Orchestrator Agent v0.4 — extended with new agent types.
Reads data profile + business context + document content
to decide the full agent roster.
"""

from __future__ import annotations
import json
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)

ALL_AGENTS = ["trend", "anomaly", "root_cause", "funnel", "cohort",
              "forecast", "experiment", "ml_cluster", "nlp", "vision",
              "debate", "insight"]

_PLAN_SYSTEM = """You are an analytics orchestrator deciding which analysis agents to run.
Available agents and when to use them:
- trend: has date column + numeric KPI
- anomaly: has date + enough rows (≥20)
- root_cause: has categorical dimensions + numeric KPI
- funnel: has stage/event column with conversion-like values
- cohort: has user ID + date column + ≥50 rows
- forecast: has date + numeric KPI + ≥10 rows
- experiment: has A/B test / variant / group column
- ml_cluster: has multiple numeric columns + ≥20 rows (no date needed)
- nlp: has text columns or document text chunks
- vision: has image descriptions in document
- debate: always include if any other agents run (critic role)
- insight: always last

Return ONLY a JSON list of agent names in run order. insight must be last. No explanation."""


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    description = "Plans agent roster from data profile + business context"

    def _run(self, context: AnalysisContext) -> AgentResult:
        profile = context.data_profile
        doc = context.document
        biz = context.business_context

        if not profile:
            context.active_agents = ALL_AGENTS
            return AgentResult(
                agent=self.name, status="success",
                summary="No profile — activating all agents as fallback.",
                data={"plan": ALL_AGENTS, "method": "fallback"},
            )

        plan = self._heuristic_plan(profile, doc, biz)

        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            plan = self._llm_enrich(plan, profile, doc, biz)

        # debate always before insight
        if len(plan) > 2 and "debate" not in plan:
            plan.insert(-1, "debate")
        if "insight" not in plan:
            plan.append("insight")

        context.active_agents = plan
        logger.info(f"Agent plan: {plan}")

        return AgentResult(
            agent=self.name, status="success",
            summary=f"Activated {len(plan)} agents: {', '.join(plan)}.",
            data={"plan": plan},
        )

    def _heuristic_plan(self, profile: dict, doc, biz: dict) -> list[str]:
        plan = []
        rows = profile.get("rows", 0)
        has_ts   = profile.get("has_time_series", False)
        has_fun  = profile.get("has_funnel_signal", False)
        has_coh  = profile.get("has_cohort_signal", False)
        has_dims = bool(profile.get("dimensions"))
        has_txt  = (doc and doc.has_text) if doc else False
        has_img  = (doc and bool(doc.image_descriptions)) if doc else False
        kpis     = profile.get("kpis", [])
        dims     = profile.get("dimensions", [])

        if has_ts and rows >= 14:    plan.append("trend")
        if has_ts and rows >= 20:    plan.append("anomaly")
        if has_dims and rows >= 10:  plan.append("root_cause")
        if has_fun:                  plan.append("funnel")
        if has_coh and rows >= 50:   plan.append("cohort")
        if has_ts and rows >= 10:    plan.append("forecast")

        # Experiment: check column names for A/B keywords
        ab_kws = ["variant","group","treatment","control","experiment","arm","bucket"]
        if any(any(kw in str(c).lower() for kw in ab_kws) for c in dims + kpis):
            plan.append("experiment")
        elif rows >= 20 and len(kpis) >= 2:
            plan.append("ml_cluster")

        if has_txt:  plan.append("nlp")
        if has_img:  plan.append("vision")

        if len(plan) > 0:
            plan.append("debate")
        plan.append("insight")
        return plan

    def _llm_enrich(self, heuristic: list, profile: dict, doc, biz: dict) -> list[str]:
        try:
            from llm.client import LLMClient
            llm = LLMClient()
            doc_summary = doc.summary() if doc else "no document"
            user = (
                f"Data profile: {json.dumps(profile, indent=2, default=str)}\n"
                f"Document: {doc_summary}\n"
                f"Business context: {json.dumps(biz, default=str)}\n"
                f"Heuristic plan: {heuristic}\n\n"
                "Refine or confirm. Return JSON list only."
            )
            raw = llm.complete(system=_PLAN_SYSTEM, user=user)
            raw = raw.strip().strip("```json").strip("```").strip()
            plan = json.loads(raw)
            if isinstance(plan, list) and all(isinstance(x, str) for x in plan):
                return plan
        except Exception as e:
            logger.warning(f"LLM plan enrichment failed: {e}")
        return heuristic
