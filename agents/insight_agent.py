"""
agents/insight_agent.py — v0.4
Synthesises ALL agent findings including:
- Debate challenges
- Business context from ContextEngine
- Forecast direction
- Experiment results
- NLP / vision insights
- Cluster segments
Saves key findings to OrgMemory for future sessions.
"""

from __future__ import annotations
import json
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from core.config import config
from core.logger import get_logger

logger = get_logger("agent.insight")

_BRIEF_SYSTEM = """You are a senior analytics lead writing a business intelligence brief.
Use ONLY the facts provided. Calibrate language to the stated audience.
Structure:
## What happened
## Why it happened
## Key findings (include experiment / forecast / segment data if present)
## Confidence assessment (cite any debate challenges)
## Recommended actions
Keep each section to 3-5 bullet points. Use specific numbers."""

_FOLLOWUP_SYSTEM = """You are an analytics lead suggesting the next 4 best questions.
Questions must be specific — mention metric, segment, or time period.
Incorporate the business context provided.
Return a JSON list of exactly 4 question strings. Nothing else."""


class InsightAgent(BaseAgent):
    name = "insight"
    description = "Synthesises all findings into brief, saves to org memory"

    def _run(self, context: AnalysisContext) -> AgentResult:
        summaries = context.get_summaries()
        narrative = {k: v for k, v in summaries.items()
                     if k not in ("orchestrator", "eda", "insight")}

        if not narrative:
            return self.skip("No findings to synthesise.")

        brief = self._generate_brief(context, narrative)
        followups = self._generate_followups(context, narrative)

        context.final_brief = brief
        context.follow_up_questions = followups

        # Save insight to org memory
        self._save_to_memory(context, brief)

        return AgentResult(
            agent=self.name, status="success",
            summary=f"Brief generated from {len(narrative)} agent(s). "
                    f"{len(followups)} follow-up questions.",
            data={
                "brief": brief,
                "follow_up_questions": followups,
                "agents_synthesised": list(narrative.keys()),
            },
        )

    def _generate_brief(self, context: AnalysisContext, summaries: dict) -> str:
        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            try:
                return self._llm_brief(context, summaries)
            except Exception as e:
                logger.warning(f"LLM brief failed: {e}")
        return self._rule_brief(context, summaries)

    def _llm_brief(self, context: AnalysisContext, summaries: dict) -> str:
        from llm.client import LLMClient
        llm = LLMClient()

        biz = context.business_context
        audience = biz.get("audience", "business stakeholders")
        urgency  = biz.get("urgency", "medium")

        # Collect structured numbers
        rc   = context.results.get("root_cause")
        anom = context.results.get("anomaly")
        fcst = context.results.get("forecast")
        exp  = context.results.get("experiment")
        dbte = context.results.get("debate")
        clst = context.results.get("ml_cluster")

        facts: dict = {
            "kpi": context.kpi_col,
            "audience": audience,
            "urgency": urgency,
            "business_context": biz,
            "agent_summaries": summaries,
        }
        if rc and rc.status == "success":
            facts["delta"] = rc.data.get("delta")
            facts["pct_change"] = rc.data.get("pct_change")
            facts["top_drivers"] = rc.data.get("movers", {})
        if anom and anom.status == "success":
            facts["anomaly_count"] = anom.data.get("anomaly_count")
            facts["anomaly_method"] = anom.data.get("method_used")
        if fcst and fcst.status == "success":
            facts["forecast"] = {
                "method": fcst.data.get("method"),
                "direction": fcst.data.get("direction"),
                "pct_change": fcst.data.get("pct_change"),
                "horizon": fcst.data.get("horizon"),
            }
        if exp and exp.status == "success":
            facts["experiment"] = {
                "significant": exp.data.get("significant"),
                "lift_pct": exp.data.get("lift_pct"),
                "p_value": exp.data.get("p_value"),
            }
        if dbte and dbte.status == "success":
            facts["debate_verdict"] = dbte.data.get("verdict")
            facts["red_flags"] = dbte.data.get("red_flags", [])
        if clst and clst.status == "success":
            facts["segments"] = clst.data.get("cluster_names", [])

        return llm.complete(
            system=_BRIEF_SYSTEM,
            user=f"Facts:\n{json.dumps(facts, indent=2, default=str)}",
        )

    def _rule_brief(self, context: AnalysisContext, summaries: dict) -> str:
        lines = [f"## Analysis Brief — {context.kpi_col or 'Dataset'}\n"]
        lines.append("## What happened")
        for agent in ["trend", "root_cause"]:
            r = context.results.get(agent)
            if r and r.status == "success":
                lines.append(f"- {r.summary}")
        lines.append("\n## Anomalies")
        r = context.results.get("anomaly")
        lines.append(f"- {r.summary}" if r and r.status == "success" else "- None detected.")
        lines.append("\n## Forecast")
        r = context.results.get("forecast")
        lines.append(f"- {r.summary}" if r and r.status == "success" else "- Not computed.")
        lines.append("\n## Experiment / A-B")
        r = context.results.get("experiment")
        lines.append(f"- {r.summary}" if r and r.status == "success" else "- No experiment detected.")
        lines.append("\n## Segments")
        r = context.results.get("ml_cluster")
        lines.append(f"- {r.summary}" if r and r.status == "success" else "- No clustering run.")
        lines.append("\n## Confidence")
        r = context.results.get("debate")
        lines.append(f"- {r.summary}" if r and r.status == "success" else "- No debate review.")
        lines.append("\n## All findings")
        for agent, s in summaries.items():
            lines.append(f"- **{agent}**: {s}")
        return "\n".join(lines)

    def _generate_followups(self, context: AnalysisContext, summaries: dict) -> list[str]:
        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            try:
                from llm.client import LLMClient
                llm = LLMClient()
                biz = context.business_context
                facts = "\n".join(f"- {k}: {v}" for k, v in summaries.items())
                raw = llm.complete(
                    system=_FOLLOWUP_SYSTEM,
                    user=f"Business context: {biz}\nKPI: {context.kpi_col}\n\nFindings:\n{facts}",
                )
                raw = raw.strip().strip("```json").strip("```").strip()
                qs = json.loads(raw)
                return qs[:4] if isinstance(qs, list) else []
            except Exception as e:
                logger.warning(f"LLM follow-ups failed: {e}")
        return self._rule_followups(context)

    def _rule_followups(self, context: AnalysisContext) -> list[str]:
        kpi = context.kpi_col or "the KPI"
        dims = context.data_profile.get("dimensions", [])
        dim = dims[0] if dims else "top segment"
        return [
            f"Which {dim} contributed most to the change in {kpi} last week?",
            f"Is the {kpi} trend consistent across all channels?",
            f"Are there hour-level or day-of-week patterns in the anomalies?",
            f"How does the current {kpi} compare to the same period last quarter?",
        ]

    def _save_to_memory(self, context: AnalysisContext, brief: str):
        try:
            from context_engine.org_memory import OrgMemory
            mem = OrgMemory()
            kpi = context.kpi_col or "unknown"
            mem.save_insight(
                kpi=kpi,
                finding=brief[:500],
                date_range=f"{context.df[context.date_col].min() if context.date_col and not context.df.empty and context.date_col in context.df.columns else 'unknown'} to today",
            )
            # Save agent plan pattern
            profile = context.data_profile
            sig = f"rows:{profile.get('rows',0)}_ts:{profile.get('has_time_series')}_funnel:{profile.get('has_funnel_signal')}"
            mem.save_pattern(sig, context.active_agents, outcome_quality=4)
        except Exception as e:
            logger.warning(f"Could not save to org memory: {e}")
