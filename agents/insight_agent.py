"""
agents/insight_agent.py — v0.7
Synthesises all findings, builds recommendation candidates, ranks them, and saves key findings.
"""

from __future__ import annotations
import json
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from core.config import config
from core.logger import get_logger
from insights.recommendation_ranker import RecommendationRanker

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
        narrative = {k: v for k, v in summaries.items() if k not in ("orchestrator", "eda", "insight")}

        if not narrative:
            return self.skip("No findings to synthesise.")

        if not context.recommendation_candidates:
            context.recommendation_candidates = self._collect_recommendation_candidates(context)

        brief = self._generate_brief(context, narrative)
        followups = self._generate_followups(context, narrative)
        ranked_recommendations = self._rank_recommendations(context)

        context.final_brief = brief
        context.follow_up_questions = followups

        self._save_to_memory(context, brief)

        return AgentResult(
            agent=self.name, status="success",
            summary=f"Brief generated from {len(narrative)} agent(s). {len(followups)} follow-up questions and {len(ranked_recommendations)} ranked actions.",
            data={
                "brief": brief,
                "follow_up_questions": followups,
                "recommendation_candidates": context.recommendation_candidates,
                "ranked_recommendations": [r.to_dict() for r in ranked_recommendations],
                "agents_synthesised": list(narrative.keys()),
            },
        )

    def _collect_recommendation_candidates(self, context: AnalysisContext) -> list[dict]:
        candidates: list[dict] = []
        kpi = context.kpi_col or 'the KPI'

        dq = (context.data_quality_report or {}).get('score', 1.0)
        if dq < 0.6:
            candidates.append({
                'action': 'Resolve data-quality blockers before escalating business conclusions',
                'confidence': 0.98, 'urgency': 1.0, 'business_value': 1.0, 'effort': 0.2,
            })

        rc = context.results.get('root_cause')
        if rc and rc.status == 'success':
            movers = (rc.data or {}).get('movers', {})
            top_neg = (movers.get('negative') or [])[:2]
            for item in top_neg:
                candidates.append({
                    'action': f"Investigate {item.get('dimension')}={item.get('value')} as a likely drag on {kpi}",
                    'confidence': 0.82, 'urgency': 0.8, 'business_value': 0.85, 'effort': 0.35,
                })

        anom = context.results.get('anomaly')
        if anom and anom.status == 'success' and (anom.data or {}).get('anomaly_count', 0) > 0:
            candidates.append({
                'action': f"Validate raw data and operational logs for recent {kpi} anomalies before escalation",
                'confidence': 0.88, 'urgency': 0.85, 'business_value': 0.8, 'effort': 0.3,
            })

        debate = context.results.get('debate')
        if debate and debate.status == 'success' and (debate.data or {}).get('red_flags'):
            candidates.append({
                'action': 'Address debate red flags and confounders before treating the narrative as final',
                'confidence': 0.9, 'urgency': 0.75, 'business_value': 0.9, 'effort': 0.25,
            })

        plan = getattr(context, 'research_plan', None)
        if plan and getattr(plan, 'data_gaps', None):
            candidates.append({
                'action': f"Collect missing evidence: {', '.join(plan.data_gaps[:2])}",
                'confidence': 0.76, 'urgency': 0.65, 'business_value': 0.7, 'effort': 0.45,
            })

        if not candidates:
            candidates = [
                {'action': f'Check the top segment driving change in {kpi}', 'confidence': 0.7, 'urgency': 0.8, 'business_value': 0.8, 'effort': 0.3},
                {'action': 'Validate data freshness and event completeness before escalation', 'confidence': 0.9, 'urgency': 0.9, 'business_value': 0.9, 'effort': 0.2},
                {'action': 'Review same-weekday baseline to avoid false day-over-day inference', 'confidence': 0.75, 'urgency': 0.6, 'business_value': 0.7, 'effort': 0.2},
            ]
        return candidates

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
        urgency = biz.get("urgency", "medium")

        facts = {
            "kpi": context.kpi_col,
            "audience": audience,
            "urgency": urgency,
            "business_context": biz,
            "agent_summaries": summaries,
            "recommendation_candidates": context.recommendation_candidates,
        }
        debate = context.results.get("debate")
        if debate and debate.status == "success":
            facts["debate"] = debate.data

        prompt = json.dumps(facts, ensure_ascii=False, indent=2)
        return llm.complete(_BRIEF_SYSTEM, prompt)

    def _rule_brief(self, context: AnalysisContext, summaries: dict) -> str:
        lines = ["## What happened"]
        for agent, summary in list(summaries.items())[:4]:
            lines.append(f"- {agent}: {summary}")
        lines.append("\n## Why it happened")
        rc = context.results.get('root_cause')
        if rc and rc.status == 'success':
            lines.append(f"- {rc.summary}")
        else:
            lines.append("- Root-cause evidence is still incomplete.")
        lines.append("\n## Confidence assessment")
        debate = context.results.get('debate')
        if debate and debate.status == 'success':
            lines.append(f"- Debate verdict: {debate.data.get('verdict', 'medium')}")
            for flag in (debate.data.get('red_flags') or [])[:2]:
                lines.append(f"- Red flag: {flag}")
        else:
            lines.append("- Confidence is moderate pending challenge review.")
        lines.append("\n## Recommended actions")
        for rec in self._rank_recommendations(context)[:3]:
            lines.append(f"- {rec.action} (score={rec.score})")
        return '\n'.join(lines)

    def _generate_followups(self, context: AnalysisContext, summaries: dict) -> list[str]:
        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            try:
                return self._llm_followups(context, summaries)
            except Exception as e:
                logger.warning(f"LLM follow-up generation failed: {e}")
        return self._rule_followups(context)

    def _llm_followups(self, context: AnalysisContext, summaries: dict) -> list[str]:
        from llm.client import LLMClient
        llm = LLMClient()
        prompt = json.dumps({
            "kpi": context.kpi_col,
            "business_context": context.business_context,
            "summaries": summaries,
        }, ensure_ascii=False, indent=2)
        raw = llm.complete(_FOLLOWUP_SYSTEM, prompt)
        return json.loads(raw)

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

    def _rank_recommendations(self, context: AnalysisContext):
        candidates = context.recommendation_candidates or []
        if not candidates:
            candidates = self._collect_recommendation_candidates(context)
            context.recommendation_candidates = candidates
        return RecommendationRanker().rank(candidates)

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
            profile = context.data_profile
            sig = f"rows:{profile.get('rows',0)}_ts:{profile.get('has_time_series')}_funnel:{profile.get('has_funnel_signal')}"
            mem.save_pattern(sig, context.active_agents, outcome_quality=4)
        except Exception as e:
            logger.warning(f"Could not save to org memory: {e}")
