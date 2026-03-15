"""
science/hypothesis_agent.py
Generates testable hypotheses from four independent sources:
  1. Data-driven  — patterns in the data itself
  2. Business-driven — from org memory + business context
  3. Web-driven  — from external context enrichment (if available)
  4. Prior-driven — from past analyses in org memory

Novelty scoring penalises repeated hypotheses.
Output: populated ResearchPlan with N hypotheses ranked by novelty + testability.
"""

from __future__ import annotations
import uuid
import json
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from science.research_plan import ResearchPlan, Hypothesis, HypothesisStatus
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)

_HYPOTHESIS_SYSTEM = """You are a senior data scientist generating testable hypotheses.
Given data profile, business context, and analysis findings so far,
generate 3-5 specific, testable hypotheses to explain the observed pattern.

Each hypothesis must:
- Be specific (mention metric, direction, magnitude where possible)
- Be testable with the available data
- Have a clear null hypothesis implied
- Be distinct from the others

Return ONLY valid JSON list:
[{"statement": "...", "source": "data", "testable": true, "missing_data": []}]
source must be one of: data, business, web, prior"""

_NOVELTY_SYSTEM = """Score how novel this hypothesis is compared to the list of prior hypotheses.
Return a single float 0.0-1.0. 0.0=identical to prior, 1.0=completely novel.
Return ONLY the float. Nothing else."""


class HypothesisAgent(BaseAgent):
    name = "hypothesis"
    description = "Generates testable hypotheses from data patterns, business context, web signals, and prior analyses"

    def _run(self, context: AnalysisContext) -> AgentResult:
        plan = ResearchPlan()
        all_hypotheses = []

        # Source 1: Data-driven
        data_hyps = self._data_driven(context)
        all_hypotheses.extend(data_hyps)

        # Source 2: Business-driven
        biz_hyps = self._business_driven(context)
        all_hypotheses.extend(biz_hyps)

        # Source 3: Prior-driven (org memory)
        prior_hyps = self._prior_driven(context)
        all_hypotheses.extend(prior_hyps)

        # Source 4: Web-driven (if enrichment available)
        web_hyps = self._web_driven(context)
        all_hypotheses.extend(web_hyps)

        # Score novelty and add to plan
        seen_statements = []
        for h in all_hypotheses:
            h.novelty_score = self._score_novelty(h.statement, seen_statements)
            seen_statements.append(h.statement)
            plan.add_hypothesis(h)

        # Sort by novelty, keep top 6
        plan.hypotheses.sort(key=lambda h: h.novelty_score, reverse=True)
        plan.hypotheses = plan.hypotheses[:6]

        # Store in context
        context.research_plan = plan

        n_hyps = len(plan.hypotheses)
        novel_count = sum(1 for h in plan.hypotheses if h.novelty_score > 0.5)
        summary = (
            f"Generated {n_hyps} hypotheses from {len(set(h.source for h in plan.hypotheses))} sources. "
            f"{novel_count} novel hypotheses. "
            f"Top: '{plan.hypotheses[0].statement[:60]}...'" if plan.hypotheses else ""
        )

        return AgentResult(
            agent=self.name, status="success",
            summary=summary,
            data={
                "research_plan": plan,
                "n_hypotheses": n_hyps,
                "hypotheses": [
                    {"id": h.id, "statement": h.statement,
                     "source": h.source, "novelty": h.novelty_score}
                    for h in plan.hypotheses
                ],
            },
        )

    # ------------------------------------------------------------------
    # Source 1: Data-driven hypotheses
    # ------------------------------------------------------------------

    def _data_driven(self, context: AnalysisContext) -> list[Hypothesis]:
        hypotheses = []
        kpi = context.kpi_col or "the KPI"
        df = context.df

        # Hypothesis from anomaly if present
        anom = context.results.get("anomaly")
        if anom and anom.status == "success" and anom.data.get("anomaly_count", 0) > 0:
            records = anom.data.get("anomaly_records", [])
            if records:
                date = str(records[0].get("date", ""))[:10]
                hypotheses.append(Hypothesis(
                    id=str(uuid.uuid4()),
                    statement=f"The anomaly in {kpi} on {date} was caused by a data pipeline error rather than a real business event",
                    source="data",
                    testable=True,
                ))

        # Hypothesis from root cause
        rc = context.results.get("root_cause")
        if rc and rc.status == "success":
            movers = rc.data.get("movers", {})
            neg = movers.get("negative", [])
            if neg:
                d = neg[0]
                hypotheses.append(Hypothesis(
                    id=str(uuid.uuid4()),
                    statement=f"The decline in {kpi} is primarily driven by a change in {d.get('dimension', 'segment')}={d.get('value', '?')}",
                    source="data",
                    testable=True,
                ))

        # LLM-generated data-driven hypotheses
        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            llm_hyps = self._llm_hypotheses(context, "data")
            hypotheses.extend(llm_hyps)

        return hypotheses

    # ------------------------------------------------------------------
    # Source 2: Business-driven
    # ------------------------------------------------------------------

    def _business_driven(self, context: AnalysisContext) -> list[Hypothesis]:
        biz = context.business_context
        if not biz:
            return []

        hypotheses = []
        kpi = context.kpi_col or "the KPI"
        company = biz.get("company", "the company")
        goal = biz.get("primary_goal", "")

        if goal:
            hypotheses.append(Hypothesis(
                id=str(uuid.uuid4()),
                statement=f"The change in {kpi} is directly related to recent changes in {company}'s strategy toward '{goal}'",
                source="business",
                testable=False,
                missing_data=["strategy_change_date", "initiative_launch_data"],
            ))

        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            llm_hyps = self._llm_hypotheses(context, "business")
            hypotheses.extend(llm_hyps)

        return hypotheses

    # ------------------------------------------------------------------
    # Source 3: Prior-driven
    # ------------------------------------------------------------------

    def _prior_driven(self, context: AnalysisContext) -> list[Hypothesis]:
        try:
            from context_engine.org_memory import OrgMemory
            mem = OrgMemory()
            kpi = context.kpi_col or ""

            # v0.6: use semantic retrieval instead of keyword lookup
            query = f"anomaly pattern {kpi} {context.business_context.get('industry','')}"
            semantic_hits = mem.semantic_prior_insights(query, kpi=kpi, n=5)

            hypotheses = []
            for hit in semantic_hits[:3]:
                text = hit.get("text", "")
                if not text:
                    continue
                score = hit.get("score", 0.5)
                hypotheses.append(Hypothesis(
                    id=str(uuid.uuid4()),
                    statement=f"This pattern resembles a prior finding (similarity={score:.2f}): {text[:80]}",
                    source="prior",
                    novelty_score=round(1 - score, 3),   # high similarity → low novelty
                    testable=True,
                ))

            # Fallback: if vector store returned nothing, use keyword
            if not hypotheses:
                for insight in mem.prior_insights(kpi, n=3):
                    if not insight:
                        continue
                    hypotheses.append(Hypothesis(
                        id=str(uuid.uuid4()),
                        statement=f"This follows the same pattern as a previous finding: {insight[:80]}",
                        source="prior",
                        novelty_score=0.3,
                        testable=True,
                    ))
            return hypotheses
        except Exception as e:
            logger.warning(f"Prior-driven hypothesis failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Source 4: Web-driven
    # ------------------------------------------------------------------

    def _web_driven(self, context: AnalysisContext) -> list[Hypothesis]:
        """If web enrichment context is available, generate web-based hypotheses."""
        enrichment = getattr(context, "enrichment_context", None)
        if not enrichment:
            return []

        hypotheses = []
        for item in enrichment.get("findings", [])[:3]:
            title = item.get("title", "")
            if title:
                hypotheses.append(Hypothesis(
                    id=str(uuid.uuid4()),
                    statement=f"External event may explain the pattern: {title[:80]}",
                    source="web",
                    testable=False,
                    missing_data=["event_date_alignment"],
                ))
        return hypotheses

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    def _llm_hypotheses(self, context: AnalysisContext, source: str) -> list[Hypothesis]:
        try:
            from llm.client import LLMClient
            llm = LLMClient()
            profile_summary = {
                "kpi": context.kpi_col,
                "rows": len(context.df),
                "dimensions": context.data_profile.get("dimensions", [])[:4],
                "business_context": context.business_context,
                "agent_summaries": {
                    k: v.summary[:80]
                    for k, v in context.results.items()
                    if v.status == "success"
                },
            }
            raw = llm.complete(
                system=_HYPOTHESIS_SYSTEM,
                user=f"Source type: {source}\nContext:\n{json.dumps(profile_summary, default=str)}",
            )
            raw = raw.strip().strip("```json").strip("```").strip()
            items = json.loads(raw)
            return [
                Hypothesis(
                    id=str(uuid.uuid4()),
                    statement=item["statement"],
                    source=source,
                    testable=item.get("testable", True),
                    missing_data=item.get("missing_data", []),
                )
                for item in items if "statement" in item
            ]
        except Exception as e:
            logger.warning(f"LLM hypothesis generation ({source}) failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Novelty scoring — penalises repetition
    # ------------------------------------------------------------------

    def _score_novelty(self, statement: str, seen: list[str]) -> float:
        if not seen:
            return 1.0
        from science.research_plan import ResearchPlan
        max_sim = max(ResearchPlan._similarity(statement.lower(), s.lower()) for s in seen)
        return round(1.0 - max_sim, 3)
