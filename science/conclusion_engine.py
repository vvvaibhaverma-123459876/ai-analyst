"""
science/conclusion_engine.py
ConclusionEngine — closes every hypothesis with a formal verdict.

Runs AFTER all analysis agents complete.
For each testable hypothesis:
  1. Collects evidence from assigned agents
  2. Weighs supporting vs opposing evidence
  3. Applies Bayesian-style confidence update
  4. Assigns: CONFIRMED | REJECTED | INCONCLUSIVE
  5. Writes the conclusion paragraph

The output is the ranked hypothesis list with verdicts —
the scientific output that distinguishes this system from a dashboard.
"""

from __future__ import annotations
import json
from agents.context import AnalysisContext
from science.research_plan import ResearchPlan, Hypothesis, HypothesisStatus
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)

_CONCLUSION_SYSTEM = """You are a scientific data analyst closing hypotheses after testing.
For each hypothesis and its evidence, write:
1. Verdict: CONFIRMED | REJECTED | INCONCLUSIVE
2. Confidence: 0.0-1.0
3. One-sentence explanation citing the evidence

Return ONLY valid JSON list:
[{"hypothesis_id": "...", "verdict": "...", "confidence": 0.0, "explanation": "..."}]"""


class ConclusionEngine:

    def close_hypotheses(self, context: AnalysisContext) -> ResearchPlan:
        """
        Main entry point. Runs after all analysis agents.
        Returns updated ResearchPlan with verdicts.
        """
        plan: ResearchPlan = getattr(context, "research_plan", None)
        if plan is None:
            return ResearchPlan()

        testable = plan.testable_hypotheses()
        if not testable:
            logger.info("No testable hypotheses to close.")
            return plan

        # Collect evidence from analysis agents
        self._collect_evidence(plan, context)

        # Score each hypothesis
        for h in testable:
            self._score_hypothesis(h, context)

        # LLM final verdicts
        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            self._llm_verdicts(testable, context)
        else:
            self._rule_verdicts(testable)

        # Build primary conclusion
        confirmed = [h for h in plan.hypotheses if h.status == HypothesisStatus.CONFIRMED]
        if confirmed:
            top = sorted(confirmed, key=lambda h: h.confidence, reverse=True)[0]
            plan.primary_conclusion = (
                f"Primary finding: {top.statement} "
                f"(confidence={top.confidence:.0%}). {top.verdict}"
            )
            plan.overall_confidence = top.confidence
        else:
            inconclusive = [h for h in plan.hypotheses if h.status == HypothesisStatus.INCONCLUSIVE]
            if inconclusive:
                plan.primary_conclusion = (
                    f"No hypothesis confirmed. Most likely: {inconclusive[0].statement[:60]}. "
                    f"Additional data required: {', '.join(plan.data_gaps[:2])}"
                )
            else:
                plan.primary_conclusion = "All tested hypotheses rejected. Data may not support current framing."

        logger.info(f"ConclusionEngine: {len(confirmed)} confirmed, "
                    f"{len([h for h in plan.hypotheses if h.status == HypothesisStatus.REJECTED])} rejected.")
        return plan

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------

    def _collect_evidence(self, plan: ResearchPlan, context: AnalysisContext):
        """Map agent results to hypotheses as evidence."""
        for h in plan.testable_hypotheses():
            for agent_name in h.assigned_agents:
                result = context.results.get(agent_name)
                if not result or result.status != "success":
                    continue

                # Determine if this agent's finding supports or opposes the hypothesis
                supports = self._does_support(h, result, context)
                plan.add_evidence(h.id, {
                    "agent": agent_name,
                    "summary": result.summary[:120],
                    "supports": supports,
                    "confidence": self._extract_confidence(result),
                })

    def _does_support(self, h: Hypothesis, result, context: AnalysisContext) -> bool:
        """Simple heuristic: does the agent's finding align with the hypothesis direction?"""
        stmt = h.statement.lower()
        summary = result.summary.lower()

        # Look for directional alignment
        pos_words = {"increase", "up", "rise", "growth", "confirm", "significant", "found"}
        neg_words = {"decrease", "down", "drop", "decline", "no evidence", "not significant"}

        hyp_positive = any(w in stmt for w in pos_words)
        hyp_negative = any(w in stmt for w in neg_words)
        result_positive = any(w in summary for w in pos_words)
        result_negative = any(w in summary for w in neg_words)

        if hyp_positive and result_positive:
            return True
        if hyp_negative and result_negative:
            return True
        if hyp_positive and result_negative:
            return False
        if hyp_negative and result_positive:
            return False
        return True  # neutral → assume supporting

    def _extract_confidence(self, result) -> float:
        data = result.data or {}
        # Try common confidence fields
        for key in ["confidence", "p_value", "silhouette_score"]:
            val = data.get(key)
            if val is not None:
                if key == "p_value":
                    return round(1.0 - float(val), 3)
                return round(float(val), 3)
        return 0.5

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_hypothesis(self, h: Hypothesis, context: AnalysisContext):
        if not h.evidence:
            h.confidence = 0.0
            return

        n_support = h.supporting_evidence_count
        n_oppose = h.opposing_evidence_count
        n_total = n_support + n_oppose

        if n_total == 0:
            h.confidence = 0.0
            return

        # Base confidence from evidence ratio
        base = n_support / n_total

        # Weight by individual evidence confidence scores
        weighted_support = sum(
            e.get("confidence", 0.5) for e in h.evidence if e.get("supports")
        )
        weighted_oppose = sum(
            e.get("confidence", 0.5) for e in h.evidence if not e.get("supports")
        )
        total_weight = weighted_support + weighted_oppose
        if total_weight > 0:
            h.confidence = round(weighted_support / total_weight, 3)
        else:
            h.confidence = round(base, 3)

    # ------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------

    def _rule_verdicts(self, hypotheses: list[Hypothesis]):
        for h in hypotheses:
            if h.confidence >= 0.70:
                h.status = HypothesisStatus.CONFIRMED
                h.verdict = f"Confirmed with {h.confidence:.0%} confidence based on {len(h.evidence)} evidence sources."
            elif h.confidence <= 0.30:
                h.status = HypothesisStatus.REJECTED
                h.verdict = f"Rejected — opposing evidence outweighs supporting (confidence={h.confidence:.0%})."
            else:
                h.status = HypothesisStatus.INCONCLUSIVE
                h.verdict = f"Inconclusive — mixed evidence (confidence={h.confidence:.0%}). More data needed."

    def _llm_verdicts(self, hypotheses: list[Hypothesis], context: AnalysisContext):
        try:
            from llm.client import LLMClient
            llm = LLMClient()

            hyp_data = [
                {
                    "hypothesis_id": h.id,
                    "statement": h.statement,
                    "evidence": h.evidence[:4],
                    "supporting": h.supporting_evidence_count,
                    "opposing": h.opposing_evidence_count,
                    "current_confidence": h.confidence,
                }
                for h in hypotheses
            ]

            raw = llm.complete(
                system=_CONCLUSION_SYSTEM,
                user=f"KPI: {context.kpi_col}\nHypotheses with evidence:\n{json.dumps(hyp_data, default=str)}",
            )
            raw = raw.strip().strip("```json").strip("```").strip()
            verdicts = json.loads(raw)

            id_map = {h.id: h for h in hypotheses}
            for v in verdicts:
                h = id_map.get(v.get("hypothesis_id"))
                if h:
                    verdict_str = v.get("verdict", "INCONCLUSIVE").upper()
                    h.status = {
                        "CONFIRMED": HypothesisStatus.CONFIRMED,
                        "REJECTED": HypothesisStatus.REJECTED,
                        "INCONCLUSIVE": HypothesisStatus.INCONCLUSIVE,
                    }.get(verdict_str, HypothesisStatus.INCONCLUSIVE)
                    h.confidence = float(v.get("confidence", h.confidence))
                    h.verdict = v.get("explanation", "")

        except Exception as e:
            logger.warning(f"LLM verdicts failed, using rule-based: {e}")
            self._rule_verdicts(hypotheses)
