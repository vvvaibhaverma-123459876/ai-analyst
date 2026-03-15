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
from science.evidence_registry import EvidenceRegistry, EvidenceRecord, EvidenceState
from science.uncertainty_model import UncertaintyModel
from guardian.confidence_scorer import ConfidenceScorer
from guardian.evidence_grader import EvidenceGrader
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

    def __init__(self):
        self._evidence_registry = EvidenceRegistry()
        self._uncertainty_model = UncertaintyModel()
        self._confidence_scorer = ConfidenceScorer()
        self._evidence_grader = EvidenceGrader()


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
                # v0.6: auto-design experiment for top inconclusive hypothesis
                self._propose_experiment(inconclusive[0], context, plan)
            else:
                plan.primary_conclusion = "All tested hypotheses rejected. Data may not support current framing."

        logger.info(f"ConclusionEngine: {len(confirmed)} confirmed, "
                    f"{len([h for h in plan.hypotheses if h.status == HypothesisStatus.REJECTED])} rejected.")
        return plan

    # ------------------------------------------------------------------
    # v0.6: Experiment auto-design for inconclusive hypotheses
    # ------------------------------------------------------------------

    def _propose_experiment(self, hypothesis, context, plan):
        """Auto-design an experiment when a hypothesis is inconclusive."""
        try:
            from science.experiment_designer import ExperimentDesigner
            designer = ExperimentDesigner()
            spec = designer.design(
                hypothesis=hypothesis.statement,
                metric=context.kpi_col or "metric",
                df=context.df,
                kpi_col=context.kpi_col or "",
                expected_lift_pct=5.0,
            )
            plan.experiment_spec = spec
            logger.info("ExperimentDesigner: proposed spec id=%s", spec.spec_id)
        except Exception as e:
            logger.warning("ExperimentDesigner failed (non-fatal): %s", e)


    def _collect_evidence(self, plan: ResearchPlan, context: AnalysisContext):
        """Map agent results to hypotheses as evidence."""
        for h in plan.testable_hypotheses():
            for agent_name in h.assigned_agents:
                result = context.results.get(agent_name)
                if not result or result.status != "success":
                    continue

                supports = self._does_support(h, result, context)
                record = EvidenceRecord(
                    hypothesis_id=h.id,
                    agent=agent_name,
                    summary=result.summary[:160],
                    supports=supports,
                    confidence=self._extract_confidence(result),
                    metadata={"agent": agent_name},
                )
                plan.add_evidence(h.id, {
                    "agent": record.agent,
                    "summary": record.summary,
                    "supports": record.supports,
                    "confidence": record.confidence,
                    "metadata": record.metadata,
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
        records = [
            EvidenceRecord(
                hypothesis_id=h.id,
                agent=e.get("agent", "unknown"),
                summary=e.get("summary", ""),
                supports=e.get("supports"),
                confidence=float(e.get("confidence", 0.5)),
                metadata=e.get("metadata", {}),
            )
            for e in h.evidence
        ]

        summary = self._evidence_registry.summarise(records, h.missing_data)
        uncertainty = self._uncertainty_model.assess(
            evidence_confidence=summary.confidence,
            n_evidence=len(records),
            data_gaps=len(h.missing_data),
            contradictions=1 if summary.state == EvidenceState.CONTRADICTED else 0,
        )
        confidence = self._confidence_scorer.score(
            evidence_confidence=summary.confidence,
            contradictions=1 if summary.state == EvidenceState.CONTRADICTED else 0,
            data_gaps=len(h.missing_data),
        )
        dq_score = (getattr(context, 'data_quality_report', {}) or {}).get('score')
        evidence_grade = self._evidence_grader.grade(
            support_ratio=getattr(summary, 'support_ratio', summary.confidence),
            confidence=confidence.score,
            contradictions=1 if summary.state == EvidenceState.CONTRADICTED else 0,
            data_quality_score=dq_score,
        )

        if dq_score is not None and dq_score < 0.5:
            h.confidence = round(min(confidence.score, evidence_grade.summary_strength) * max(0.45, dq_score + 0.2), 3)
        else:
            h.confidence = confidence.score
        h.evidence_state = summary.state.value
        h.evidence_grade = evidence_grade.grade
        h.uncertainty_level = uncertainty.level

    # ------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------

    def _rule_verdicts(self, hypotheses: list[Hypothesis]):
        for h in hypotheses:
            if h.evidence_state == EvidenceState.UNTESTABLE.value:
                h.status = HypothesisStatus.NOT_TESTABLE
                h.verdict = "Not testable with currently available data."
            elif h.evidence_state == EvidenceState.SUPPORTED.value and h.confidence >= 0.65:
                h.status = HypothesisStatus.CONFIRMED
                h.verdict = (
                    f"Confirmed with {h.confidence:.0%} confidence based on {len(h.evidence)} evidence sources; "
                    f"grade={h.evidence_grade}, uncertainty={h.uncertainty_level}."
                )
            elif h.evidence_state == EvidenceState.CONTRADICTED.value and h.confidence <= 0.35:
                h.status = HypothesisStatus.REJECTED
                h.verdict = (
                    f"Rejected — opposing evidence outweighs support (confidence={h.confidence:.0%}); "
                    f"grade={h.evidence_grade}, uncertainty={h.uncertainty_level}."
                )
            else:
                h.status = HypothesisStatus.INCONCLUSIVE
                h.verdict = (
                    f"Inconclusive — evidence state={h.evidence_state} (confidence={h.confidence:.0%}); "
                    f"grade={h.evidence_grade}, uncertainty={h.uncertainty_level}."
                )

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
