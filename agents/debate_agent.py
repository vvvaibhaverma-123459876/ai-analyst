"""
agents/debate_agent.py
Debate Agent — the system's internal critic.

Always runs after all analysis agents.
Its job: challenge every major finding and surface alternative explanations.

This prevents the system from presenting a single narrative as truth
when the data could support multiple interpretations.

Output:
  - challenges: list of {finding, challenge, alternative_explanation, confidence}
  - verdict: overall confidence in primary narrative (high/medium/low)
  - red_flags: data quality or methodology issues spotted
"""

from __future__ import annotations
import json
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)

_DEBATE_SYSTEM = """You are a senior data scientist reviewing an analytics report
before it reaches business leadership. Your job is to be the devil's advocate.

For each major finding provided, you must:
1. Identify the strongest counter-argument or alternative explanation
2. Flag any statistical concerns (sample size, confounding, correlation ≠ causation)
3. Note any data quality issues that could invalidate the finding
4. Rate your confidence that the finding is genuinely correct (high/medium/low)

Be specific. Use the data facts provided.
Return ONLY valid JSON in this format:
{
  "challenges": [
    {
      "finding": "...",
      "challenge": "...",
      "alternative_explanation": "...",
      "confidence_in_finding": "high|medium|low",
      "data_quality_flag": "..." or null
    }
  ],
  "verdict": "high|medium|low",
  "verdict_reason": "...",
  "red_flags": ["..."]
}"""


class DebateAgent(BaseAgent):
    name = "debate"
    description = "Challenges findings, surfaces alternative explanations, flags data quality issues"

    def _run(self, context: AnalysisContext) -> AgentResult:
        summaries = context.get_summaries()
        # Filter to analysis agents only
        analysis_summaries = {
            k: v for k, v in summaries.items()
            if k not in ("eda", "orchestrator", "debate", "insight")
        }

        if not analysis_summaries:
            return self.skip("No analysis findings to challenge.")

        # Rule-based challenges first (always run, no LLM cost)
        rule_challenges = self._rule_challenges(context)

        # LLM deep challenge
        llm_result = None
        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            llm_result = self._llm_challenge(analysis_summaries, context)

        challenges = rule_challenges
        verdict = "medium"
        verdict_reason = "Automated review complete."
        red_flags = []

        if llm_result:
            challenges = llm_result.get("challenges", rule_challenges)
            verdict = llm_result.get("verdict", "medium")
            verdict_reason = llm_result.get("verdict_reason", "")
            red_flags = llm_result.get("red_flags", [])

        n_low = sum(1 for c in challenges if c.get("confidence_in_finding") == "low")
        n_flags = len(red_flags)

        summary = (
            f"Debate review: {len(challenges)} finding(s) challenged. "
            f"Overall narrative confidence: {verdict}. "
            f"{n_low} low-confidence finding(s). "
            f"{n_flags} red flag(s) raised."
        )
        if verdict_reason:
            summary += f" {verdict_reason[:80]}"

        return AgentResult(
            agent=self.name,
            status="success",
            summary=summary,
            data={
                "challenges": challenges,
                "verdict": verdict,
                "verdict_reason": verdict_reason,
                "red_flags": red_flags,
                "n_challenges": len(challenges),
                "n_low_confidence": n_low,
            },
        )

    def _rule_challenges(self, context: AnalysisContext) -> list[dict]:
        challenges = []
        df = context.df

        # Small sample warning
        n = len(df)
        if n < 100:
            challenges.append({
                "finding": "All findings from this dataset",
                "challenge": f"Sample size is small (n={n}). Results may not generalise.",
                "alternative_explanation": "Patterns could be noise rather than signal.",
                "confidence_in_finding": "low",
                "data_quality_flag": f"Only {n} rows. Consider collecting more data.",
            })

        # Root cause — correlation warning
        rc = context.results.get("root_cause")
        if rc and rc.status == "success":
            delta = rc.data.get("delta", 0)
            if abs(delta) > 0:
                challenges.append({
                    "finding": rc.summary[:100],
                    "challenge": "Driver attribution shows correlation, not causation.",
                    "alternative_explanation":
                        "The segment that changed most may reflect a confounding variable "
                        "(e.g. seasonality, external campaign, data pipeline issue).",
                    "confidence_in_finding": "medium",
                    "data_quality_flag": None,
                })

        # Anomaly — data quality check
        anom = context.results.get("anomaly")
        if anom and anom.status == "success" and anom.data.get("anomaly_count", 0) > 0:
            challenges.append({
                "finding": anom.summary[:100],
                "challenge": "Anomalies could be data pipeline errors, not real business events.",
                "alternative_explanation":
                    "Check for ETL issues, duplicate ingestion, or timezone misalignment "
                    "on the flagged dates before escalating.",
                "confidence_in_finding": "medium",
                "data_quality_flag": "Verify raw data source on anomaly dates.",
            })

        # Experiment — underpowered test warning
        exp = context.results.get("experiment")
        if exp and exp.status == "success":
            p = exp.data.get("p_value", 1.0)
            n_a = exp.data.get("results", {}).get("ttest", {}).get("n_a", 0)
            if n_a < 100:
                challenges.append({
                    "finding": f"A/B test result (p={p:.3f})",
                    "challenge": f"Test may be underpowered (group A n={n_a}). "
                                 f"Risk of false positive or false negative.",
                    "alternative_explanation": "Run test longer before concluding significance.",
                    "confidence_in_finding": "low" if n_a < 30 else "medium",
                    "data_quality_flag": f"Minimum sample size recommendation: n≥200 per group.",
                })

        return challenges

    def _llm_challenge(
        self, summaries: dict, context: AnalysisContext
    ) -> dict | None:
        try:
            from llm.client import LLMClient
            llm = LLMClient()

            facts = {
                "kpi": context.kpi_col,
                "rows": len(context.df),
                "findings": summaries,
                "business_context": context.business_context,
            }

            raw = llm.complete(
                system=_DEBATE_SYSTEM,
                user=f"Analysis facts:\n{json.dumps(facts, indent=2, default=str)}",
            )
            raw = raw.strip().strip("```json").strip("```").strip()
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"LLM debate failed: {e}")
            return None
