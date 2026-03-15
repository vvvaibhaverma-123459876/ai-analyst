"""
llm/insight_generator.py
Generates analyst narratives, executive summaries, and follow-up questions.
"""

from llm.client import LLMClient
from llm.prompts import Prompts
from core.exceptions import LLMError
from core.logger import get_logger

logger = get_logger(__name__)


class InsightGenerator:

    def __init__(self, llm_client: LLMClient = None):
        self._llm = llm_client or LLMClient()

    def executive_summary(self, payload: dict) -> str:
        """Full 4-section executive summary."""
        try:
            return self._llm.complete(
                system=Prompts.EXEC_SUMMARY_SYSTEM,
                user=Prompts.exec_summary_user(payload),
            )
        except Exception as e:
            raise LLMError(f"Executive summary generation failed: {e}") from e

    def analyst_narrative(self, analysis_output: dict) -> str:
        """Short 3-5 sentence analyst commentary."""
        try:
            return self._llm.complete(
                system=Prompts.INSIGHT_SYSTEM,
                user=Prompts.insight_user(analysis_output),
            )
        except Exception as e:
            raise LLMError(f"Analyst narrative generation failed: {e}") from e

    def follow_up_questions(self, question: str, analysis_summary: str) -> list[str]:
        """Returns list of 4 follow-up question strings."""
        try:
            raw = self._llm.complete(
                system=Prompts.FOLLOWUP_SYSTEM,
                user=Prompts.followup_user(question, analysis_summary),
            )
            lines = [
                line.strip().lstrip("1234567890.)- ").strip()
                for line in raw.splitlines()
                if line.strip() and line.strip()[0].isdigit()
            ]
            return lines[:4] if lines else [raw]
        except Exception as e:
            raise LLMError(f"Follow-up question generation failed: {e}") from e

    def root_cause_narrative(
        self, kpi: str, delta: float, pct: float,
        drivers: list, anomalies: list
    ) -> str:
        """Root cause explanation paragraph."""
        try:
            return self._llm.complete(
                system=Prompts.ROOT_CAUSE_SYSTEM,
                user=Prompts.root_cause_user(kpi, delta, pct, drivers, anomalies),
            )
        except Exception as e:
            raise LLMError(f"Root cause narrative failed: {e}") from e
