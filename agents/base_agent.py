"""
agents/base_agent.py
Abstract base every agent must implement.
All agents expose one public method: run(context) → AgentResult.
"""

from __future__ import annotations
import time
from abc import ABC, abstractmethod
from agents.context import AnalysisContext, AgentResult
from core.logger import get_logger


class BaseAgent(ABC):

    name: str = "base"          # override in subclass
    description: str = ""       # shown in UI

    def __init__(self):
        self.logger = get_logger(f"agent.{self.name}")

    def run(self, context: AnalysisContext) -> AgentResult:
        """
        Entry point called by the runner.
        Wraps _run() with timing and error handling.
        """
        self.logger.info(f"[{self.name}] starting")
        t0 = time.time()
        try:
            result = self._run(context)
            result.duration_sec = round(time.time() - t0, 2)
            self.logger.info(f"[{self.name}] done in {result.duration_sec}s — {result.status}")
            return result
        except Exception as e:
            self.logger.error(f"[{self.name}] error: {e}")
            return AgentResult(
                agent=self.name,
                status="error",
                summary=f"{self.name} failed: {e}",
                data={},
                error=str(e),
                duration_sec=round(time.time() - t0, 2),
            )

    @abstractmethod
    def _run(self, context: AnalysisContext) -> AgentResult:
        """Implement analysis logic here. Return an AgentResult."""

    def skip(self, reason: str) -> AgentResult:
        """Convenience: return a skip result."""
        self.logger.info(f"[{self.name}] skipped — {reason}")
        return AgentResult(
            agent=self.name,
            status="skipped",
            summary=f"Skipped: {reason}",
            data={},
        )
