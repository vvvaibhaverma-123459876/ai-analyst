"""
jury/base_juror.py
Abstract interface every juror implements.
A juror is a single method within an agent jury.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from agents.context import AnalysisContext


@dataclass
class JurorVerdict:
    juror_name: str
    method: str
    finding: dict            # structured result
    confidence: float        # 0.0 – 1.0
    summary: str
    status: str = "success"  # success | skipped | error
    error: str = ""


class BaseJuror(ABC):
    name: str = "base_juror"
    method: str = "base"

    @abstractmethod
    def deliberate(self, context: AnalysisContext) -> JurorVerdict:
        """Run this juror's method and return a verdict."""
