"""
jury/base_juror.py
Abstract interface every juror implements.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from agents.context import AnalysisContext


@dataclass(init=False)
class JurorVerdict:
    juror_name: str
    method: str
    finding: dict
    confidence: float
    summary: str
    status: str = "success"
    error: str = ""

    def __init__(
        self,
        juror_name: str | None = None,
        method: str = "",
        finding: dict | None = None,
        confidence: float = 0.0,
        summary: str = "",
        status: str = "success",
        error: str = "",
        *,
        juror: str | None = None,
        data: dict | None = None,
    ):
        # Backward compatible aliases: old tests used juror= and data=.
        self.juror_name = juror_name or juror or "unknown_juror"
        self.method = method or self.juror_name
        self.finding = finding if finding is not None else (data or {})
        self.confidence = float(confidence or 0.0)
        self.summary = summary
        self.status = status
        self.error = error

    @property
    def juror(self) -> str:
        return self.juror_name

    @property
    def data(self) -> dict:
        return self.finding


class BaseJuror(ABC):
    name: str = "base_juror"
    method: str = "base"

    @abstractmethod
    def deliberate(self, context: AnalysisContext) -> JurorVerdict:
        """Run this juror's method and return a verdict."""
