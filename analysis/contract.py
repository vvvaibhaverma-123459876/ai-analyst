"""
analysis/contract.py  — v9
Shared analysis contract.

Every analysis module must implement this interface so benchmarks, agents,
and the pipeline all speak the same language.  The concrete modules inherit
AnalysisContract and implement the three methods.

Why this matters
----------------
Before v9 each module had its own output shape and method names driven by
history rather than design.  The benchmark suite exposed this drift.  This
contract makes the drift impossible going forward.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any
import pandas as pd


@dataclass
class AnalysisResult:
    """
    Canonical output shape for every analysis module.
    Agents, benchmarks, and the pipeline all read from this.
    """
    module: str                          # e.g. "anomaly", "funnel", "root_cause"
    ok: bool                             # False if input validation failed
    anomaly_count: int = 0               # anomaly / outlier count (0 if N/A)
    records: list[dict] = field(default_factory=list)   # primary output rows
    summary: str = ""                    # one-sentence human summary
    confidence: float = 1.0             # 0–1 output confidence
    metadata: dict[str, Any] = field(default_factory=dict)  # module-specific extras
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    method: str = ""                     # which method produced this result

    def to_dict(self) -> dict:
        return asdict(self)

    def to_agent_data(self) -> dict:
        """Flattened dict shape for AgentResult.data — backward-compatible."""
        d = self.to_dict()
        # Surface top-level fields agents expect
        d["anomaly_count"] = self.anomaly_count
        d["anomaly_records"] = self.records
        d["movers"] = self.metadata.get("movers", {})
        d["forecast"] = self.metadata.get("forecast", [])
        d["periods"] = self.metadata.get("periods", 0)
        return d


class AnalysisContract(ABC):
    """
    Base class every analysis module inherits.
    Provides validate_inputs() and to_benchmark_output() defaults.
    Subclasses implement analyze().
    """

    module_name: str = "base"

    @abstractmethod
    def analyze(self, df: pd.DataFrame, **kwargs) -> AnalysisResult:
        """Run analysis. Returns AnalysisResult regardless of input quality."""

    def validate_inputs(self, df: pd.DataFrame, required_cols: list[str] = None,
                        min_rows: int = 3) -> list[str]:
        """
        Returns list of validation errors.  Empty list = inputs are valid.
        Called at the start of analyze() — subclasses can extend.
        """
        errors: list[str] = []
        if df is None or len(df) == 0:
            errors.append("DataFrame is empty.")
            return errors
        if len(df) < min_rows:
            errors.append(f"Too few rows: {len(df)} < {min_rows}.")
        for col in (required_cols or []):
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'.")
        return errors

    def to_benchmark_output(self, result: AnalysisResult) -> dict:
        """
        Stable dict shape for benchmarks.
        Benchmarks should call this, not access result internals directly.
        """
        return {
            "ok": result.ok,
            "module": result.module,
            "anomaly_count": result.anomaly_count,
            "record_count": len(result.records),
            "summary": result.summary,
            "confidence": result.confidence,
            "method": result.method,
            "warnings": result.warnings,
            "errors": result.errors,
        }
