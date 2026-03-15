from __future__ import annotations

"""Confidence scoring helper for guardian and science layers."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ConfidenceScore:
    score: float
    grade: str
    reasons: list[str] = field(default_factory=list)


class ConfidenceScorer:
    def score(self, *, evidence_confidence: float, reliability: float | None = None, contradictions: int = 0, data_gaps: int = 0) -> ConfidenceScore:
        score = float(evidence_confidence)
        reasons: list[str] = []
        if reliability is not None:
            score = (score * 0.7) + (reliability * 0.3)
            reasons.append(f"Blended with agent reliability={reliability:.2f}.")
        if contradictions:
            score *= max(0.0, 1.0 - 0.15 * contradictions)
            reasons.append(f"Reduced due to {contradictions} contradiction(s).")
        if data_gaps:
            score *= max(0.0, 1.0 - 0.10 * data_gaps)
            reasons.append(f"Reduced due to {data_gaps} data gap(s).")
        score = round(max(0.0, min(1.0, score)), 3)
        if score >= 0.8:
            grade = "strong"
        elif score >= 0.55:
            grade = "moderate"
        elif score >= 0.3:
            grade = "weak"
        else:
            grade = "speculative"
        return ConfidenceScore(score=score, grade=grade, reasons=reasons)
