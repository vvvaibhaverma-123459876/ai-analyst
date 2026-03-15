from __future__ import annotations

"""Simple uncertainty scorer combining evidence strength and data gaps."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class UncertaintyAssessment:
    score: float
    level: str
    reasons: list[str] = field(default_factory=list)


class UncertaintyModel:
    def assess(self, *, evidence_confidence: float, n_evidence: int, data_gaps: int = 0, contradictions: int = 0) -> UncertaintyAssessment:
        score = float(evidence_confidence)
        reasons: list[str] = []

        if n_evidence <= 1:
            score *= 0.75
            reasons.append("Limited evidence volume.")
        if data_gaps:
            score *= max(0.0, 1.0 - 0.12 * data_gaps)
            reasons.append(f"{data_gaps} data gap(s) weaken certainty.")
        if contradictions:
            score *= max(0.0, 1.0 - 0.18 * contradictions)
            reasons.append(f"{contradictions} contradiction(s) detected.")

        score = max(0.0, min(1.0, round(score, 3)))
        if score >= 0.75:
            level = "low_uncertainty"
        elif score >= 0.45:
            level = "medium_uncertainty"
        else:
            level = "high_uncertainty"
        return UncertaintyAssessment(score=score, level=level, reasons=reasons)
