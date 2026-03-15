from __future__ import annotations
from dataclasses import dataclass

@dataclass
class EvidenceGrade:
    grade: str
    summary_strength: float
    reasons: list[str]

class EvidenceGrader:
    def grade(self, support_ratio: float, confidence: float, contradictions: int, data_quality_score: float | None = None) -> EvidenceGrade:
        score = 0.45 * support_ratio + 0.45 * confidence + 0.10 * (data_quality_score if data_quality_score is not None else 0.7)
        score -= min(0.2, contradictions * 0.08)
        reasons = []
        if contradictions:
            reasons.append(f'{contradictions} contradiction(s) reduced evidence strength')
        if data_quality_score is not None and data_quality_score < 0.6:
            reasons.append('data quality reduced evidence strength')
        if score >= 0.8:
            grade = 'strong'
        elif score >= 0.65:
            grade = 'moderate'
        elif score >= 0.45:
            grade = 'weak'
        else:
            grade = 'speculative'
        return EvidenceGrade(grade, round(max(score, 0.0), 3), reasons)
