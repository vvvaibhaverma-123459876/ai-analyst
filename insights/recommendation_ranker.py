"""
insights/recommendation_ranker.py  — v9
Evidence-driven recommendation ranker.

v9 changes (Phase 7):
  - Candidates carry source, reason, supporting_evidence, reversibility
  - Low-evidence recommendations auto-penalised in score
  - Rank is now evidence_quality × (confidence + urgency + value) − effort
  - to_dict() returns full provenance for audit
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class RankedRecommendation:
    action: str
    source: str             = ""     # which analysis module generated this
    reason: str             = ""     # why this action is recommended
    supporting_evidence: list[str] = field(default_factory=list)
    confidence: float       = 0.5
    urgency: float          = 0.5
    business_value: float   = 0.5
    effort: float           = 0.5
    reversibility: float    = 0.5   # 1.0 = fully reversible, 0.0 = irreversible
    evidence_quality: float = 1.0   # penalty multiplier from evidence grader
    score: float            = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RecommendationRanker:
    """
    Scores and ranks recommendation candidates.

    Score formula (v9):
        raw  = 0.35·confidence + 0.30·urgency + 0.30·business_value − 0.15·effort
        adj  = raw × evidence_quality   ← low evidence drags the whole score down
        final = clamp(adj, 0.0, 1.0)

    This means:
      - A high-confidence, high-urgency action still loses to a moderate one
        if its evidence quality is poor.
      - Irreversible actions are NOT penalised in the score — reversibility is
        surfaced in the output for the human to weigh.
    """

    def rank(self, actions: list[dict]) -> list[RankedRecommendation]:
        ranked: list[RankedRecommendation] = []
        for action in actions:
            confidence      = float(action.get("confidence",      0.5))
            urgency         = float(action.get("urgency",         0.5))
            business_value  = float(action.get("business_value",  0.5))
            effort          = float(action.get("effort",          0.5))
            evidence_quality= float(action.get("evidence_quality",1.0))
            reversibility   = float(action.get("reversibility",   0.5))

            raw   = 0.35 * confidence + 0.30 * urgency + 0.30 * business_value - 0.15 * effort
            adj   = raw * max(0.0, min(1.0, evidence_quality))
            score = round(max(0.0, min(1.0, adj)), 4)

            ranked.append(RankedRecommendation(
                action=action.get("action", ""),
                source=action.get("source", ""),
                reason=action.get("reason", ""),
                supporting_evidence=action.get("supporting_evidence", []),
                confidence=confidence,
                urgency=urgency,
                business_value=business_value,
                effort=effort,
                reversibility=reversibility,
                evidence_quality=evidence_quality,
                score=score,
            ))

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked
