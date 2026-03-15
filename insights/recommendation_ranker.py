from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass
class RankedRecommendation:
    action: str
    confidence: float
    urgency: float
    business_value: float
    effort: float
    score: float
    def to_dict(self) -> dict:
        return asdict(self)

class RecommendationRanker:
    def rank(self, actions: list[dict]) -> list[RankedRecommendation]:
        ranked: list[RankedRecommendation] = []
        for action in actions:
            confidence = float(action.get('confidence', 0.5))
            urgency = float(action.get('urgency', 0.5))
            business_value = float(action.get('business_value', 0.5))
            effort = float(action.get('effort', 0.5))
            score = 0.35 * confidence + 0.30 * urgency + 0.30 * business_value - 0.15 * effort
            ranked.append(RankedRecommendation(action=action['action'], confidence=confidence, urgency=urgency, business_value=business_value, effort=effort, score=round(score, 3)))
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked
