from __future__ import annotations

"""Structured evidence model for hypothesis closing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceState(str, Enum):
    SUPPORTED = "supported"
    WEAKENED = "weakened"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    UNTESTABLE = "untestable"


@dataclass(slots=True)
class EvidenceRecord:
    hypothesis_id: str
    agent: str
    summary: str
    supports: bool | None
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceSummary:
    state: EvidenceState
    support_weight: float
    oppose_weight: float
    net_support: float
    confidence: float
    reasons: list[str] = field(default_factory=list)


class EvidenceRegistry:
    def summarise(self, records: list[EvidenceRecord], missing_data: list[str] | None = None) -> EvidenceSummary:
        missing_data = missing_data or []
        if missing_data and not records:
            return EvidenceSummary(
                state=EvidenceState.UNTESTABLE,
                support_weight=0.0,
                oppose_weight=0.0,
                net_support=0.0,
                confidence=0.0,
                reasons=[f"Missing data: {', '.join(missing_data)}"],
            )

        support = sum(max(0.0, min(1.0, r.confidence)) for r in records if r.supports is True)
        oppose = sum(max(0.0, min(1.0, r.confidence)) for r in records if r.supports is False)
        neutral = [r for r in records if r.supports is None]
        total = support + oppose
        if total == 0:
            state = EvidenceState.INCONCLUSIVE if records or neutral else EvidenceState.UNTESTABLE
            return EvidenceSummary(
                state=state,
                support_weight=support,
                oppose_weight=oppose,
                net_support=0.0,
                confidence=0.0,
                reasons=["No directional evidence available."],
            )

        net = (support - oppose) / total
        confidence = abs(net)
        if net >= 0.35:
            state = EvidenceState.SUPPORTED
        elif net <= -0.35:
            state = EvidenceState.CONTRADICTED
        else:
            state = EvidenceState.INCONCLUSIVE

        reasons = [r.summary for r in sorted(records, key=lambda x: x.confidence, reverse=True)[:3]]
        if neutral:
            reasons.append(f"{len(neutral)} source(s) provided neutral or non-directional evidence.")

        return EvidenceSummary(
            state=state,
            support_weight=round(support, 3),
            oppose_weight=round(oppose, 3),
            net_support=round(net, 3),
            confidence=round(confidence, 3),
            reasons=reasons,
        )
