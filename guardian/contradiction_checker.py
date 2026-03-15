from __future__ import annotations

"""Cross-agent contradiction detection helpers."""

from typing import Iterable

from agents.context import AgentResult

POSITIVE_WORDS = {"increase", "increased", "up", "rise", "growth", "grew", "improved", "higher"}
NEGATIVE_WORDS = {"decrease", "decreased", "down", "drop", "dropped", "decline", "declined", "lower", "failure", "worse"}


class ContradictionChecker:
    def direction(self, result: AgentResult | None) -> str:
        if result is None:
            return "unknown"
        text = (result.summary or "").lower()
        pos = any(w in text for w in POSITIVE_WORDS)
        neg = any(w in text for w in NEGATIVE_WORDS)
        if pos and not neg:
            return "positive"
        if neg and not pos:
            return "negative"
        return "mixed"

    def detect(self, results: dict[str, AgentResult], focus_agents: Iterable[str] | None = None) -> list[dict]:
        names = list(focus_agents) if focus_agents else list(results.keys())
        contradictions: list[dict] = []
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                d1 = self.direction(results.get(left))
                d2 = self.direction(results.get(right))
                if {d1, d2} == {"positive", "negative"}:
                    contradictions.append({
                        "left_agent": left,
                        "right_agent": right,
                        "left_summary": results[left].summary,
                        "right_summary": results[right].summary,
                        "reason": "Opposite directional findings on the same run.",
                    })
        return contradictions
