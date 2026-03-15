"""
semantic/source_authority.py  — v9
Source Authority and Conflict Resolver.

When two sources disagree (e.g. metric registry vs SQL result, data vs
external enrichment, two agents contradicting each other), this module:

  1. Looks up each source's authority level
  2. Compares freshness and completeness
  3. Chooses the winner or marks the conflict UNRESOLVED
  4. Downgrades the output confidence when unresolved

Authority levels (descending):
  primary_truth  – metric registry, policy store, governance rules
  governed_data  – connector-sourced data with quality score >= 0.7
  agent_output   – output from a single pipeline agent
  external       – web enrichment, third-party APIs
  prior          – org memory / historical patterns
  unknown        – no authority declared
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any
from core.logger import get_logger

logger = get_logger(__name__)


class Authority(IntEnum):
    primary_truth  = 5
    governed_data  = 4
    agent_output   = 3
    external       = 2
    prior          = 1
    unknown        = 0


AUTHORITY_MAP: dict[str, Authority] = {
    "metric_registry":  Authority.primary_truth,
    "policy_store":     Authority.primary_truth,
    "join_graph":       Authority.primary_truth,
    "grain_resolver":   Authority.primary_truth,
    "connector":        Authority.governed_data,
    "data_quality":     Authority.governed_data,
    "trend":            Authority.agent_output,
    "anomaly":          Authority.agent_output,
    "root_cause":       Authority.agent_output,
    "funnel":           Authority.agent_output,
    "cohort":           Authority.agent_output,
    "forecast":         Authority.agent_output,
    "experiment":       Authority.agent_output,
    "debate":           Authority.agent_output,
    "guardian":         Authority.agent_output,
    "enrichment":       Authority.external,
    "web":              Authority.external,
    "org_memory":       Authority.prior,
    "prior":            Authority.prior,
}


@dataclass
class SourceClaim:
    """One source's claim about a value or conclusion."""
    source: str
    value: Any
    confidence: float = 1.0
    freshness_ts: str = ""       # ISO timestamp — newer = preferred
    completeness: float = 1.0   # 0–1
    authority: Authority = Authority.unknown

    def __post_init__(self):
        if self.authority == Authority.unknown:
            self.authority = AUTHORITY_MAP.get(self.source.lower(), Authority.unknown)


@dataclass
class ConflictResolution:
    winner: SourceClaim | None
    loser:  SourceClaim | None
    resolved: bool
    reason: str
    confidence_penalty: float = 0.0   # subtract from output confidence
    conflict_type: str = ""            # "authority" | "freshness" | "unresolvable"


class SourceConflictResolver:
    """
    Compares two source claims and returns a resolution.

    Usage:
        resolver = SourceConflictResolver()
        resolution = resolver.resolve(claim_a, claim_b)
        final_conf = base_conf - resolution.confidence_penalty
    """

    def resolve(self, a: SourceClaim, b: SourceClaim) -> ConflictResolution:
        if a.value == b.value:
            return ConflictResolution(
                winner=a, loser=None, resolved=True,
                reason="Sources agree.", confidence_penalty=0.0,
            )

        # Step 1: authority
        if a.authority != b.authority:
            winner, loser = (a, b) if a.authority > b.authority else (b, a)
            penalty = max(0.0, 0.10 * (winner.authority - loser.authority))
            return ConflictResolution(
                winner=winner, loser=loser, resolved=True,
                reason=f"Authority: {winner.source}({winner.authority.name}) "
                       f"> {loser.source}({loser.authority.name})",
                confidence_penalty=round(penalty, 3),
                conflict_type="authority",
            )

        # Step 2: freshness (ISO timestamps — lexicographic works for ISO)
        if a.freshness_ts and b.freshness_ts and a.freshness_ts != b.freshness_ts:
            winner, loser = (a, b) if a.freshness_ts > b.freshness_ts else (b, a)
            return ConflictResolution(
                winner=winner, loser=loser, resolved=True,
                reason=f"Freshness: {winner.source} is newer.",
                confidence_penalty=0.05,
                conflict_type="freshness",
            )

        # Step 3: completeness
        if abs(a.completeness - b.completeness) > 0.15:
            winner, loser = (a, b) if a.completeness >= b.completeness else (b, a)
            return ConflictResolution(
                winner=winner, loser=loser, resolved=True,
                reason=f"Completeness: {winner.source} ({winner.completeness:.0%}) "
                       f"> {loser.source} ({loser.completeness:.0%})",
                confidence_penalty=0.08,
                conflict_type="completeness",
            )

        # Unresolvable — equal authority, same freshness/completeness, different values
        return ConflictResolution(
            winner=None, loser=None, resolved=False,
            reason=f"Unresolvable conflict between {a.source} and {b.source}: "
                   f"same authority, different values.",
            confidence_penalty=0.20,
            conflict_type="unresolvable",
        )

    def resolve_many(self, claims: list[SourceClaim]) -> tuple[SourceClaim | None, float]:
        """
        Resolve a list of claims to a single winner.
        Returns (winning_claim, total_confidence_penalty).
        """
        if not claims:
            return None, 0.0
        if len(claims) == 1:
            return claims[0], 0.0

        current = claims[0]
        total_penalty = 0.0
        for other in claims[1:]:
            resolution = self.resolve(current, other)
            total_penalty += resolution.confidence_penalty
            if resolution.winner:
                current = resolution.winner
            else:
                # Unresolved — keep highest-authority claim
                current = max([current, other], key=lambda c: c.authority)
                logger.warning(
                    "Unresolvable source conflict: %s vs %s — kept %s by authority",
                    current.source, other.source, current.source,
                )
        return current, round(min(total_penalty, 0.40), 3)
