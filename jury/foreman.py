"""
jury/foreman.py
Foreman — reconciles juror verdicts using the foreman protocol.

Protocol:
  Unanimous  (all agree)   → high confidence, promote finding
  Majority   (>50% agree)  → medium confidence, note minority
  Split      (50/50)       → report disagreement as a finding itself
  No consensus             → halt, request more data, do not publish

The disagreement IS the finding in split/no-consensus cases —
it surfaces uncertainty that would be hidden by a single method.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from jury.base_juror import BaseJuror, JurorVerdict
from agents.context import AnalysisContext, AgentResult
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ForemanVerdict:
    consensus: str              # "unanimous" | "majority" | "split" | "none"
    confidence: float
    primary_finding: dict
    minority_finding: dict | None
    all_verdicts: list[JurorVerdict]
    summary: str
    publish: bool               # False = halt pipeline for this finding


class Foreman:

    def __init__(self, jury_name: str, jurors: list[BaseJuror], max_workers: int = 4):
        self._jury_name = jury_name
        self._jurors = jurors
        self._max_workers = max_workers

    def deliberate(
        self,
        context: AnalysisContext,
        agreement_fn: Callable[[list[JurorVerdict]], bool] = None,
    ) -> ForemanVerdict:
        """
        Run all jurors in parallel, then reconcile their verdicts.
        agreement_fn: optional custom function to determine if two verdicts agree.
        """
        verdicts = self._run_jurors(context)
        successful = [v for v in verdicts if v.status == "success"]

        if not successful:
            return ForemanVerdict(
                consensus="none",
                confidence=0.0,
                primary_finding={},
                minority_finding=None,
                all_verdicts=verdicts,
                summary=f"{self._jury_name}: no jurors produced valid output.",
                publish=False,
            )

        # Group by agreement
        groups = self._group_by_agreement(successful, agreement_fn)
        largest_group = max(groups, key=len)
        minority_group = [g for g in groups if g != largest_group]
        minority = minority_group[0] if minority_group else []

        n_total = len(successful)
        n_agree = len(largest_group)
        agree_ratio = n_agree / n_total

        # Consensus level
        if agree_ratio == 1.0:
            consensus = "unanimous"
            confidence = round(sum(v.confidence for v in largest_group) / n_agree, 3)
            publish = True
        elif agree_ratio > 0.5:
            consensus = "majority"
            confidence = round(
                (sum(v.confidence for v in largest_group) / n_agree) * agree_ratio, 3
            )
            publish = True
        elif agree_ratio == 0.5:
            consensus = "split"
            confidence = 0.4
            publish = True  # publish WITH the disagreement flagged
        else:
            consensus = "none"
            confidence = 0.2
            publish = False

        # Primary finding = merged from majority
        primary = self._merge_findings(largest_group)

        # Minority finding
        minority_finding = self._merge_findings(minority) if minority else None

        # Build summary
        disagree_note = ""
        if minority:
            disagree_note = (
                f" Minority ({len(minority)}/{n_total}): "
                f"{minority[0].summary[:60]}"
            )

        verdict_str = {
            "unanimous": f"All {n_total} jurors agree",
            "majority":  f"{n_agree}/{n_total} jurors agree",
            "split":     f"Split verdict — {n_agree}/{n_total} for primary finding",
            "none":      f"No consensus — {n_total} jurors disagree",
        }[consensus]

        summary = (
            f"{self._jury_name} jury: {verdict_str} "
            f"(confidence={confidence:.2f}).{disagree_note}"
        )

        logger.info(f"[{self._jury_name} foreman] {consensus} — conf={confidence:.2f}")

        return ForemanVerdict(
            consensus=consensus,
            confidence=confidence,
            primary_finding=primary,
            minority_finding=minority_finding,
            all_verdicts=verdicts,
            summary=summary,
            publish=publish,
        )

    def _run_jurors(self, context: AnalysisContext) -> list[JurorVerdict]:
        verdicts = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(juror.deliberate, context): juror.name
                for juror in self._jurors
            }
            for future in as_completed(futures):
                juror_name = futures[future]
                try:
                    v = future.result()
                    verdicts.append(v)
                    logger.info(f"  [{juror_name}] {v.status} conf={v.confidence:.2f}")
                except Exception as e:
                    logger.warning(f"  [{juror_name}] error: {e}")
                    verdicts.append(JurorVerdict(
                        juror_name=juror_name, method="unknown",
                        finding={}, confidence=0.0,
                        summary=f"Error: {e}", status="error", error=str(e),
                    ))
        return verdicts

    def _group_by_agreement(
        self,
        verdicts: list[JurorVerdict],
        agreement_fn: Callable | None,
    ) -> list[list[JurorVerdict]]:
        """Cluster verdicts by directional agreement."""
        if agreement_fn:
            # Custom agreement function
            groups = []
            for v in verdicts:
                placed = False
                for g in groups:
                    if agreement_fn([g[0], v]):
                        g.append(v)
                        placed = True
                        break
                if not placed:
                    groups.append([v])
            return groups

        # Default: group by primary finding direction
        pos_group, neg_group, neutral_group = [], [], []
        pos_words = {"increase", "up", "rise", "high", "anomaly detected",
                     "significant", "confirmed", "found"}
        neg_words = {"decrease", "down", "drop", "low", "no anomaly",
                     "not significant", "rejected", "none"}

        for v in verdicts:
            s = v.summary.lower()
            if any(w in s for w in pos_words):
                pos_group.append(v)
            elif any(w in s for w in neg_words):
                neg_group.append(v)
            else:
                neutral_group.append(v)

        groups = [g for g in [pos_group, neg_group, neutral_group] if g]
        return groups if groups else [verdicts]

    def _merge_findings(self, verdicts: list[JurorVerdict]) -> dict:
        """Merge structured findings from a group of agreeing jurors."""
        if not verdicts:
            return {}
        merged = {}
        for v in verdicts:
            for k, val in v.finding.items():
                if k not in merged:
                    merged[k] = val
                elif isinstance(val, (int, float)) and isinstance(merged[k], (int, float)):
                    merged[k] = (merged[k] + val) / 2    # average numerics
        merged["_jurors"] = [v.juror_name for v in verdicts]
        merged["_consensus_confidence"] = round(
            sum(v.confidence for v in verdicts) / len(verdicts), 3
        )
        return merged

    def to_agent_result(self, agent_name: str, fv: ForemanVerdict) -> AgentResult:
        """Convert a ForemanVerdict to the standard AgentResult interface."""
        status = "success" if fv.publish else "error"
        return AgentResult(
            agent=agent_name,
            status=status,
            summary=fv.summary,
            data={
                "consensus": fv.consensus,
                "confidence": fv.confidence,
                "finding": fv.primary_finding,
                "minority_finding": fv.minority_finding,
                "juror_count": len(fv.all_verdicts),
                "publish": fv.publish,
                **fv.primary_finding,
            },
            error="" if fv.publish else f"No consensus — publishing blocked ({fv.consensus})",
        )
