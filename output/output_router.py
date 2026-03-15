"""
output/output_router.py
Urgency-aware output router with security-aware output classification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from agents.context import AnalysisContext
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OutputDecision:
    modes: list[str]
    urgency: str
    reason: str
    alert_channels: list[str] = field(default_factory=list)
    alert_message: str = ""
    output_classification: str = "INTERNAL"


class OutputRouter:

    def decide(self, context: AnalysisContext) -> OutputDecision:
        biz = context.business_context
        explicit_urgency = biz.get("urgency", "medium").lower()
        computed_urgency = self._compute_urgency(context)
        urgency = self._max_urgency(explicit_urgency, computed_urgency)

        modes = ["brief", "conversational"]
        reasons = []

        alert_channels = []
        alert_msg = ""
        if urgency in ("high", "critical"):
            modes.append("alert")
            alert_channels = self._get_alert_channels(biz)
            alert_msg = self._build_alert_message(context, urgency)
            reasons.append(f"Urgency={urgency} — alert fired to {alert_channels}")

        anom = context.results.get("anomaly")
        if (anom and anom.status == "success" and anom.data.get("severity_counts", {}).get("high", 0) >= 2):
            if "alert" not in modes:
                modes.append("alert")
                alert_channels = self._get_alert_channels(biz)
                alert_msg = self._build_alert_message(context, urgency)
            reasons.append(f"{anom.data['severity_counts']['high']} high-severity anomalies detected")

        exp = context.results.get("experiment")
        if exp and exp.status == "success" and exp.data.get("significant"):
            reasons.append(f"A/B test significant: lift={exp.data.get('lift_pct', 0):+.1f}%")

        if not reasons:
            reasons.append(f"Standard analysis complete (urgency={urgency})")

        classification = 'INTERNAL'
        if getattr(context, 'security_shell', None):
            try:
                classification = context.security_shell.classify_output({
                    'brief': context.final_brief,
                    'follow_up_questions': context.follow_up_questions,
                    'recommendations': context.recommendation_candidates,
                })
            except Exception:
                classification = 'INTERNAL'

        decision = OutputDecision(
            modes=list(dict.fromkeys(modes)),
            urgency=urgency,
            reason=" | ".join(reasons),
            alert_channels=alert_channels,
            alert_message=alert_msg,
            output_classification=classification,
        )
        logger.info(f"Output decision: {decision.modes}, urgency={urgency}, classification={classification}")
        return decision

    def _compute_urgency(self, context: AnalysisContext) -> str:
        score = 0
        anom = context.results.get("anomaly")
        if anom and anom.status == "success":
            sev = anom.data.get("severity_counts", {})
            score += sev.get("high", 0) * 3
            score += sev.get("medium", 0) * 1
        rc = context.results.get("root_cause")
        if rc and rc.status == "success":
            pct = abs(float(rc.data.get('pct_change', 0) or 0))
            if pct >= 20: score += 4
            elif pct >= 10: score += 2
        if score >= 6: return 'critical'
        if score >= 4: return 'high'
        if score >= 2: return 'medium'
        return 'low'

    def _max_urgency(self, a: str, b: str) -> str:
        order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        return a if order.get(a, 1) >= order.get(b, 1) else b

    def _get_alert_channels(self, biz: dict) -> list[str]:
        return biz.get('alert_channels', ['email'])

    def _build_alert_message(self, context: AnalysisContext, urgency: str) -> str:
        return f"[{urgency.upper()}] {context.kpi_col or 'KPI'} requires attention."
