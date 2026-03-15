"""
output/output_router.py
Urgency-aware output router.

Decides which output mode(s) to activate based on:
  - business_context.urgency (low / medium / high / critical)
  - anomaly severity and count
  - experiment significance
  - debate verdict confidence
  - explicit user preference

Output modes:
  "brief"          → structured 4-section report (always produced)
  "conversational" → opens a stateful chat thread
  "alert"          → fires Slack / email notification
  "scheduled"      → queues for next scheduled run
"""

from __future__ import annotations
from dataclasses import dataclass, field
from agents.context import AnalysisContext
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OutputDecision:
    modes: list[str]               # ["brief", "alert", "conversational"]
    urgency: str                   # low | medium | high | critical
    reason: str                    # human-readable explanation
    alert_channels: list[str] = field(default_factory=list)
    alert_message: str = ""


class OutputRouter:

    def decide(self, context: AnalysisContext) -> OutputDecision:
        """
        Analyse the finished context and decide what outputs to produce.
        Always produces at least "brief".
        """
        biz = context.business_context
        explicit_urgency = biz.get("urgency", "medium").lower()

        # Compute urgency from findings
        computed_urgency = self._compute_urgency(context)

        # Take the higher of the two
        urgency = self._max_urgency(explicit_urgency, computed_urgency)

        modes = ["brief"]
        reasons = []

        # Always open conversational mode — user can always ask follow-ups
        modes.append("conversational")

        # Alert on high / critical
        alert_channels = []
        alert_msg = ""
        if urgency in ("high", "critical"):
            modes.append("alert")
            alert_channels = self._get_alert_channels(biz)
            alert_msg = self._build_alert_message(context, urgency)
            reasons.append(f"Urgency={urgency} — alert fired to {alert_channels}")

        # Anomaly spike → alert even at medium urgency
        anom = context.results.get("anomaly")
        if (anom and anom.status == "success"
                and anom.data.get("severity_counts", {}).get("high", 0) >= 2):
            if "alert" not in modes:
                modes.append("alert")
                alert_channels = self._get_alert_channels(biz)
                alert_msg = self._build_alert_message(context, urgency)
            reasons.append(f"{anom.data['severity_counts']['high']} high-severity anomalies detected")

        # Significant experiment result → always include in brief + alert if high
        exp = context.results.get("experiment")
        if exp and exp.status == "success" and exp.data.get("significant"):
            reasons.append(
                f"A/B test significant: lift={exp.data.get('lift_pct', 0):+.1f}%"
            )

        if not reasons:
            reasons.append(f"Standard analysis complete (urgency={urgency})")

        decision = OutputDecision(
            modes=list(dict.fromkeys(modes)),   # deduplicate preserving order
            urgency=urgency,
            reason=" | ".join(reasons),
            alert_channels=alert_channels,
            alert_message=alert_msg,
        )
        logger.info(f"Output decision: {decision.modes}, urgency={urgency}")
        return decision

    # ------------------------------------------------------------------

    def _compute_urgency(self, context: AnalysisContext) -> str:
        score = 0

        # Anomaly severity
        anom = context.results.get("anomaly")
        if anom and anom.status == "success":
            sev = anom.data.get("severity_counts", {})
            score += sev.get("high", 0) * 3
            score += sev.get("medium", 0) * 1

        # Large KPI drop
        rc = context.results.get("root_cause")
        if rc and rc.status == "success":
            pct = abs(rc.data.get("pct_change", 0))
            if pct > 20:  score += 4
            elif pct > 10: score += 2
            elif pct > 5:  score += 1

        # Debate low confidence
        dbte = context.results.get("debate")
        if dbte and dbte.status == "success":
            if dbte.data.get("verdict") == "low":
                score -= 1    # lower urgency if findings unreliable

        if score >= 7:  return "critical"
        if score >= 4:  return "high"
        if score >= 2:  return "medium"
        return "low"

    def _max_urgency(self, a: str, b: str) -> str:
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return a if order.get(a, 1) >= order.get(b, 1) else b

    def _get_alert_channels(self, biz: dict) -> list[str]:
        channels = []
        if biz.get("slack_webhook"):  channels.append("slack")
        if biz.get("alert_email"):    channels.append("email")
        if not channels:              channels.append("in_app")
        return channels

    def _build_alert_message(self, context: AnalysisContext, urgency: str) -> str:
        kpi = context.kpi_col or "KPI"
        rc  = context.results.get("root_cause")
        anom = context.results.get("anomaly")

        lines = [f"[{urgency.upper()}] AI Analyst Alert — {kpi}"]

        if rc and rc.status == "success":
            delta = rc.data.get("delta", 0)
            pct   = rc.data.get("pct_change", 0)
            lines.append(f"Change: {delta:+,.0f} ({pct:+.1f}%)")

        if anom and anom.status == "success":
            n = anom.data.get("anomaly_count", 0)
            if n:
                lines.append(f"Anomalies: {n} detected")

        brief_first_line = context.final_brief.splitlines()[0] if context.final_brief else ""
        if brief_first_line:
            lines.append(brief_first_line[:120])

        lines.append("→ Open AI Analyst for full details.")
        return "\n".join(lines)
