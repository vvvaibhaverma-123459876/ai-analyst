"""
output/alert_dispatcher.py
Sends alerts to Slack and/or email.
Configure via .env:
  SLACK_WEBHOOK_URL=https://hooks.slack.com/...
  ALERT_EMAIL=team@company.com
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
"""

from __future__ import annotations
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from core.logger import get_logger

logger = get_logger(__name__)


class AlertDispatcher:

    def __init__(self):
        self._slack_url   = os.getenv("SLACK_WEBHOOK_URL", "")
        self._alert_email = os.getenv("ALERT_EMAIL", "")
        self._smtp_host   = os.getenv("SMTP_HOST", "")
        self._smtp_port   = int(os.getenv("SMTP_PORT", "587"))
        self._smtp_user   = os.getenv("SMTP_USER", "")
        self._smtp_pass   = os.getenv("SMTP_PASS", "")

    def dispatch(self, channels: list[str], message: str, urgency: str = "high") -> dict:
        """
        Send to all configured channels.
        Returns dict of {channel: success/error}.
        """
        results = {}
        for channel in channels:
            if channel == "slack":
                results["slack"] = self._send_slack(message, urgency)
            elif channel == "email":
                results["email"] = self._send_email(message, urgency)
            elif channel == "in_app":
                results["in_app"] = "queued"   # shown in UI notification bar
                logger.info(f"In-app alert queued: {message[:80]}")
        return results

    def _send_slack(self, message: str, urgency: str) -> str:
        if not self._slack_url:
            return "skipped — SLACK_WEBHOOK_URL not set"
        try:
            import requests
            emoji = {"critical": ":rotating_light:", "high": ":warning:",
                     "medium": ":information_source:", "low": ":white_check_mark:"}
            payload = {
                "text": f"{emoji.get(urgency, ':bell:')} *AI Analyst Alert*",
                "blocks": [{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"```{message}```"},
                }],
            }
            resp = requests.post(self._slack_url, json=payload, timeout=5)
            resp.raise_for_status()
            logger.info("Slack alert sent.")
            return "sent"
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")
            return f"error: {e}"

    def _send_email(self, message: str, urgency: str) -> str:
        if not self._alert_email or not self._smtp_host:
            return "skipped — ALERT_EMAIL or SMTP_HOST not set"
        try:
            msg = MIMEMultipart()
            msg["From"]    = self._smtp_user
            msg["To"]      = self._alert_email
            msg["Subject"] = f"[{urgency.upper()}] AI Analyst Alert"
            msg.attach(MIMEText(message, "plain"))
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.starttls()
                if self._smtp_user and self._smtp_pass:
                    server.login(self._smtp_user, self._smtp_pass)
                server.send_message(msg)
            logger.info(f"Email alert sent to {self._alert_email}")
            return "sent"
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            return f"error: {e}"
