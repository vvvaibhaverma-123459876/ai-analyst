"""
output/conversation_engine.py
Stateful conversational engine for post-analysis follow-up.

After the pipeline runs, the user can ask natural language questions
about the results. This engine:
  1. Maintains conversation history
  2. Has full access to the AnalysisContext (all agent results)
  3. Answers questions by reasoning over findings + re-querying data
  4. Can trigger targeted re-runs of specific agents on demand
  5. Never forgets what was already analysed this session
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from agents.context import AnalysisContext
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)

_CONVERSATION_SYSTEM = """You are a senior data analyst in a live conversation.
You have access to a completed analysis. The user can ask any follow-up question.

Your capabilities:
- Answer questions about the data findings directly
- Explain methodology (why certain agents ran, what methods were used)
- Suggest what to investigate next
- Re-interpret findings from a different angle
- Flag if a question requires re-running analysis on a different segment

Rules:
- Use ONLY facts from the analysis context provided
- If you cannot answer from context, say so and suggest what data is needed
- Be conversational but precise — use numbers when available
- Keep responses to 3-6 sentences unless asked to elaborate
- If the user asks for a chart or table, describe what it would show

Analysis context summary is injected in each message."""


@dataclass
class ConversationTurn:
    role: str    # "user" | "assistant"
    content: str


class ConversationEngine:

    def __init__(self, context: AnalysisContext):
        self._context = context
        self._history: list[ConversationTurn] = []
        self._context_summary = self._build_context_summary()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Send a user message, get an assistant response.
        Maintains full conversation history.
        """
        self._history.append(ConversationTurn(role="user", content=user_message))

        # Check for special commands
        response = self._handle_special_commands(user_message)
        if response is None:
            response = self._llm_response(user_message)

        self._history.append(ConversationTurn(role="assistant", content=response))
        logger.info(f"Conversation turn {len(self._history)//2}: {user_message[:60]}")
        return response

    def reset(self):
        """Clear conversation history (keep context)."""
        self._history = []

    # ------------------------------------------------------------------
    # Special commands (no LLM needed)
    # ------------------------------------------------------------------

    def _handle_special_commands(self, msg: str) -> str | None:
        msg_lower = msg.lower().strip()

        if any(w in msg_lower for w in ["show agents", "which agents", "what ran"]):
            plan = self._context.active_agents
            statuses = {
                name: self._context.results.get(name, None)
                for name in plan
            }
            lines = [f"**{name}**: {r.status if r else 'not run'}"
                     for name, r in statuses.items()]
            return "Agents that ran this session:\n" + "\n".join(lines)

        if any(w in msg_lower for w in ["show anomalies", "list anomalies", "what anomalies"]):
            anom = self._context.results.get("anomaly")
            if anom and anom.status == "success":
                records = anom.data.get("anomaly_records", [])
                if records:
                    lines = [f"- {r['date']}: {r['value']:,.2f}" for r in records[:10]]
                    return f"Anomalies detected ({len(records)} total):\n" + "\n".join(lines)
            return "No anomalies were detected in this analysis."

        if any(w in msg_lower for w in ["show drivers", "top drivers", "what drove"]):
            rc = self._context.results.get("root_cause")
            if rc and rc.status == "success":
                movers = rc.data.get("movers", {})
                pos = movers.get("positive", [])
                neg = movers.get("negative", [])
                lines = ["**Positive drivers:**"]
                for d in pos[:3]:
                    lines.append(f"  {d.get('dimension')}={d.get('value')}: {d.get('delta'):+,.0f}")
                lines.append("**Negative drivers:**")
                for d in neg[:3]:
                    lines.append(f"  {d.get('dimension')}={d.get('value')}: {d.get('delta'):+,.0f}")
                return "\n".join(lines)
            return "Root cause analysis was not run or found no drivers."

        if any(w in msg_lower for w in ["show forecast", "what is forecast", "prediction"]):
            fcst = self._context.results.get("forecast")
            if fcst and fcst.status == "success":
                return (
                    f"Forecast ({fcst.data.get('method')}, "
                    f"{fcst.data.get('horizon')} periods): "
                    f"{self._context.kpi_col} expected to go "
                    f"{fcst.data.get('direction')} by "
                    f"~{fcst.data.get('pct_change', 0):.1f}% "
                    f"(from {fcst.data.get('last_actual', 0):,.1f} "
                    f"to {fcst.data.get('last_forecast', 0):,.1f})."
                )
            return "Forecasting was not run for this dataset."

        if any(w in msg_lower for w in ["debate", "confidence", "how reliable", "trust"]):
            dbte = self._context.results.get("debate")
            if dbte and dbte.status == "success":
                verdict = dbte.data.get("verdict", "unknown")
                flags = dbte.data.get("red_flags", [])
                challenges = dbte.data.get("challenges", [])
                lines = [f"Narrative confidence: **{verdict}**"]
                if flags:
                    lines.append("Red flags: " + " | ".join(flags[:3]))
                if challenges:
                    lines.append(f"\nMain challenge: {challenges[0].get('challenge', '')}")
                return "\n".join(lines)
            return "No debate review was run."

        if "summarise" in msg_lower or "summarize" in msg_lower or "brief" in msg_lower:
            return self._context.final_brief[:1500] if self._context.final_brief else "No brief generated yet."

        return None   # fall through to LLM

    # ------------------------------------------------------------------
    # LLM response with full context injection
    # ------------------------------------------------------------------

    def _llm_response(self, user_message: str) -> str:
        if not (config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY):
            return self._rule_response(user_message)
        try:
            from llm.client import LLMClient
            llm = LLMClient()

            # Build message list with history
            messages = self._build_messages(user_message)

            # Use multi-turn if provider supports it, otherwise inject history
            return llm.complete(
                system=_CONVERSATION_SYSTEM + "\n\n" + self._context_summary,
                user=self._history_as_text() + f"\n\nUser: {user_message}",
            )
        except Exception as e:
            logger.warning(f"Conversation LLM failed: {e}")
            return self._rule_response(user_message)

    def _rule_response(self, msg: str) -> str:
        msg_lower = msg.lower()
        kpi = self._context.kpi_col or "the KPI"
        if "why" in msg_lower or "cause" in msg_lower or "reason" in msg_lower:
            rc = self._context.results.get("root_cause")
            if rc and rc.status == "success":
                return rc.summary
            return f"Root cause analysis wasn't run. Try re-uploading with categorical dimensions."
        if "trend" in msg_lower or "direction" in msg_lower:
            t = self._context.results.get("trend")
            if t and t.status == "success":
                return t.summary
        if "next" in msg_lower or "recommend" in msg_lower or "action" in msg_lower:
            if self._context.follow_up_questions:
                return "Recommended next questions:\n" + "\n".join(
                    f"- {q}" for q in self._context.follow_up_questions
                )
        return (
            f"Based on the analysis, {kpi} showed: {self._context_summary[:300]}. "
            f"Ask me about anomalies, drivers, forecast, segments, or the full brief."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context_summary(self) -> str:
        ctx = self._context
        lines = [f"KPI: {ctx.kpi_col}", f"Dataset: {len(ctx.df):,} rows"]
        if ctx.business_context:
            biz = ctx.business_context
            lines.append(f"Business: {biz.get('company', 'unknown')} | "
                         f"Goal: {biz.get('primary_goal', 'unknown')} | "
                         f"Audience: {biz.get('audience', 'unknown')}")
        lines.append(f"Agents run: {', '.join(ctx.active_agents)}")
        for name, result in ctx.results.items():
            if result.status == "success" and name not in ("orchestrator",):
                lines.append(f"{name}: {result.summary[:100]}")
        return "\n".join(lines)

    def _history_as_text(self) -> str:
        turns = self._history[:-1]   # exclude latest user message
        if not turns:
            return ""
        lines = []
        for turn in turns[-6:]:    # last 3 exchanges
            lines.append(f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}")
        return "\n".join(lines)

    def _build_messages(self, user_message: str) -> list[dict]:
        messages = []
        for turn in self._history[-6:]:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    @property
    def history(self) -> list[ConversationTurn]:
        return self._history
