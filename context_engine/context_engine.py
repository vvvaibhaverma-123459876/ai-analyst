"""
context_engine/context_engine.py
The "ask upfront" system.

Flow:
  1. Load org memory (what we already know)
  2. Analyse the ParsedDocument to understand what questions are needed
  3. Generate targeted questions (only what we don't already know)
  4. Accept answers and enrich AnalysisContext.business_context
  5. Save new knowledge back to OrgMemory

The context engine runs BEFORE the orchestrator.
Its output enriches every subsequent agent's work.
"""

from __future__ import annotations
import json
from context_engine.org_memory import OrgMemory
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)

_QUESTION_SYSTEM = """You are an analytics lead preparing to analyse a dataset.
Given the data profile and existing org context, generate the minimum set of
clarifying questions needed BEFORE analysis begins.

Rules:
- Only ask what you cannot infer from the data itself
- Skip questions already answered in org context
- Max 5 questions total
- Focus on: business goal, primary KPI definition, key audience, urgency level, known segments
- Return ONLY a JSON list of question strings. No explanation. No markdown."""

_ENRICH_SYSTEM = """You are an analytics context builder.
Given user answers to clarifying questions, extract structured business context.
Return ONLY valid JSON with these keys (use null if not determinable):
{
  "company": string,
  "industry": string,
  "primary_goal": string,
  "kpi_name": string,
  "kpi_definition": string,
  "audience": string,
  "urgency": "low|medium|high|critical",
  "known_segments": list of strings,
  "business_notes": string
}"""


class ContextEngine:

    def __init__(self, org_memory: OrgMemory = None):
        self._memory = org_memory or OrgMemory()

    # ------------------------------------------------------------------
    # Step 1: Generate questions
    # ------------------------------------------------------------------

    def generate_questions(
        self,
        data_profile: dict,
        document_summary: str = "",
    ) -> list[str]:
        """
        Returns a list of questions to ask the user upfront.
        Returns empty list if org memory already covers everything.
        """
        existing = self._memory.to_prompt_context()

        # If we have rich existing context, ask nothing or very little
        ctx = self._memory.get_all_context()
        already_known = {
            "company": ctx.get("company"),
            "primary_goal": ctx.get("primary_goal"),
            "urgency": ctx.get("urgency"),
        }
        fully_known = all(v is not None for v in already_known.values())
        if fully_known and len(ctx) >= 5:
            logger.info("Org context sufficient — skipping upfront questions.")
            return []

        if not (config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY):
            return self._rule_based_questions(data_profile, ctx)

        try:
            from llm.client import LLMClient
            llm = LLMClient()
            user_prompt = (
                f"Existing org context:\n{existing}\n\n"
                f"Data profile:\n{json.dumps(data_profile, indent=2, default=str)}\n\n"
                f"Document summary: {document_summary}\n\n"
                "What questions must be answered before analysis?"
            )
            raw = llm.complete(system=_QUESTION_SYSTEM, user=user_prompt)
            raw = raw.strip().strip("```json").strip("```").strip()
            questions = json.loads(raw)
            if isinstance(questions, list):
                logger.info(f"Generated {len(questions)} upfront questions.")
                return [str(q) for q in questions[:5]]
        except Exception as e:
            logger.warning(f"LLM question generation failed: {e}")

        return self._rule_based_questions(data_profile, ctx)

    def _rule_based_questions(self, data_profile: dict, existing: dict) -> list[str]:
        questions = []
        if not existing.get("primary_goal"):
            questions.append(
                "What is the primary business goal of this analysis? "
                "(e.g. understand a drop, monitor growth, prepare a report)"
            )
        if not existing.get("kpi_name") and data_profile.get("kpis"):
            kpis = ", ".join(data_profile["kpis"][:4])
            questions.append(
                f"Which KPI should be the focus? Detected candidates: {kpis}"
            )
        if not existing.get("audience"):
            questions.append(
                "Who is the primary audience for this analysis? "
                "(e.g. CEO, product team, engineering, investors)"
            )
        if not existing.get("urgency"):
            questions.append(
                "What is the urgency? "
                "(low = weekly report, medium = daily review, high = incident, critical = live issue)"
            )
        if not existing.get("company"):
            questions.append(
                "What company or product does this data relate to? "
                "(helps calibrate industry benchmarks and terminology)"
            )
        return questions[:5]

    # ------------------------------------------------------------------
    # Step 2: Enrich context from answers
    # ------------------------------------------------------------------

    def enrich_from_answers(
        self,
        questions: list[str],
        answers: list[str],
    ) -> dict:
        """
        Takes Q&A pairs, extracts structured business context,
        saves to OrgMemory, and returns the enriched context dict.
        """
        if not questions or not answers:
            return self._memory.get_all_context()

        qa_text = "\n".join(
            f"Q: {q}\nA: {a}"
            for q, a in zip(questions, answers)
            if a and a.strip()
        )

        enriched = {}

        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            try:
                from llm.client import LLMClient
                llm = LLMClient()
                raw = llm.complete(
                    system=_ENRICH_SYSTEM,
                    user=f"Q&A pairs:\n{qa_text}"
                )
                raw = raw.strip().strip("```json").strip("```").strip()
                enriched = json.loads(raw)
            except Exception as e:
                logger.warning(f"LLM enrichment failed, using raw answers: {e}")
                enriched = self._parse_answers_directly(questions, answers)
        else:
            enriched = self._parse_answers_directly(questions, answers)

        # Save to org memory
        for key, value in enriched.items():
            if value is not None:
                self._memory.set(key, value)

        # Save KPI definition if present
        if enriched.get("kpi_name") and enriched.get("kpi_definition"):
            self._memory.save_kpi(
                name=enriched["kpi_name"],
                definition=enriched["kpi_definition"],
            )

        logger.info(f"Context enriched with {len(enriched)} fields.")
        return {**self._memory.get_all_context(), **enriched}

    def _parse_answers_directly(self, questions: list[str], answers: list[str]) -> dict:
        """Simple fallback: store raw answers keyed by question keywords."""
        result = {}
        for q, a in zip(questions, answers):
            if not a or not a.strip():
                continue
            q_lower = q.lower()
            if "goal" in q_lower or "objective" in q_lower:
                result["primary_goal"] = a
            elif "kpi" in q_lower or "metric" in q_lower or "focus" in q_lower:
                result["kpi_name"] = a
            elif "audience" in q_lower or "who" in q_lower:
                result["audience"] = a
            elif "urgency" in q_lower or "priority" in q_lower:
                result["urgency"] = a.lower()
            elif "company" in q_lower or "product" in q_lower or "business" in q_lower:
                result["company"] = a
        return result

    # ------------------------------------------------------------------
    # Load full context for agent pipeline
    # ------------------------------------------------------------------

    def load_context(self) -> dict:
        """Returns complete current business context from memory."""
        return self._memory.get_all_context()

    def context_summary_for_llm(self) -> str:
        """Returns formatted context string for LLM prompt injection."""
        return self._memory.to_prompt_context()
