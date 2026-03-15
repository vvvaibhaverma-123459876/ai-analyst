"""
agents/nlp_agent.py
NLP Agent — analyses unstructured text in the dataset.

Handles:
  - Text columns in a DataFrame (reviews, notes, feedback, comments)
  - Free-text documents from ParsedDocument.text_chunks
  - Sentiment scoring (positive / negative / neutral)
  - Topic extraction / keyword clustering
  - Entity extraction (names, dates, numbers, organisations)
  - Summary generation via LLM
"""

from __future__ import annotations
import re
import pandas as pd
from collections import Counter
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)

TEXT_KEYWORDS = ["text", "comment", "note", "review", "feedback",
                 "description", "message", "reason", "remarks", "summary"]

_NLP_SYSTEM = """You are a text analytics specialist.
Given a sample of text data, provide:
1. Overall sentiment (positive/negative/neutral with % breakdown)
2. Top 5 themes or topics
3. Key entities mentioned (people, companies, places, products)
4. 2-3 sentence business insight
Keep response concise and factual."""


class NLPAgent(BaseAgent):
    name = "nlp"
    description = "Text sentiment, topic extraction, entity recognition from text columns"

    def _run(self, context: AnalysisContext) -> AgentResult:
        df = context.df
        doc = context.document

        text_col = self._detect_text_col(df)
        doc_text = ""
        if doc and doc.has_text:
            doc_text = doc.all_text[:5000]

        if text_col is None and not doc_text:
            return self.skip("No text columns or document text found.")

        findings = {}
        all_text_sample = ""

        # ── DataFrame text column ──────────────────────────────────
        if text_col:
            texts = df[text_col].dropna().astype(str).tolist()
            all_text_sample = "\n".join(texts[:200])

            findings["text_col"] = text_col
            findings["total_entries"] = len(texts)
            findings["avg_length"] = round(
                sum(len(t) for t in texts) / len(texts), 1
            ) if texts else 0

            # Basic sentiment (rule-based fast path)
            sentiment_counts = self._rule_sentiment(texts[:500])
            findings["sentiment"] = sentiment_counts

            # Top keywords
            findings["top_keywords"] = self._top_keywords(texts[:500])

        # ── Document text ─────────────────────────────────────────
        if doc_text:
            findings["document_text_length"] = len(doc_text)
            all_text_sample = (all_text_sample + "\n\n" + doc_text)[:5000]

        # ── LLM deep analysis ─────────────────────────────────────
        llm_analysis = ""
        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            llm_analysis = self._llm_analyse(all_text_sample, context)
            findings["llm_analysis"] = llm_analysis

        # ── Summary ──────────────────────────────────────────────
        if text_col:
            sent = findings.get("sentiment", {})
            kw = findings.get("top_keywords", [])
            dominant = max(sent, key=sent.get) if sent else "unknown"
            summary = (
                f"Text analysis on '{text_col}' ({findings['total_entries']} entries): "
                f"dominant sentiment = {dominant} "
                f"({sent.get(dominant, 0):.0%}). "
                f"Top keywords: {', '.join(kw[:5])}. "
            )
        else:
            summary = f"Document text analysed ({len(doc_text)} chars). "

        if llm_analysis:
            first_line = llm_analysis.splitlines()[0][:120] if llm_analysis else ""
            summary += first_line

        return AgentResult(
            agent=self.name,
            status="success",
            summary=summary,
            data={
                "text_col": text_col,
                "findings": findings,
                "llm_analysis": llm_analysis,
                "top_keywords": findings.get("top_keywords", []),
                "sentiment": findings.get("sentiment", {}),
            },
        )

    def _detect_text_col(self, df: pd.DataFrame) -> str | None:
        for col in df.select_dtypes(include="object").columns:
            if any(kw in col.lower() for kw in TEXT_KEYWORDS):
                avg_len = df[col].dropna().astype(str).str.len().mean()
                if avg_len > 20:
                    return col
        for col in df.select_dtypes(include="object").columns:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > 40 and df[col].nunique() > 10:
                return col
        return None

    def _rule_sentiment(self, texts: list[str]) -> dict:
        pos_words = {"good", "great", "excellent", "positive", "happy", "best",
                     "love", "perfect", "amazing", "fantastic", "increase", "growth",
                     "improve", "success", "gain", "up", "profit", "win"}
        neg_words = {"bad", "poor", "negative", "issue", "problem", "fail", "loss",
                     "decrease", "drop", "decline", "error", "wrong", "terrible",
                     "worst", "down", "concern", "risk", "missed"}
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for text in texts:
            words = set(re.findall(r"\b\w+\b", text.lower()))
            pos = len(words & pos_words)
            neg = len(words & neg_words)
            if pos > neg:
                counts["positive"] += 1
            elif neg > pos:
                counts["negative"] += 1
            else:
                counts["neutral"] += 1
        total = sum(counts.values()) or 1
        return {k: round(v / total, 3) for k, v in counts.items()}

    def _top_keywords(self, texts: list[str], n: int = 20) -> list[str]:
        stop = {"the","a","an","is","in","it","of","to","and","or","for",
                "with","on","at","by","from","this","that","are","was","be",
                "have","has","had","not","but","as","if","its","i","we","you"}
        words = []
        for text in texts:
            words.extend(re.findall(r"\b[a-z]{3,}\b", text.lower()))
        counts = Counter(w for w in words if w not in stop)
        return [w for w, _ in counts.most_common(n)]

    def _llm_analyse(self, text_sample: str, context: AnalysisContext) -> str:
        biz_ctx = context.business_context
        try:
            from llm.client import LLMClient
            llm = LLMClient()
            user_prompt = (
                f"Business context: {biz_ctx}\n\n"
                f"Text sample (up to 4000 chars):\n{text_sample[:4000]}"
            )
            return llm.complete(system=_NLP_SYSTEM, user=user_prompt)
        except Exception as e:
            logger.warning(f"LLM NLP analysis failed: {e}")
            return ""
