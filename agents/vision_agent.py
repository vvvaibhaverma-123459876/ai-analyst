"""
agents/vision_agent.py
Vision Agent — interprets image content already extracted by the ingestion layer.

The ImageParser in ingestion/ already ran OCR + LLM vision description.
This agent takes those descriptions and:
  1. Identifies what type of visual each image is (chart, table, screenshot, doc)
  2. Extracts any quantitative findings
  3. Cross-references with structured data where possible
  4. Synthesises into analytical findings

This separation is deliberate: ingestion extracts, vision agent interprets.
"""

from __future__ import annotations
import re
import pandas as pd
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)

_VISION_INTERPRET_SYSTEM = """You are a data analyst interpreting visual data descriptions.
Given descriptions of charts, dashboards, or document images, extract:
1. Key metrics and their values
2. Trends visible in charts (up/down/flat, magnitude)
3. Any anomalies or highlighted items
4. How this visual evidence relates to the structured data analysis
Be specific about numbers. Format as structured bullet points."""


class VisionAgent(BaseAgent):
    name = "vision"
    description = "Interprets charts, screenshots, and document images for analytical findings"

    def _run(self, context: AnalysisContext) -> AgentResult:
        doc = context.document

        if doc is None:
            return self.skip("No document in context.")

        descriptions = doc.image_descriptions
        ocr_chunks = [c for c in doc.text_chunks if "[OCR text]" in c or "[Image" in c]

        if not descriptions and not ocr_chunks:
            return self.skip("No image descriptions or OCR content found.")

        all_image_content = "\n\n---\n\n".join(descriptions + ocr_chunks)

        # Extract any numbers mentioned
        numbers_found = self._extract_numbers(all_image_content)

        # Classify image types
        image_types = [self._classify_image(desc) for desc in descriptions]

        # LLM interpretation
        interpretation = ""
        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            interpretation = self._llm_interpret(all_image_content, context)

        # Try to build a mini DataFrame from any table descriptions
        table_from_vision = self._extract_table(descriptions)

        n_images = len(descriptions)
        type_summary = ", ".join(set(image_types)) if image_types else "unknown"

        summary = (
            f"Vision analysis: {n_images} image(s) processed "
            f"({type_summary}). "
            f"{len(numbers_found)} numeric values extracted from visuals. "
        )
        if interpretation:
            first = interpretation.splitlines()[0][:120] if interpretation else ""
            summary += first

        return AgentResult(
            agent=self.name,
            status="success",
            summary=summary,
            data={
                "n_images": n_images,
                "image_types": image_types,
                "numbers_found": numbers_found,
                "raw_descriptions": descriptions,
                "interpretation": interpretation,
                "table_from_vision": table_from_vision,
            },
        )

    def _classify_image(self, description: str) -> str:
        desc_lower = description.lower()
        if any(w in desc_lower for w in ["bar chart", "line chart", "pie chart",
                                          "histogram", "scatter", "chart", "graph", "plot"]):
            return "chart"
        if any(w in desc_lower for w in ["table", "rows", "columns", "csv", "spreadsheet"]):
            return "table"
        if any(w in desc_lower for w in ["dashboard", "metric", "kpi", "widget"]):
            return "dashboard"
        if any(w in desc_lower for w in ["screenshot", "interface", "ui", "screen"]):
            return "screenshot"
        return "document"

    def _extract_numbers(self, text: str) -> list[dict]:
        pattern = r"([\w\s]{1,30})\s*[:=]\s*([\d,\.]+\s*[%kKmMbB]?)"
        matches = re.findall(pattern, text)
        results = []
        for label, value in matches[:20]:
            results.append({"label": label.strip(), "value": value.strip()})
        return results

    def _extract_table(self, descriptions: list[str]) -> pd.DataFrame | None:
        for desc in descriptions:
            if "TABLE:" in desc:
                try:
                    import csv, io
                    csv_part = desc.split("TABLE:")[-1].strip()
                    lines = [l.strip() for l in csv_part.splitlines() if l.strip()]
                    if len(lines) >= 2:
                        reader = csv.reader(lines)
                        rows = list(reader)
                        df = pd.DataFrame(rows[1:], columns=rows[0])
                        return df
                except Exception:
                    pass
        return None

    def _llm_interpret(self, content: str, context: AnalysisContext) -> str:
        try:
            from llm.client import LLMClient
            llm = LLMClient()
            kpi_info = f"Primary KPI being analysed: {context.kpi_col}" if context.kpi_col else ""
            user_prompt = (
                f"{kpi_info}\n\n"
                f"Image content extracted:\n{content[:4000]}"
            )
            return llm.complete(system=_VISION_INTERPRET_SYSTEM, user=user_prompt)
        except Exception as e:
            logger.warning(f"Vision LLM interpretation failed: {e}")
            return ""
