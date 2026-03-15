"""
ingestion/image_parser.py
Handles: .png, .jpg, .jpeg, .gif, .webp, .bmp, .tiff
Strategy:
  1. OCR via pytesseract (extracts text from scanned docs / screenshots)
  2. LLM vision description (extracts chart data, table structure, insight)
  3. Merges both into text_chunks + optional dataframe if table detected
"""

from __future__ import annotations
import base64
import io
import re
import pandas as pd
from ingestion.base_parser import BaseParser, ParsedDocument
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

_VISION_SYSTEM = """You are an expert data analyst examining an image.
Describe what you see concisely:
- If it is a chart/graph: state chart type, axes, key values, trend direction, any anomalies.
- If it is a table: extract the data as CSV rows.
- If it is a screenshot of a dashboard: list each metric, its value, and any visible trends.
- If it is a scanned document: summarise key data points and figures.
Return plain text only. For tables, prefix with TABLE: then CSV data."""


class ImageParser(BaseParser):

    def can_parse(self, filename: str, mime_type: str = "") -> bool:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return ext in IMAGE_EXTENSIONS or "image" in mime_type

    def parse(self, source, filename: str = "") -> ParsedDocument:
        doc = ParsedDocument(source_name=filename, source_type="image")

        # Load image bytes
        try:
            if hasattr(source, "read"):
                img_bytes = source.read()
            elif isinstance(source, bytes):
                img_bytes = source
            else:
                with open(source, "rb") as f:
                    img_bytes = f.read()
        except Exception as e:
            doc.add_warning(f"Image load failed: {e}")
            return doc

        # ── OCR ──────────────────────────────────────────────────────
        ocr_text = self._ocr(img_bytes, doc)
        if ocr_text.strip():
            doc.text_chunks.append(f"[OCR text]\n{ocr_text}")
            logger.info(f"OCR extracted {len(ocr_text)} chars")

        # ── LLM Vision ───────────────────────────────────────────────
        vision_text = self._vision_describe(img_bytes, filename, doc)
        if vision_text:
            doc.image_descriptions.append(vision_text)
            logger.info(f"Vision description: {len(vision_text)} chars")

            # If vision returned a TABLE, try to parse it as DataFrame
            if "TABLE:" in vision_text:
                self._extract_table_from_vision(vision_text, doc)
            else:
                doc.text_chunks.append(f"[Image analysis: {filename}]\n{vision_text}")

        doc.metadata["filename"] = filename
        doc.metadata["image_size_bytes"] = len(img_bytes)
        return doc

    def _ocr(self, img_bytes: bytes, doc: ParsedDocument) -> str:
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(img)
        except ImportError:
            pass
        except Exception as e:
            doc.add_warning(f"OCR failed: {e}")
        return ""

    def _vision_describe(self, img_bytes: bytes, filename: str, doc: ParsedDocument) -> str:
        if not (config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY):
            doc.add_warning("No LLM API key — image visual analysis skipped.")
            return ""
        try:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
            mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

            if config.LLM_PROVIDER == "anthropic" and config.ANTHROPIC_API_KEY:
                return self._vision_anthropic(b64, mime)
            elif config.OPENAI_API_KEY:
                return self._vision_openai(b64, mime)
        except Exception as e:
            doc.add_warning(f"Vision description failed: {e}")
        return ""

    def _vision_openai(self, b64: str, mime: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_SYSTEM},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}}
                ]
            }],
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()

    def _vision_anthropic(self, b64: str, mime: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": mime, "data": b64}},
                    {"type": "text", "text": _VISION_SYSTEM}
                ]
            }]
        )
        return msg.content[0].text.strip()

    def _extract_table_from_vision(self, vision_text: str, doc: ParsedDocument):
        try:
            csv_part = vision_text.split("TABLE:")[-1].strip()
            lines = [l.strip() for l in csv_part.splitlines() if l.strip()]
            if len(lines) >= 2:
                import csv as csv_mod
                reader = csv_mod.reader(lines)
                rows = list(reader)
                df = pd.DataFrame(rows[1:], columns=rows[0])
                doc.dataframes.append(df)
                doc.table_names.append("image_table")
                doc.text_chunks.append(f"[Table extracted from image]\n{csv_part[:300]}")
                logger.info(f"Image table extracted: {df.shape}")
        except Exception as e:
            doc.text_chunks.append(f"[Image analysis]\n{vision_text}")
            logger.warning(f"Image table parse failed: {e}")
