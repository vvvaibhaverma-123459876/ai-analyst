"""
ingestion/text_parser.py
Handles: .txt, .md, .log, .html, and any plain text.
Detects embedded CSV-like tables within text.
"""

from __future__ import annotations
import re
import io
import pandas as pd
from ingestion.base_parser import BaseParser, ParsedDocument
from core.logger import get_logger

logger = get_logger(__name__)


class TextParser(BaseParser):

    EXTENSIONS = {".txt", ".md", ".log", ".html", ".htm", ".rtf"}

    def can_parse(self, filename: str, mime_type: str = "") -> bool:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return (ext in self.EXTENSIONS or
                "text/plain" in mime_type or
                "text/html" in mime_type or
                "markdown" in mime_type)

    def parse(self, source, filename: str = "") -> ParsedDocument:
        doc = ParsedDocument(source_name=filename, source_type="text")
        try:
            if hasattr(source, "read"):
                raw = source.read()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
            else:
                with open(source, "r", errors="replace") as f:
                    raw = f.read()

            # Strip HTML tags if needed
            if filename.lower().endswith((".html", ".htm")):
                raw = re.sub(r"<[^>]+>", " ", raw)
                raw = re.sub(r"\s+", " ", raw).strip()

            # Chunk into ~500 word blocks
            words = raw.split()
            chunk_size = 500
            for i in range(0, len(words), chunk_size):
                doc.text_chunks.append(" ".join(words[i:i + chunk_size]))

            # Detect embedded CSV-like tables (lines with consistent delimiters)
            self._detect_embedded_tables(raw, doc)

            logger.info(f"Text parsed: {len(doc.text_chunks)} chunks")

        except Exception as e:
            doc.add_warning(f"Text parse error: {e}")

        return doc

    def _detect_embedded_tables(self, text: str, doc: ParsedDocument):
        """Look for blocks of comma/tab-separated lines that look like tables."""
        lines = text.splitlines()
        candidate_start = None
        candidate_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            comma_count = stripped.count(",")
            tab_count = stripped.count("\t")
            if comma_count >= 2 or tab_count >= 2:
                if candidate_start is None:
                    candidate_start = i
                candidate_lines.append(stripped)
            else:
                if len(candidate_lines) >= 4:
                    self._try_parse_table(candidate_lines, doc)
                candidate_start = None
                candidate_lines = []

        if len(candidate_lines) >= 4:
            self._try_parse_table(candidate_lines, doc)

    def _try_parse_table(self, lines: list[str], doc: ParsedDocument):
        try:
            sep = "\t" if lines[0].count("\t") >= lines[0].count(",") else ","
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep)
            if not df.empty and len(df.columns) >= 2:
                doc.dataframes.append(df)
                doc.table_names.append(f"embedded_table_{len(doc.dataframes)}")
                logger.info(f"Embedded table detected: {df.shape}")
        except Exception:
            pass
