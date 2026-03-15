"""
ingestion/ingestion_engine.py
Central dispatcher — accepts any file/source, auto-detects format,
routes to the right parser, and returns a unified ParsedDocument.

This is the ONLY entry point the rest of the system uses.
No agent or UI should call individual parsers directly.
"""

from __future__ import annotations
import mimetypes
from pathlib import Path
from typing import Union, IO

from ingestion.base_parser import ParsedDocument
from ingestion.csv_parser import CSVParser
from ingestion.excel_parser import ExcelParser
from ingestion.json_parser import JSONParser
from ingestion.pdf_parser import PDFParser
from ingestion.image_parser import ImageParser
from ingestion.word_parser import WordParser
from ingestion.text_parser import TextParser
from ingestion.sql_parser import SQLParser
from ingestion.stream_parser import StreamParser
from core.logger import get_logger

logger = get_logger(__name__)

# Parser registry — tried in order, first match wins
_PARSERS = [
    CSVParser(),
    ExcelParser(),
    JSONParser(),
    PDFParser(),
    ImageParser(),
    WordParser(),
    TextParser(),
]


class IngestionEngine:
    """
    Usage:
        engine = IngestionEngine()
        doc = engine.ingest(uploaded_file, filename="sales.csv")
        df  = doc.primary_df
        txt = doc.all_text
    """

    def ingest(
        self,
        source: Union[str, bytes, IO],
        filename: str = "",
        mime_type: str = "",
    ) -> ParsedDocument:
        """
        Auto-detect format and parse.
        Returns ParsedDocument regardless of success/failure.
        """
        filename = filename or getattr(source, "name", "") or ""
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or ""

        logger.info(f"Ingesting: '{filename}' mime='{mime_type}'")

        parser = self._find_parser(filename, mime_type)
        if parser is None:
            doc = ParsedDocument(source_name=filename, source_type="unknown")
            doc.add_warning(f"No parser found for '{filename}' ({mime_type}). "
                            f"Attempting text fallback.")
            parser = TextParser()

        doc = parser.parse(source, filename=filename)
        logger.info(f"Ingestion complete: {doc.summary()}")
        return doc

    def ingest_multiple(
        self,
        sources: list[tuple],
    ) -> list[ParsedDocument]:
        """
        sources: list of (source, filename) tuples.
        Returns list of ParsedDocuments.
        """
        return [self.ingest(src, fname) for src, fname in sources]

    def ingest_sql(
        self,
        connection_string: str = None,
        engine=None,
        query: str = None,
        filename: str = "database",
    ) -> ParsedDocument:
        parser = SQLParser(
            connection_string=connection_string,
            engine=engine,
            query=query,
        )
        return parser.parse(filename=filename)

    def ingest_stream(
        self,
        records: list[dict] = None,
        api_url: str = None,
        kafka_config: dict = None,
        filename: str = "stream",
    ) -> ParsedDocument:
        parser = StreamParser(
            records=records,
            api_url=api_url,
            kafka_config=kafka_config,
        )
        return parser.parse(filename=filename)

    def _find_parser(self, filename: str, mime_type: str):
        for parser in _PARSERS:
            if parser.can_parse(filename, mime_type):
                return parser
        return None

    @staticmethod
    def supported_extensions() -> list[str]:
        return [
            ".csv", ".tsv",
            ".xlsx", ".xls", ".ods",
            ".json", ".jsonl",
            ".pdf",
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
            ".docx", ".doc",
            ".txt", ".md", ".log", ".html",
        ]
