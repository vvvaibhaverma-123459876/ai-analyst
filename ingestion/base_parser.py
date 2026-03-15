"""
ingestion/base_parser.py
Abstract base all format parsers implement.
Every parser returns a ParsedDocument — a unified container
the rest of the system works with regardless of input format.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class ParsedDocument:
    """
    Unified internal representation of any ingested data source.
    Agents never see raw files — only ParsedDocuments.
    """
    source_name: str = ""
    source_type: str = ""          # "csv", "excel", "pdf", "image", "json",
                                   # "sql", "text", "stream", "word"

    # Structured data (may be multiple tables from one file)
    dataframes: list[pd.DataFrame] = field(default_factory=list)
    table_names: list[str] = field(default_factory=list)

    # Unstructured text extracted from the source
    text_chunks: list[str] = field(default_factory=list)

    # Image descriptions (from OCR / vision model)
    image_descriptions: list[str] = field(default_factory=list)

    # Raw metadata about the source
    metadata: dict[str, Any] = field(default_factory=dict)

    # Schema summary (column names, types, sample values)
    schema_summary: dict = field(default_factory=dict)

    # Parse errors / warnings
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def primary_df(self) -> pd.DataFrame:
        """First / largest dataframe, or empty."""
        if not self.dataframes:
            return pd.DataFrame()
        return max(self.dataframes, key=lambda d: len(d))

    @property
    def has_structured_data(self) -> bool:
        return any(not df.empty for df in self.dataframes)

    @property
    def has_text(self) -> bool:
        return bool(self.text_chunks)

    @property
    def all_text(self) -> str:
        return "\n\n".join(self.text_chunks)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        parts = [f"Source: {self.source_name} ({self.source_type})"]
        if self.has_structured_data:
            total_rows = sum(len(d) for d in self.dataframes)
            parts.append(f"Tables: {len(self.dataframes)}, total rows: {total_rows:,}")
        if self.has_text:
            parts.append(f"Text chunks: {len(self.text_chunks)}")
        if self.image_descriptions:
            parts.append(f"Images described: {len(self.image_descriptions)}")
        if self.warnings:
            parts.append(f"Warnings: {len(self.warnings)}")
        return " | ".join(parts)


class BaseParser(ABC):
    """Abstract parser interface."""

    @abstractmethod
    def can_parse(self, filename: str, mime_type: str = "") -> bool:
        """Return True if this parser handles the given file."""

    @abstractmethod
    def parse(self, source, filename: str = "") -> ParsedDocument:
        """
        Parse source (file path, bytes, or file-like object).
        Always returns a ParsedDocument — never raises.
        Errors go into doc.warnings.
        """
