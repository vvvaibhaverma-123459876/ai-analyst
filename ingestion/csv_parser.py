"""
ingestion/csv_parser.py
Handles: .csv, .tsv, .txt (tabular)
"""

from __future__ import annotations
import pandas as pd
from ingestion.base_parser import BaseParser, ParsedDocument
from core.logger import get_logger

logger = get_logger(__name__)


class CSVParser(BaseParser):

    EXTENSIONS = {".csv", ".tsv", ".txt"}

    def can_parse(self, filename: str, mime_type: str = "") -> bool:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return ext in self.EXTENSIONS or "csv" in mime_type or "tab-separated" in mime_type

    def parse(self, source, filename: str = "") -> ParsedDocument:
        doc = ParsedDocument(source_name=filename, source_type="csv")
        try:
            sep = "\t" if filename.endswith(".tsv") else ","
            df = pd.read_csv(source, sep=sep, low_memory=False)
            doc.dataframes.append(df)
            doc.table_names.append(filename or "data")
            doc.schema_summary = self._schema(df)
            logger.info(f"CSV parsed: {df.shape}")
        except Exception as e:
            doc.add_warning(f"CSV parse error: {e}")
        return doc

    def _schema(self, df: pd.DataFrame) -> dict:
        return {
            col: {
                "dtype": str(df[col].dtype),
                "nulls": int(df[col].isna().sum()),
                "sample": df[col].dropna().head(3).tolist(),
            }
            for col in df.columns
        }
