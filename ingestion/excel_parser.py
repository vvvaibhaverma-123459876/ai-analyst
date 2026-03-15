"""
ingestion/excel_parser.py
Handles: .xlsx, .xls, .ods
Extracts all sheets as separate DataFrames.
"""

from __future__ import annotations
import pandas as pd
from ingestion.base_parser import BaseParser, ParsedDocument
from core.logger import get_logger

logger = get_logger(__name__)


class ExcelParser(BaseParser):

    EXTENSIONS = {".xlsx", ".xls", ".ods", ".xlsm"}

    def can_parse(self, filename: str, mime_type: str = "") -> bool:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return ext in self.EXTENSIONS or "spreadsheet" in mime_type or "excel" in mime_type

    def parse(self, source, filename: str = "") -> ParsedDocument:
        doc = ParsedDocument(source_name=filename, source_type="excel")
        try:
            xl = pd.ExcelFile(source)
            for sheet in xl.sheet_names:
                try:
                    df = xl.parse(sheet)
                    if df.empty:
                        continue
                    doc.dataframes.append(df)
                    doc.table_names.append(sheet)
                    logger.info(f"Sheet '{sheet}': {df.shape}")
                except Exception as e:
                    doc.add_warning(f"Sheet '{sheet}' failed: {e}")

            if doc.dataframes:
                doc.schema_summary = {
                    name: {"cols": list(df.columns), "rows": len(df)}
                    for name, df in zip(doc.table_names, doc.dataframes)
                }
        except Exception as e:
            doc.add_warning(f"Excel parse error: {e}")
        return doc
