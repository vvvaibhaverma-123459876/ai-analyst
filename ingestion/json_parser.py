"""
ingestion/json_parser.py
Handles: .json, .jsonl, .ndjson
Flattens nested JSON where possible.
"""

from __future__ import annotations
import json
import pandas as pd
from ingestion.base_parser import BaseParser, ParsedDocument
from core.logger import get_logger

logger = get_logger(__name__)


class JSONParser(BaseParser):

    EXTENSIONS = {".json", ".jsonl", ".ndjson"}

    def can_parse(self, filename: str, mime_type: str = "") -> bool:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return ext in self.EXTENSIONS or "json" in mime_type

    def parse(self, source, filename: str = "") -> ParsedDocument:
        doc = ParsedDocument(source_name=filename, source_type="json")
        try:
            if hasattr(source, "read"):
                raw = source.read()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
            else:
                with open(source, "r") as f:
                    raw = f.read()

            # Try JSONL first
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            if len(lines) > 1:
                try:
                    records = [json.loads(l) for l in lines]
                    df = pd.json_normalize(records)
                    doc.dataframes.append(df)
                    doc.table_names.append("records")
                    logger.info(f"JSONL parsed: {df.shape}")
                    return doc
                except Exception:
                    pass

            # Standard JSON
            data = json.loads(raw)
            if isinstance(data, list):
                df = pd.json_normalize(data)
                doc.dataframes.append(df)
                doc.table_names.append("records")
            elif isinstance(data, dict):
                # Try each top-level key as a potential table
                for key, val in data.items():
                    if isinstance(val, list) and val:
                        try:
                            df = pd.json_normalize(val)
                            doc.dataframes.append(df)
                            doc.table_names.append(key)
                        except Exception:
                            doc.text_chunks.append(f"{key}: {json.dumps(val)[:500]}")
                    else:
                        doc.text_chunks.append(f"{key}: {str(val)[:500]}")
                if not doc.dataframes:
                    df = pd.json_normalize([data])
                    doc.dataframes.append(df)
                    doc.table_names.append("root")

            logger.info(f"JSON parsed: {len(doc.dataframes)} tables")
        except Exception as e:
            doc.add_warning(f"JSON parse error: {e}")
        return doc
