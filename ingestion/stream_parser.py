"""
ingestion/stream_parser.py
Handles: streaming data sources.
Supported: Kafka topic snapshot, webhook payload buffer, API polling.
For now implements a snapshot model — pulls N recent records and
normalises them as a DataFrame. True streaming (continuous) is
handled by the alert scheduler, not this parser.
"""

from __future__ import annotations
import json
import pandas as pd
from datetime import datetime
from ingestion.base_parser import BaseParser, ParsedDocument
from core.logger import get_logger

logger = get_logger(__name__)


class StreamParser(BaseParser):
    """
    Accepts:
      - A list of JSON records (already buffered)
      - A Kafka consumer (optional dep)
      - A URL for API polling
    """

    def __init__(
        self,
        records: list[dict] = None,
        kafka_config: dict = None,
        api_url: str = None,
        api_headers: dict = None,
        max_records: int = 10000,
    ):
        self._records = records
        self._kafka_config = kafka_config
        self._api_url = api_url
        self._api_headers = api_headers or {}
        self._max_records = max_records

    def can_parse(self, filename: str, mime_type: str = "") -> bool:
        return "stream" in filename.lower() or "kafka" in filename.lower()

    def parse(self, source=None, filename: str = "stream") -> ParsedDocument:
        doc = ParsedDocument(source_name=filename, source_type="stream")
        doc.metadata["snapshot_time"] = datetime.now().isoformat()

        records = []

        # ── Buffered records ─────────────────────────────────────────
        if self._records:
            records = self._records[:self._max_records]

        # ── API polling ──────────────────────────────────────────────
        elif self._api_url:
            records = self._poll_api(doc)

        # ── Kafka snapshot ───────────────────────────────────────────
        elif self._kafka_config:
            records = self._kafka_snapshot(doc)

        # ── Source is raw bytes/string ───────────────────────────────
        elif source is not None:
            try:
                if hasattr(source, "read"):
                    raw = source.read()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                else:
                    raw = str(source)
                records = [json.loads(l) for l in raw.splitlines() if l.strip()]
            except Exception as e:
                doc.add_warning(f"Stream source parse failed: {e}")

        if records:
            try:
                df = pd.json_normalize(records)
                doc.dataframes.append(df)
                doc.table_names.append("stream_snapshot")
                doc.metadata["record_count"] = len(df)
                logger.info(f"Stream snapshot: {len(df)} records")
            except Exception as e:
                doc.add_warning(f"Stream normalisation failed: {e}")
        else:
            doc.add_warning("No records received from stream source.")

        return doc

    def _poll_api(self, doc: ParsedDocument) -> list[dict]:
        try:
            import requests
            resp = requests.get(self._api_url, headers=self._api_headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data[:self._max_records]
            if isinstance(data, dict):
                for key in ("data", "records", "results", "items"):
                    if key in data and isinstance(data[key], list):
                        return data[key][:self._max_records]
                return [data]
        except Exception as e:
            doc.add_warning(f"API poll failed: {e}")
        return []

    def _kafka_snapshot(self, doc: ParsedDocument) -> list[dict]:
        try:
            from kafka import KafkaConsumer
            cfg = self._kafka_config
            consumer = KafkaConsumer(
                cfg["topic"],
                bootstrap_servers=cfg.get("bootstrap_servers", "localhost:9092"),
                auto_offset_reset="earliest",
                consumer_timeout_ms=cfg.get("timeout_ms", 5000),
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            records = []
            for msg in consumer:
                records.append(msg.value)
                if len(records) >= self._max_records:
                    break
            consumer.close()
            return records
        except ImportError:
            doc.add_warning("kafka-python not installed. Run: pip install kafka-python")
        except Exception as e:
            doc.add_warning(f"Kafka snapshot failed: {e}")
        return []
