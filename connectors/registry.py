"""
connectors/registry.py  — v0.6
Connector registry: discovers, instantiates, and health-checks all connectors.

Usage:
    from connectors.registry import ConnectorRegistry
    registry = ConnectorRegistry()
    df = registry.execute("SELECT * FROM orders LIMIT 100")
    schema = registry.get_active_schema()
"""

from __future__ import annotations
import os
from typing import Any
import pandas as pd
from core.logger import get_logger

logger = get_logger(__name__)

CONNECTOR_PRIORITY = ["postgres", "snowflake", "bigquery", "redshift", "athena", "csv"]


class ConnectorRegistry:
    """
    Auto-detects configured connectors by inspecting env vars.
    Returns a unified execute() / get_schema() interface regardless of backend.
    """

    def __init__(self):
        self._connectors: dict[str, Any] = {}
        self._active: str | None = None
        self._discover()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self):
        """Probe all supported connectors and register those that connect."""
        probes = [
            ("postgres",   self._try_postgres),
            ("snowflake",  self._try_snowflake),
            ("bigquery",   self._try_bigquery),
            ("redshift",   self._try_redshift),
            ("athena",     self._try_athena),
        ]
        for name, factory in probes:
            try:
                connector = factory()
                if connector is not None:
                    connector.connect()
                    if connector.test_connection():
                        self._connectors[name] = connector
                        logger.info("ConnectorRegistry: %s registered", name)
                        if self._active is None:
                            self._active = name
            except Exception as e:
                logger.debug("Connector %s not available: %s", name, e)

    def _try_postgres(self):
        if os.getenv("DATABASE_URL") or os.getenv("POSTGRES_HOST"):
            from connectors.postgres_connector import PostgresConnector
            return PostgresConnector()
        return None

    def _try_snowflake(self):
        if os.getenv("SNOWFLAKE_ACCOUNT"):
            from connectors.snowflake_connector import SnowflakeConnector
            return SnowflakeConnector()
        return None

    def _try_bigquery(self):
        if os.getenv("BQ_PROJECT_ID"):
            from connectors.bigquery_connector import BigQueryConnector
            return BigQueryConnector()
        return None

    def _try_redshift(self):
        if os.getenv("REDSHIFT_HOST"):
            from connectors.redshift_connector import RedshiftConnector
            return RedshiftConnector()
        return None

    def _try_athena(self):
        if os.getenv("ATHENA_S3_STAGING_DIR"):
            from connectors.athena_connector import AthenaConnector
            return AthenaConnector()
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def available(self) -> list[str]:
        return list(self._connectors.keys())

    def active(self) -> str | None:
        return self._active

    def set_active(self, name: str):
        if name not in self._connectors:
            raise ValueError(f"Connector '{name}' not registered. Available: {self.available()}")
        self._active = name
        logger.info("Active connector set to: %s", name)

    def get(self, name: str = None):
        key = name or self._active
        if key is None:
            raise RuntimeError("No active connector. Configure at least one data source.")
        return self._connectors[key]

    def execute(self, query: str, connector_name: str = None) -> pd.DataFrame:
        return self.get(connector_name).execute(query)

    def get_active_schema(self) -> dict:
        return self.get().get_schema()

    def health_check(self) -> dict[str, bool]:
        return {
            name: conn.test_connection()
            for name, conn in self._connectors.items()
        }

    def status_summary(self) -> str:
        if not self._connectors:
            return "No connectors registered. Upload a CSV or configure a data source."
        statuses = self.health_check()
        lines = [f"Active connector: {self._active}"]
        for name, healthy in statuses.items():
            icon = "✓" if healthy else "✗"
            lines.append(f"  {icon} {name}")
        return "\n".join(lines)
