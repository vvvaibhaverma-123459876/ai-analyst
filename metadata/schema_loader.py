"""
metadata/schema_loader.py
Loads schema from config YAML or live from a connector.
Returns a normalised dict the LLM can use as context.
"""

from core.config import config
from core.exceptions import MetadataError
from core.logger import get_logger

logger = get_logger(__name__)


class SchemaLoader:

    def __init__(self, connector=None):
        """
        connector: optional live connector (Athena etc.).
        If None, falls back to configs/tables.yaml.
        """
        self._connector = connector
        self._schema_cache: dict = {}

    def load(self, force_refresh: bool = False) -> dict:
        if self._schema_cache and not force_refresh:
            return self._schema_cache

        if self._connector is not None:
            try:
                self._schema_cache = self._connector.get_schema()
                logger.info("Schema loaded from live connector.")
                return self._schema_cache
            except Exception as e:
                logger.warning(f"Live schema fetch failed, falling back to YAML: {e}")

        # Fallback: tables.yaml
        tables = config.TABLES.get("tables", {})
        if not tables:
            raise MetadataError("No schema found in configs/tables.yaml and no connector provided.")

        schema = {}
        for table_name, table_def in tables.items():
            schema[table_name] = {
                "description": table_def.get("description", ""),
                "date_column": table_def.get("date_column", ""),
                "grain": table_def.get("grain", ""),
                "columns": table_def.get("columns", []),
            }

        self._schema_cache = schema
        logger.info(f"Schema loaded from YAML: {list(schema.keys())}")
        return schema

    def to_prompt_context(self) -> str:
        """Returns schema as a plain-text string for LLM prompts."""
        schema = self.load()
        lines = []
        for table, meta in schema.items():
            lines.append(f"Table: {table} — {meta.get('description', '')}")
            for col in meta.get("columns", []):
                if isinstance(col, dict):
                    lines.append(f"  {col['name']} ({col['type']}): {col.get('description', '')}")
                else:
                    lines.append(f"  {col}")
        return "\n".join(lines)
