"""
llm/sql_generator.py  — v0.6
Generates SQL from structured intent using LLM + schema context.

v0.6 upgrade: uses MultiHopJoinBuilder for deterministic cross-table JOIN
resolution instead of injecting hints and hoping the LLM gets ON conditions
right. The join scaffolding is computed from configs/joins.yaml; only SELECT,
WHERE, GROUP BY and ORDER BY are delegated to the LLM.
"""

from llm.client import LLMClient
from llm.prompts import Prompts
from core.exceptions import SQLGenerationError
from core.logger import get_logger
from semantic.metric_registry import MetricRegistry
from semantic.join_graph import JoinGraph
from sql.multihop_generator import MultiHopJoinBuilder
from metadata.schema_loader import SchemaLoader

logger = get_logger(__name__)


class SQLGenerator:

    def __init__(self, llm_client: LLMClient = None):
        self._llm          = llm_client or LLMClient()
        self._metric_registry = MetricRegistry()
        self._join_graph   = JoinGraph()
        self._multihop     = MultiHopJoinBuilder()
        self._schema_loader = SchemaLoader()

    def generate(
        self,
        intent: dict,
        schema_context: str | None = None,
        metric_context: str | None = None,
    ) -> str:
        """
        Returns a raw SQL string.
        Raises SQLGenerationError if LLM returns nothing useful.
        """
        schema_dict    = self._safe_load_schema()
        schema_context = schema_context or self._schema_loader.to_prompt_context()
        metric_context = self._build_metric_context(intent, metric_context)

        # --- v0.6: deterministic multi-hop join clause ---
        join_clause = self._multihop.resolve_for_intent(intent, schema_dict)
        if join_clause:
            schema_context = (
                f"{schema_context}\n\n"
                f"DETERMINISTIC JOIN CLAUSE (use exactly as-is for the FROM block):\n"
                f"{join_clause}"
            )
        else:
            # Legacy single-table hint path
            legacy_hint = self._build_join_context(intent, schema_dict)
            if legacy_hint:
                schema_context = f"{schema_context}\n\nJOIN GUIDANCE:\n{legacy_hint}"

        user_prompt = Prompts.sql_user(intent, schema_context, metric_context)
        try:
            sql = self._llm.complete(system=Prompts.SQL_SYSTEM, user=user_prompt)
        except Exception as e:
            raise SQLGenerationError(f"LLM failed to generate SQL: {e}") from e

        sql = sql.strip()
        if not sql:
            raise SQLGenerationError("LLM returned empty SQL.")

        if sql.startswith("```"):
            lines = sql.splitlines()
            sql = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        logger.info(f"Generated SQL:\n{sql}")
        return sql

    def _build_metric_context(self, intent: dict, metric_context: str | None) -> str:
        base = self._metric_registry.to_prompt_context()
        metric_name = intent.get("metric")
        resolved = None
        if metric_name:
            try:
                resolved = self._metric_registry.get(metric_name).key
            except Exception:
                resolved = self._metric_registry.resolve(str(metric_name))
        if resolved:
            detail = self._metric_registry.explain(resolved)
            base += f"\n\nREQUESTED METRIC DETAIL:\n{detail}"
        if metric_context and metric_context not in base:
            base += f"\n\nLEGACY METRIC CONTEXT:\n{metric_context}"
        return base

    def _safe_load_schema(self) -> dict:
        try:
            return self._schema_loader.load()
        except Exception:
            return {}

    def _build_join_context(self, intent: dict, schema_dict: dict) -> str:
        """Legacy single-hop hint — kept as fallback."""
        metric_name = intent.get("metric")
        metric_key = None
        if metric_name:
            try:
                metric_key = self._metric_registry.get(metric_name).key
            except Exception:
                metric_key = self._metric_registry.resolve(str(metric_name))

        candidate_tables: list[str] = []
        if metric_key:
            candidate_tables.extend(self._metric_registry.get(metric_key).source_tables)

        requested_fields = list(intent.get("dimensions", []) or []) + list(
            (intent.get("filters", {}) or {}).keys()
        )
        for field in requested_fields:
            for table_name, meta in schema_dict.items():
                for col in meta.get("columns", []):
                    col_name = col.get("name") if isinstance(col, dict) else col
                    if col_name == field and table_name not in candidate_tables:
                        candidate_tables.append(table_name)
                        break

        candidate_tables = [t for t in candidate_tables if t in schema_dict]
        if len(candidate_tables) < 2:
            return ""

        base = candidate_tables[0]
        join_lines: list[str] = []
        seen_edges: set[tuple[str, str]] = set()
        for table in candidate_tables[1:]:
            path = self._join_graph.find_path(base, table)
            if not path:
                continue
            for edge in path:
                edge_key = (edge.get("from", base), edge.get("to", table))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                on_pairs  = edge.get("on", []) or []
                if not on_pairs:
                    continue
                join_type   = edge.get("join_type", "LEFT")
                left_table  = edge.get("from", base)
                right_table = edge.get("to", table)
                conditions  = " AND ".join(
                    f"{left_table}.{pair.get('left')} = {right_table}.{pair.get('right')}"
                    for pair in on_pairs
                )
                join_lines.append(f"{join_type} JOIN {right_table} ON {conditions}")
        if not join_lines:
            return ""
        return "Base table: " + base + "\n" + "\n".join(f"- {line}" for line in join_lines)



class SQLGenerator:

    def __init__(self, llm_client: LLMClient = None):
        self._llm = llm_client or LLMClient()
        self._metric_registry = MetricRegistry()
        self._join_graph = JoinGraph()
        self._schema_loader = SchemaLoader()

    def generate(
        self,
        intent: dict,
        schema_context: str | None = None,
        metric_context: str | None = None,
    ) -> str:
        """
        Returns a raw SQL string.
        Raises SQLGenerationError if LLM returns nothing useful.
        """
        schema_dict = self._safe_load_schema()
        schema_context = schema_context or self._schema_loader.to_prompt_context()
        metric_context = self._build_metric_context(intent, metric_context)
        join_context = self._build_join_context(intent, schema_dict)
        if join_context:
            schema_context = f"{schema_context}\n\nJOIN GUIDANCE:\n{join_context}"

        user_prompt = Prompts.sql_user(intent, schema_context, metric_context)
        try:
            sql = self._llm.complete(system=Prompts.SQL_SYSTEM, user=user_prompt)
        except Exception as e:
            raise SQLGenerationError(f"LLM failed to generate SQL: {e}") from e

        sql = sql.strip()
        if not sql:
            raise SQLGenerationError("LLM returned empty SQL.")

        # Strip markdown fences if model ignored instructions
        if sql.startswith("```"):
            lines = sql.splitlines()
            sql = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        logger.info(f"Generated SQL:\n{sql}")
        return sql

    def _build_metric_context(self, intent: dict, metric_context: str | None) -> str:
        base = self._metric_registry.to_prompt_context()
        metric_name = intent.get("metric")
        resolved = None
        if metric_name:
            try:
                resolved = self._metric_registry.get(metric_name).key
            except Exception:
                resolved = self._metric_registry.resolve(str(metric_name))

        if resolved:
            detail = self._metric_registry.explain(resolved)
            base += f"\n\nREQUESTED METRIC DETAIL:\n{detail}"

        if metric_context and metric_context not in base:
            base += f"\n\nLEGACY METRIC CONTEXT:\n{metric_context}"
        return base

    def _safe_load_schema(self) -> dict:
        try:
            return self._schema_loader.load()
        except Exception:
            return {}

    def _build_join_context(self, intent: dict, schema_dict: dict) -> str:
        metric_name = intent.get("metric")
        metric_key = None
        if metric_name:
            try:
                metric_key = self._metric_registry.get(metric_name).key
            except Exception:
                metric_key = self._metric_registry.resolve(str(metric_name))

        candidate_tables: list[str] = []
        if metric_key:
            candidate_tables.extend(self._metric_registry.get(metric_key).source_tables)

        requested_fields = list(intent.get("dimensions", []) or []) + list((intent.get("filters", {}) or {}).keys())
        for field in requested_fields:
            for table_name, meta in schema_dict.items():
                for col in meta.get("columns", []):
                    col_name = col.get("name") if isinstance(col, dict) else col
                    if col_name == field and table_name not in candidate_tables:
                        candidate_tables.append(table_name)
                        break

        candidate_tables = [t for t in candidate_tables if t in schema_dict]
        if len(candidate_tables) < 2:
            return ""

        base = candidate_tables[0]
        join_lines: list[str] = []
        seen_edges: set[tuple[str, str]] = set()
        for table in candidate_tables[1:]:
            path = self._join_graph.find_path(base, table)
            if not path:
                continue
            for edge in path:
                edge_key = (edge.get("from", base), edge.get("to", table))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                on_pairs = edge.get("on", []) or []
                if not on_pairs:
                    continue
                join_type = edge.get("join_type", "LEFT")
                left_table = edge.get("from", base)
                right_table = edge.get("to", table)
                conditions = " AND ".join(
                    f"{left_table}.{pair.get('left')} = {right_table}.{pair.get('right')}"
                    for pair in on_pairs
                )
                join_lines.append(f"{join_type} JOIN {right_table} ON {conditions}")
        if not join_lines:
            return ""
        return "Base table suggestion: " + base + "\n" + "\n".join(f"- {line}" for line in join_lines)
