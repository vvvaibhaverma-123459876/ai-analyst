"""
sql/multihop_generator.py  — v0.6
Multi-hop JOIN resolver.

Given a set of required tables (from metric source_tables + requested
dimensions), this module:
  1. Uses JoinGraph.find_path() to compute the shortest join chain
  2. Renders the full FROM … JOIN … ON chain as SQL
  3. Injects it into the SQLGenerator prompt context so the LLM
     only needs to supply SELECT + WHERE + GROUP BY — the join scaffolding
     is computed and validated deterministically

Why deterministic vs LLM-generated joins?
  LLMs hallucinate ON conditions — this module ensures every join key
  is grounded in the configs/joins.yaml spec the data team owns.
"""

from __future__ import annotations
from typing import Any
from core.logger import get_logger
from semantic.join_graph import JoinGraph
from semantic.metric_registry import MetricRegistry

logger = get_logger(__name__)


class MultiHopJoinBuilder:

    def __init__(self):
        self._graph    = JoinGraph()
        self._registry = MetricRegistry()

    def build_join_clause(
        self,
        base_table: str,
        required_tables: list[str],
        schema_dict: dict[str, Any] = None,
    ) -> str:
        """
        Returns a SQL fragment:
            FROM base_table
            LEFT JOIN t2 ON base_table.fk = t2.pk
            LEFT JOIN t3 ON t2.fk = t3.pk
            ...
        Only tables reachable from base_table are included.
        """
        seen: set[str] = {base_table}
        join_lines: list[str] = [f"FROM {base_table}"]

        for target in required_tables:
            if target == base_table or target in seen:
                continue
            path = self._graph.find_path(base_table, target)
            if not path:
                logger.warning(
                    "No join path from %s to %s — skipping", base_table, target
                )
                continue
            for edge in path:
                right = edge["to"]
                if right in seen:
                    continue
                seen.add(right)
                join_type = edge.get("join_type", "LEFT")
                on_pairs  = edge.get("on", []) or []
                if not on_pairs:
                    continue
                conditions = " AND ".join(
                    f"{edge['from']}.{p['left']} = {right}.{p['right']}"
                    for p in on_pairs
                )
                join_lines.append(f"{join_type} JOIN {right} ON {conditions}")

        return "\n".join(join_lines)

    def resolve_for_intent(self, intent: dict, schema_dict: dict = None) -> str:
        """
        Given an analysis intent dict (metric, dimensions, filters),
        compute the minimal join clause covering all required tables.
        """
        required_tables: list[str] = []

        metric_name = intent.get("metric", "")
        if metric_name:
            try:
                m = self._registry.get(metric_name)
                required_tables.extend(m.source_tables)
            except Exception:
                resolved = self._registry.resolve(metric_name)
                if resolved:
                    try:
                        m = self._registry.get(resolved)
                        required_tables.extend(m.source_tables)
                    except Exception:
                        pass

        # Add tables implied by dimension columns
        if schema_dict:
            requested_fields = (
                list(intent.get("dimensions", []) or []) +
                list((intent.get("filters", {}) or {}).keys())
            )
            for field in requested_fields:
                for table_name, cols in schema_dict.items():
                    col_names = [
                        c["name"] if isinstance(c, dict) else c
                        for c in (cols if isinstance(cols, list) else cols.get("columns", []))
                    ]
                    if field in col_names and table_name not in required_tables:
                        required_tables.append(table_name)

        if not required_tables:
            return ""

        base = required_tables[0]
        rest = required_tables[1:]
        clause = self.build_join_clause(base, rest, schema_dict)
        logger.info("Multi-hop join clause:\n%s", clause)
        return clause

    def tables_for_metric(self, metric_name: str) -> list[str]:
        try:
            return self._registry.get(metric_name).source_tables
        except Exception:
            return []
