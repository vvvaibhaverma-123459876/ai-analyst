"""
semantic/dbt_adapter.py  — v0.6
dbt Semantic Layer adapter.

Auto-populates the MetricRegistry from:
  1. dbt manifest.json  (v9+, dbt-core >= 1.5)
  2. dbt metrics.yml    (legacy dbt metrics format)
  3. Cube.dev schema    (cube.js / cube cloud)

This eliminates manual YAML sync between the data team's dbt definitions
and the AI analyst's metric registry.

Usage:
    from semantic.dbt_adapter import DbtAdapter
    adapter = DbtAdapter()
    metrics = adapter.load()   # → dict suitable for MetricRegistry
    # Or auto-populate:
    adapter.populate_registry()
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from core.logger import get_logger

logger = get_logger(__name__)

DBT_PROJECT_DIR = os.getenv("DBT_PROJECT_DIR", "")
DBT_MANIFEST_PATH = os.getenv("DBT_MANIFEST_PATH", "")
CUBE_SCHEMA_DIR = os.getenv("CUBE_SCHEMA_DIR", "")


class DbtAdapter:

    def __init__(self, project_dir: str = None, manifest_path: str = None):
        self._project_dir   = project_dir   or DBT_PROJECT_DIR
        self._manifest_path = manifest_path or DBT_MANIFEST_PATH

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """
        Returns a metrics dict compatible with MetricRegistry.
        Tries manifest.json → dbt_metrics_yml → empty dict.
        """
        # Try manifest.json first (most complete)
        if self._manifest_path and Path(self._manifest_path).exists():
            metrics = self._load_from_manifest(self._manifest_path)
            if metrics:
                logger.info("dbt metrics loaded from manifest: %d metrics", len(metrics))
                return metrics

        # Try to find manifest in project dir
        if self._project_dir:
            candidates = [
                Path(self._project_dir) / "target" / "manifest.json",
                Path(self._project_dir) / "manifest.json",
            ]
            for path in candidates:
                if path.exists():
                    metrics = self._load_from_manifest(str(path))
                    if metrics:
                        logger.info("dbt manifest found at %s: %d metrics", path, len(metrics))
                        return metrics

        # Try legacy dbt metrics YAML
        if self._project_dir:
            metrics = self._load_from_metrics_yml(self._project_dir)
            if metrics:
                return metrics

        logger.info("No dbt manifest found — MetricRegistry will use configs/metrics.yaml")
        return {}

    def _load_from_manifest(self, manifest_path: str) -> dict[str, Any]:
        """Parse dbt manifest.json → MetricRegistry-compatible dict."""
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning("Could not read manifest.json: %s", e)
            return {}

        metrics_node = manifest.get("metrics", {})
        if not metrics_node:
            return {}

        result: dict[str, Any] = {}
        for node_key, node in metrics_node.items():
            name = node.get("name", "")
            if not name:
                continue

            # Determine aggregation / formula
            calc_method = node.get("calculation_method", node.get("type", "sum"))
            agg = self._map_calc_method(calc_method)

            # Numerator / denominator for ratio metrics
            numerator = denominator = None
            if agg == "ratio":
                num_node = node.get("numerator", {})
                den_node = node.get("denominator", {})
                numerator   = num_node.get("name") or num_node.get("metric_name")
                denominator = den_node.get("name") or den_node.get("metric_name")

            # Dimensions and grains
            dimensions: list[str] = []
            for d in node.get("dimensions", []):
                d_name = d if isinstance(d, str) else d.get("name", "")
                if d_name:
                    dimensions.append(d_name)

            allowed_grains: list[str] = []
            for g in node.get("time_grains", node.get("time_grain_set", [])):
                if isinstance(g, str):
                    allowed_grains.append(g)

            # Source tables
            refs = [r if isinstance(r, str) else r.get("name", "")
                    for r in node.get("refs", node.get("model", []))]
            source_tables = [r for r in refs if r]

            result[name] = {
                "description":    node.get("description", name),
                "aggregation":    agg,
                "column":         node.get("expression", node.get("sql")),
                "numerator":      numerator,
                "denominator":    denominator,
                "aliases":        [node.get("label", name)] if node.get("label") else [],
                "dimensions":     dimensions,
                "allowed_grains": allowed_grains or ["daily"],
                "owner":          node.get("meta", {}).get("owner", "unassigned"),
                "caveats":        node.get("meta", {}).get("caveats", []),
                "maturity":       node.get("meta", {}).get("maturity", "draft"),
                "source_tables":  source_tables,
                "_dbt_node":      node_key,
            }
        return result

    def _load_from_metrics_yml(self, project_dir: str) -> dict[str, Any]:
        """Parse legacy dbt metrics.yml files from project models directory."""
        import yaml
        result: dict[str, Any] = {}
        project_path = Path(project_dir)
        for yml_path in project_path.rglob("*.yml"):
            try:
                with open(yml_path) as f:
                    content = yaml.safe_load(f) or {}
                for metric in content.get("metrics", []):
                    name = metric.get("name", "")
                    if not name:
                        continue
                    result[name] = {
                        "description":    metric.get("description", name),
                        "aggregation":    self._map_calc_method(metric.get("type", "sum")),
                        "column":         metric.get("sql"),
                        "dimensions":     [d if isinstance(d, str) else d.get("name", "")
                                           for d in metric.get("dimensions", [])],
                        "allowed_grains": metric.get("time_grains", ["daily"]),
                        "owner":          metric.get("meta", {}).get("owner", "unassigned"),
                        "source_tables":  [metric.get("model", "").replace("ref('", "").rstrip("')")],
                        "maturity":       "production",
                    }
            except Exception:
                pass
        return result

    @staticmethod
    def _map_calc_method(method: str) -> str:
        mapping = {
            "sum": "sum", "count": "count", "average": "mean",
            "count_distinct": "count_distinct", "max": "max", "min": "min",
            "ratio": "ratio", "derived": "ratio", "expression": "ratio",
        }
        return mapping.get((method or "sum").lower(), "sum")

    def populate_registry(self):
        """
        Convenience: load metrics and inject into the live MetricRegistry.
        Call this once at startup if dbt manifest is available.
        """
        from semantic.metric_registry import MetricRegistry
        raw_metrics = self.load()
        if not raw_metrics:
            return
        registry = MetricRegistry(raw_metrics={"metrics": raw_metrics})
        logger.info(
            "MetricRegistry populated from dbt: %d metrics",
            len(raw_metrics),
        )
        return registry


class CubeAdapter:
    """Reads Cube.dev schema JS/YAML files and emits MetricRegistry-compatible dicts."""

    def __init__(self, schema_dir: str = None):
        self._schema_dir = schema_dir or CUBE_SCHEMA_DIR

    def load(self) -> dict[str, Any]:
        if not self._schema_dir or not Path(self._schema_dir).exists():
            return {}
        import yaml
        result: dict[str, Any] = {}
        for yml_path in Path(self._schema_dir).rglob("*.yml"):
            try:
                with open(yml_path) as f:
                    content = yaml.safe_load(f) or {}
                for cube in content.get("cubes", []):
                    for measure in cube.get("measures", []):
                        name = f"{cube['name']}.{measure['name']}"
                        result[name] = {
                            "description":    measure.get("description", name),
                            "aggregation":    measure.get("type", "sum"),
                            "column":         measure.get("sql"),
                            "dimensions":     [d["name"] for d in cube.get("dimensions", [])],
                            "allowed_grains": ["daily"],
                            "owner":          "unassigned",
                            "source_tables":  [cube.get("sql_table", cube["name"])],
                            "maturity":       "production",
                        }
            except Exception:
                pass
        logger.info("CubeAdapter loaded %d measures", len(result))
        return result
