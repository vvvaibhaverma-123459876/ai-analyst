from __future__ import annotations

"""Simple governed join graph built from configs/joins.yaml."""

from collections import defaultdict, deque
from typing import Any

from core.config import config


class JoinGraph:
    def __init__(self, raw_joins: dict[str, Any] | None = None):
        raw = raw_joins if raw_joins is not None else config.JOINS
        self._graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._load(raw)

    def _load(self, raw: Any) -> None:
        joins = raw.get("joins", raw) if isinstance(raw, dict) else raw
        if isinstance(joins, list):
            for edge in joins:
                self._add_list_edge(edge or {})
        elif isinstance(joins, dict):
            for left, edges in joins.items():
                for edge in edges or []:
                    self._add_dict_edge(left, edge or {})

    def _add_list_edge(self, edge: dict[str, Any]) -> None:
        left = edge.get("left_table") or edge.get("left")
        right = edge.get("right_table") or edge.get("right") or edge.get("to")
        if not left or not right:
            return
        on_pairs = edge.get("on", [])
        normalised = {
            "from": left,
            "to": right,
            "join_type": edge.get("join_type", "LEFT"),
            "on": on_pairs,
            "reversed": False,
        }
        self._graph[left].append(normalised)
        reverse_pairs = [
            {"left": pair.get("right"), "right": pair.get("left")} for pair in on_pairs
        ]
        reverse = {
            "from": right,
            "to": left,
            "join_type": edge.get("join_type", "LEFT"),
            "on": reverse_pairs,
            "reversed": True,
        }
        self._graph[right].append(reverse)

    def _add_dict_edge(self, left: str, edge: dict[str, Any]) -> None:
        right = edge.get("to")
        if not right:
            return
        normalised = {
            "from": left,
            "to": right,
            "join_type": edge.get("join_type", "LEFT"),
            "on": edge.get("on", []),
            "reversed": False,
        }
        self._graph[left].append(normalised)
        reverse_pairs = [
            {"left": pair.get("right"), "right": pair.get("left")} for pair in edge.get("on", [])
        ]
        reverse = {
            "from": right,
            "to": left,
            "join_type": edge.get("join_type", "LEFT"),
            "on": reverse_pairs,
            "reversed": True,
        }
        self._graph[right].append(reverse)

    def neighbors(self, table: str) -> list[dict[str, Any]]:
        return list(self._graph.get(table, []))

    def find_path(self, start: str, end: str) -> list[dict[str, Any]]:
        if start == end:
            return []
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            for edge in self._graph.get(node, []):
                nxt = edge["to"]
                if nxt in seen:
                    continue
                next_path = path + [edge]
                if nxt == end:
                    return next_path
                seen.add(nxt)
                queue.append((nxt, next_path))
        return []
