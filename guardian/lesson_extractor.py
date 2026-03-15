from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime

class LessonExtractor:
    def __init__(self, store_path: str | None = None):
        self._path = Path(store_path) if store_path else Path(__file__).resolve().parent.parent / "memory" / "lessons.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
    def extract(self, context) -> dict:
        guardian = context.results.get("guardian")
        contradictions = guardian.data.get("contradictions", []) if guardian and guardian.status == "success" else []
        return {
            "run_id": context.run_id,
            "kpi": context.kpi_col,
            "active_agents": list(context.active_agents),
            "data_quality_score": (context.data_quality_report or {}).get("score"),
            "contradictions": len(contradictions),
            "primary_conclusion": getattr(getattr(context, "research_plan", None), "primary_conclusion", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }
    def persist(self, lesson: dict) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(lesson, default=str) + "\n")
