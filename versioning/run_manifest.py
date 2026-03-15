"""
versioning/run_manifest.py  — v9
RunManifest: complete reproducibility record for every pipeline run.

v9 additions (Phase 6):
  - join_graph_version, prompt_version, config_version fields
  - output_classification stored
  - guardian_summary stored (contradiction count, evidence grade)
  - evidence_summary stored (top hypothesis conclusions)
  - replay_type enum: exact | approximate | data_reloaded | offline
  - notes list for pipeline warnings (grain coercion, DQ warnings, etc.)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
import json


@dataclass
class RunManifest:
    run_id: str
    created_at: str

    # ── Versioned governing state ────────────────────────────────────
    metric_registry_version: str    = "v9"
    join_graph_version: str         = "v9"
    policy_version: str             = "v9"
    prompt_version: str             = "v9"
    semantic_tables_version: str    = "v9"
    config_version: str             = "v9"

    # ── Runtime state ─────────────────────────────────────────────────
    sources_used: list[str]         = field(default_factory=list)
    active_agents: list[str]        = field(default_factory=list)
    data_quality_score: float | None = None
    output_classification: str      = "INTERNAL"

    # ── Guardian summary ──────────────────────────────────────────────
    guardian_summary: dict          = field(default_factory=dict)
    # e.g. {"contradiction_count": 0, "evidence_grade": "strong", "policy_blocks": 0}

    # ── Evidence summary ──────────────────────────────────────────────
    evidence_summary: dict          = field(default_factory=dict)
    # e.g. {"confirmed": 2, "rejected": 1, "inconclusive": 1, "primary_conclusion": "..."}

    # ── Replay ────────────────────────────────────────────────────────
    replay_data_path: str | None    = None
    replay_context: dict            = field(default_factory=dict)
    replay_type: str                = "data_reloaded"
    # "exact" | "approximate" | "data_reloaded" | "offline"

    # ── Notes ─────────────────────────────────────────────────────────
    notes: list[str]                = field(default_factory=list)

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def create(cls, run_id: str) -> "RunManifest":
        return cls(run_id=run_id, created_at=datetime.utcnow().isoformat())

    def persist(self, base_dir: str | None = None) -> str:
        root = (
            Path(base_dir)
            if base_dir
            else Path(__file__).resolve().parent.parent / "memory" / "manifests"
        )
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return str(path)

    @classmethod
    def load(cls, run_id: str, base_dir: str | None = None) -> "RunManifest":
        root = (
            Path(base_dir)
            if base_dir
            else Path(__file__).resolve().parent.parent / "memory" / "manifests"
        )
        path = root / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No manifest for run_id={run_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        # Forward-compat: ignore unknown fields
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
