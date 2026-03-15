from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
import json

@dataclass
class RunManifest:
    run_id: str
    created_at: str
    metric_registry_version: str = 'v7'
    policy_version: str = 'v7'
    prompt_version: str = 'v7'
    semantic_tables_version: str = 'v7'
    sources_used: list[str] = field(default_factory=list)
    active_agents: list[str] = field(default_factory=list)
    data_quality_score: float | None = None
    notes: list[str] = field(default_factory=list)
    replay_data_path: str | None = None
    replay_context: dict = field(default_factory=dict)
    def to_dict(self) -> dict:
        return asdict(self)
    @classmethod
    def create(cls, run_id: str) -> 'RunManifest':
        return cls(run_id=run_id, created_at=datetime.utcnow().isoformat())
    def persist(self, base_dir: str | None = None) -> str:
        root = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent / 'memory' / 'manifests'
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'{self.run_id}.json'
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding='utf-8')
        return str(path)
