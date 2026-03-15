from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

REQUIRES_APPROVAL = {
    "metric_definition_change",
    "join_path_change",
    "policy_change",
    "source_authority_change",
    "alert_threshold_change",
    "promote_learned_truth",
}

@dataclass
class ApprovalDecision:
    approved: bool
    reason: str
    action_type: str

class ApprovalGate:
    def __init__(self, base_dir: str | None = None):
        root = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent / "memory"
        root.mkdir(parents=True, exist_ok=True)
        self._path = root / "approvals.jsonl"
    def requires_approval(self, action_type: str) -> bool:
        return action_type in REQUIRES_APPROVAL
    def check(self, action_type: str, approved: bool = False, reason: str = "") -> ApprovalDecision:
        if self.requires_approval(action_type) and not approved:
            return ApprovalDecision(False, reason or "human approval required", action_type)
        return ApprovalDecision(True, reason or "approved or no approval required", action_type)
    def log_request(self, action_type: str, payload: dict, approved: bool = False, approver: str = "") -> None:
        rec = {"ts": datetime.utcnow().isoformat(), "action_type": action_type, "approved": approved, "approver": approver, "payload": payload}
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
