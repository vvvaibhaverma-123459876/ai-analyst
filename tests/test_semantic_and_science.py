import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic.metric_registry import MetricRegistry
from science.evidence_registry import EvidenceRegistry, EvidenceRecord, EvidenceState
from science.uncertainty_model import UncertaintyModel
from guardian.contradiction_checker import ContradictionChecker
from agents.context import AgentResult


def test_metric_registry_resolves_alias_and_validates_grain():
    registry = MetricRegistry({
        "fd_payment_conversion": {
            "description": "FD payment conversion",
            "aggregation": "ratio",
            "numerator": "fd_success_users",
            "denominator": "eligible_users",
            "aliases": ["fd conversion", "payment conversion"],
            "allowed_grains": ["daily", "weekly"],
            "dimensions": ["bank", "platform"],
            "owner": "growth",
        }
    })

    assert registry.resolve("why did payment conversion drop?") == "fd_payment_conversion"
    assert registry.validate_grain("fd_payment_conversion", "daily") is True
    assert registry.validate_dimension("fd_payment_conversion", "bank") is True
    assert registry.validate_dimension("fd_payment_conversion", "source") is False


def test_evidence_registry_summarises_support_and_uncertainty():
    registry = EvidenceRegistry()
    records = [
        EvidenceRecord("h1", "trend", "conversion dropped meaningfully", True, 0.9),
        EvidenceRecord("h1", "funnel", "payment-stage weakness found", True, 0.8),
        EvidenceRecord("h1", "debate", "some ambiguity remains", None, 0.4),
    ]
    summary = registry.summarise(records)
    uncertainty = UncertaintyModel().assess(evidence_confidence=summary.confidence, n_evidence=len(records))

    assert summary.state == EvidenceState.SUPPORTED
    assert summary.confidence > 0.8
    assert uncertainty.level in {"low_uncertainty", "medium_uncertainty"}


def test_contradiction_checker_flags_opposite_directions():
    checker = ContradictionChecker()
    results = {
        "trend": AgentResult(agent="trend", status="success", summary="Revenue increased sharply", data={}),
        "anomaly": AgentResult(agent="anomaly", status="success", summary="Revenue dropped significantly", data={}),
    }
    contradictions = checker.detect(results)
    assert contradictions
    assert contradictions[0]["reason"]
