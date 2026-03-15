import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality.data_quality_gate import DataQualityGate
from governance.approval_gate import ApprovalGate
from insights.recommendation_ranker import RecommendationRanker
from agents.context import AnalysisContext
from science.research_plan import ResearchPlan, Hypothesis, HypothesisStatus
from science.conclusion_engine import ConclusionEngine


def test_data_quality_gate_blocks_obviously_bad_data():
    df = pd.DataFrame({'x': [None, None], 'd': ['bad', 'bad']})
    report = DataQualityGate().assess(df, 'd', 'x')
    assert report.ok is False
    assert report.score <= 0.35


def test_approval_gate_requires_human_boundary_for_truth_change():
    gate = ApprovalGate(base_dir=str(ROOT / 'memory'))
    decision = gate.check('metric_definition_change', approved=False)
    assert decision.approved is False


def test_recommendation_ranker_prefers_high_confidence_high_value_actions():
    ranked = RecommendationRanker().rank([
        {'action': 'A', 'confidence': 0.9, 'urgency': 0.8, 'business_value': 0.9, 'effort': 0.2},
        {'action': 'B', 'confidence': 0.5, 'urgency': 0.5, 'business_value': 0.4, 'effort': 0.1},
    ])
    assert ranked[0].action == 'A'


def test_conclusion_engine_downgrades_with_poor_data_quality():
    ctx = AnalysisContext(df=pd.DataFrame({'d': pd.date_range('2026-01-01', periods=5), 'k': [1, 2, 3, 4, 5]}), date_col='d', kpi_col='k')
    ctx.data_quality_report = {'score': 0.3}
    h = Hypothesis(id='h1', statement='Conversion dropped', source='data', status=HypothesisStatus.TESTABLE)
    h.evidence = [
        {'agent': 'trend', 'summary': 'conversion dropped', 'supports': True, 'confidence': 0.9},
        {'agent': 'funnel', 'summary': 'payment-stage weakness found', 'supports': True, 'confidence': 0.9},
    ]
    plan = ResearchPlan(hypotheses=[h])
    ctx.research_plan = plan
    out = ConclusionEngine().close_hypotheses(ctx)
    assert out.hypotheses[0].confidence <= 0.8
