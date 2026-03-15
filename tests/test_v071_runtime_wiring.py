import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.context import AnalysisContext, AgentResult
from agents.insight_agent import InsightAgent
from security.security_shell import SecurityShell
from evaluation.replay_harness import ReplayHarness
from versioning.run_manifest import RunManifest


def test_insight_agent_populates_and_ranks_recommendations():
    ctx = AnalysisContext(
        df=pd.DataFrame({'d': pd.date_range('2026-01-01', periods=3), 'k': [1, 2, 3]}),
        date_col='d',
        kpi_col='k',
    )
    ctx.write_result(AgentResult(agent='trend', status='success', summary='KPI increased', data={}))
    ctx.write_result(AgentResult(agent='root_cause', status='success', summary='Platform drag', data={'movers': {'negative': [{'dimension': 'platform', 'value': 'android'}]}}))
    res = InsightAgent().run(ctx)
    assert res.status == 'success'
    assert ctx.recommendation_candidates
    assert res.data['ranked_recommendations']


def test_security_shell_publish_output_redacts_confidential_payload():
    shell = SecurityShell(tenant_id='acme', user_id='u1', role='analyst')
    payload, classification = shell.publish_output({'brief': 'Contact me at test@example.com', 'tenant_id': 'acme'}, requested_tenant_id='acme')
    assert classification in {'CONFIDENTIAL', 'RESTRICTED'}
    assert '[EMAIL_REDACTED]' in payload['brief']


def test_replay_harness_can_replay_manifest(tmp_path):
    data = pd.DataFrame({'d': pd.date_range('2026-01-01', periods=5), 'k': [1, 2, 3, 4, 5]})
    csv_path = tmp_path / 'run.csv'
    data.to_csv(csv_path, index=False)
    manifest = RunManifest.create('r1')
    manifest.replay_data_path = str(csv_path)
    manifest.replay_context = {'date_col': 'd', 'kpi_col': 'k', 'grain': 'Daily', 'tenant_id': 'default', 'user_id': 'system'}
    manifest.persist(base_dir=str(tmp_path))
    out = ReplayHarness(base_dir=str(tmp_path)).replay('r1')
    assert out.run_id.endswith('-replay')
    assert 'insight' in out.results
