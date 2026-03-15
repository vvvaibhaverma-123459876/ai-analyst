from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


class ReplayHarness:
    def __init__(self, base_dir: str | None = None):
        self._root = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent / 'memory' / 'manifests'

    def load_manifest(self, run_id: str) -> dict:
        path = self._root / f'{run_id}.json'
        if not path.exists():
            raise FileNotFoundError(f'No manifest for run_id={run_id}')
        return json.loads(path.read_text(encoding='utf-8'))

    def replay(self, run_id: str):
        manifest = self.load_manifest(run_id)
        data_path = manifest.get('replay_data_path')
        if not data_path:
            raise ValueError('Manifest does not include replay_data_path; cannot replay run.')
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f'Replay data not found: {data_path}')

        from agents.context import AnalysisContext
        from agents.runner import AgentRunner
        from security.security_shell import SecurityShell

        df = pd.read_csv(path)
        rc = manifest.get('replay_context', {})
        shell = SecurityShell(tenant_id=rc.get('tenant_id', 'default'), user_id=rc.get('user_id', 'system'), role=rc.get('role'))
        safe_df, _ = shell.process_dataframe(df, run_id=run_id)
        context = AnalysisContext(
            df=safe_df,
            date_col=rc.get('date_col', ''),
            kpi_col=rc.get('kpi_col', ''),
            grain=rc.get('grain', 'Daily'),
            filename=rc.get('filename', path.name),
            security_shell=shell,
            run_id=f'{run_id}-replay',
            tenant_id=rc.get('tenant_id', 'default'),
            user_id=rc.get('user_id', 'system'),
        )
        return AgentRunner().run(context)
