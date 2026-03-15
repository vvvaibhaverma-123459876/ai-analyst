"""
agents/runner.py — v0.5
Full pipeline with all new components integrated.

Execution order:
  Phase 0: Security shell + learning adaptations (setup)
  Phase 1: EDA (sync)
  Phase 2: Orchestrator (sync)
  Phase 2b: Hypothesis + Feasibility (sync, scientific reasoning)
  Phase 3: Trend (sync — builds context.ts)
  Phase 4: PARALLEL — anomaly, root_cause, funnel, cohort,
                       forecast, experiment, ml_cluster, nlp, vision
  Phase 4b: External enrichment (async, non-blocking)
  Phase 5: Debate (sync)
  Phase 5b: Guardian governor (sync — policy enforcement + scoring)
  Phase 5c: Conclusion engine (closes hypotheses)
  Phase 6: Insight (sync — final brief)
  Phase 7: Learning observation (post-run, non-blocking)
"""

from __future__ import annotations
from pathlib import Path
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from agents.context import AnalysisContext, AgentResult
from agents.base_agent import BaseAgent
from agents.eda_agent import EDAAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.trend_agent import TrendAgent
from agents.root_cause_agent import RootCauseAgent
from agents.funnel_agent import FunnelAgent
from agents.cohort_agent import CohortAgent
from agents.nlp_agent import NLPAgent
from agents.vision_agent import VisionAgent
from agents.debate_agent import DebateAgent
from agents.insight_agent import InsightAgent

# Jury-based agents (replace v0.4 single-method agents)
from jury.anomaly_jury import AnomalyJuryAgent
from jury.forecast_jury import ForecastJuryAgent

# v0.4 agents that don't yet have juries (use as-is)
from agents.experiment_agent import ExperimentAgent
from agents.ml_cluster_agent import MLClusterAgent

from science.hypothesis_agent import HypothesisAgent
from science.feasibility_agent import FeasibilityAgent
from science.conclusion_engine import ConclusionEngine
from guardian.guardian_agent import GuardianAgent
from quality.data_quality_gate import DataQualityGate
from versioning.run_manifest import RunManifest
from governance.approval_gate import ApprovalGate
from guardian.lesson_extractor import LessonExtractor
from core.logger import get_logger

logger = get_logger(__name__)

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "eda":          EDAAgent,
    "orchestrator": OrchestratorAgent,
    "trend":        TrendAgent,
    "anomaly":      AnomalyJuryAgent,          # jury version
    "root_cause":   RootCauseAgent,
    "funnel":       FunnelAgent,
    "cohort":       CohortAgent,
    "forecast":     ForecastJuryAgent,         # jury version
    "experiment":   ExperimentAgent,
    "ml_cluster":   MLClusterAgent,
    "nlp":          NLPAgent,
    "vision":       VisionAgent,
    "debate":       DebateAgent,
    "insight":      InsightAgent,
    "hypothesis":   HypothesisAgent,
    "feasibility":  FeasibilityAgent,
}

SEQUENTIAL_BEFORE = ["trend"]
PARALLEL_AGENTS   = ["anomaly", "root_cause", "funnel", "cohort",
                      "forecast", "experiment", "ml_cluster", "nlp", "vision"]
SEQUENTIAL_AFTER  = ["debate", "insight"]


class AgentRunner:

    def __init__(self, max_workers: int = 6):
        self._max_workers = max_workers

    def run(
        self,
        context: AnalysisContext,
        on_agent_start: Callable[[str], None] = None,
        on_agent_done: Callable[[AgentResult], None] = None,
    ) -> AnalysisContext:
        t0 = time.time()

        # Assign run_id
        if not context.run_id:
            context.run_id = str(uuid.uuid4())
        context.run_manifest = RunManifest.create(context.run_id).to_dict()
        if context.security_shell is None:
            try:
                from security.security_shell import SecurityShell
                context.security_shell = SecurityShell(tenant_id=context.tenant_id, user_id=context.user_id)
            except Exception:
                pass

        logger.info(f"=== v0.7 pipeline starting run_id={context.run_id} ===")

        def _start(name: str):
            if on_agent_start: on_agent_start(name)

        def _done(result: AgentResult):
            context.write_result(result)
            if on_agent_done: on_agent_done(result)

        # Phase 0: Apply learning adaptations
        self._apply_learning_adaptations(context)

        # Phase 1: EDA
        _start("eda"); _done(EDAAgent().run(context))

        # Phase 1b: Data quality gate
        dq = DataQualityGate().assess(context.df, context.date_col, context.kpi_col)
        context.data_quality_report = dq.to_dict()
        context.run_manifest['data_quality_score'] = dq.score
        if not dq.ok:
            logger.warning(f"Data quality gate blockers: {dq.blocking_reasons}")

        # Phase 1c: approval boundary marker
        approval = ApprovalGate().check('policy_change', approved=True, reason='runtime analysis only')
        context.approval_log.append({'action_type': approval.action_type, 'approved': approval.approved, 'reason': approval.reason})

        # Phase 2: Orchestrator
        _start("orchestrator"); _done(OrchestratorAgent().run(context))

        active = set(context.active_agents)
        context.run_manifest['active_agents'] = list(context.active_agents)

        # Phase 2b: Scientific reasoning (hypothesis → feasibility)
        _start("hypothesis"); _done(HypothesisAgent().run(context))
        _start("feasibility"); _done(FeasibilityAgent().run(context))

        # Phase 3: Trend (sequential — builds context.ts)
        if "trend" in active:
            _start("trend"); _done(AGENT_REGISTRY["trend"]().run(context))

        # Phase 4: Parallel analysis agents
        parallel_to_run = [n for n in PARALLEL_AGENTS if n in active]
        if parallel_to_run:
            logger.info(f"Parallel agents: {parallel_to_run}")
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(AGENT_REGISTRY[n]().run, context): n
                    for n in parallel_to_run
                }
                for future in as_completed(futures):
                    n = futures[future]
                    try:
                        _done(future.result())
                    except Exception as e:
                        logger.error(f"Agent '{n}' raised: {e}")
                        _done(AgentResult(
                            agent=n, status="error",
                            summary=f"Unhandled error: {e}",
                            data={}, error=str(e),
                        ))

        # Phase 4b: External enrichment (non-blocking)
        self._run_enrichment(context)

        # Phase 5: Debate
        if "debate" in active or True:  # always run debate
            _start("debate"); _done(DebateAgent().run(context))

        # Phase 5b: Guardian governor
        _start("guardian")
        guardian_result = GuardianAgent().run(context)
        _done(guardian_result)

        # Phase 5c: Conclusion engine (closes hypotheses)
        ce = ConclusionEngine()
        context.research_plan = ce.close_hypotheses(context)

        # Phase 6: Insight
        _start("insight"); _done(InsightAgent().run(context))

        # Phase 7: Post-run learning observations
        self._run_learning_observations(context)

        # Phase 8: lesson extraction + manifest persist
        try:
            lx = LessonExtractor()
            lx.persist(lx.extract(context))
            manifest = RunManifest(**{k: context.run_manifest.get(k) for k in RunManifest.__dataclass_fields__.keys() if k in context.run_manifest})
            manifest.active_agents = list(context.active_agents)
            manifest.data_quality_score = context.data_quality_report.get('score') if context.data_quality_report else None
            data_dir = Path(__file__).resolve().parent.parent / 'memory' / 'replay_data'
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / f'{context.run_id}.csv'
            try:
                context.df.to_csv(data_path, index=False)
                manifest.replay_data_path = str(data_path)
            except Exception as _replay_e:
                logger.warning(f'Replay snapshot persist failed (non-fatal): {_replay_e}')
            manifest.replay_context = {
                'date_col': context.date_col,
                'kpi_col': context.kpi_col,
                'grain': context.grain,
                'filename': context.filename,
                'tenant_id': context.tenant_id,
                'user_id': context.user_id,
            }
            manifest.persist()
        except Exception as e:
            logger.warning(f"Manifest/lesson persistence failed (non-fatal): {e}")

        logger.info(f"=== Pipeline done in {time.time()-t0:.2f}s ===")
        return context

    def _apply_learning_adaptations(self, context: AnalysisContext):
        """Read current learning beliefs and apply adaptations to context."""
        try:
            from learning.layer_adapters import (
                AnalysisLearner, OrchestratorLearner, InsightLearner
            )
            analysis_adapt = AnalysisLearner().adapt(context)
            if analysis_adapt.get("z_threshold"):
                context.learning_adaptations["z_threshold"] = analysis_adapt["z_threshold"]

            orch_adapt = OrchestratorLearner().adapt(context)
            if orch_adapt.get("suggested_plan"):
                context.learning_adaptations["suggested_plan"] = orch_adapt["suggested_plan"]

        except Exception as e:
            logger.warning(f"Learning adaptations failed (non-fatal): {e}")

    def _run_enrichment(self, context: AnalysisContext):
        """Trigger external enrichment based on findings."""
        try:
            from enrichment.web_enricher import WebEnricher
            shell = context.security_shell
            if shell and not shell.check_external_call("PUBLIC", context.run_id):
                return

            anom = context.results.get("anomaly")
            if not anom or anom.status != "success":
                return

            records = anom.data.get("anomaly_records", [])
            if not records:
                return

            anomaly_date = str(records[0].get("date", ""))[:10]
            enricher = WebEnricher()
            ctx_enrichment = enricher.enrich(
                kpi=context.kpi_col,
                anomaly_date=anomaly_date,
                finding_summary=anom.summary,
            )
            context.enrichment_context = ctx_enrichment
            logger.info(f"Enrichment: {ctx_enrichment.summary}")

        except Exception as e:
            logger.warning(f"Enrichment failed (non-fatal): {e}")

    def _run_learning_observations(self, context: AnalysisContext):
        """Record observations for all learning layers."""
        try:
            from learning.layer_adapters import (
                IngestionLearner, AnalysisLearner, HypothesisLearner
            )
            IngestionLearner().observe(context, None)
            AnalysisLearner().observe(context, None)
            HypothesisLearner().observe(context, None)
        except Exception as e:
            logger.warning(f"Learning observations failed (non-fatal): {e}")
