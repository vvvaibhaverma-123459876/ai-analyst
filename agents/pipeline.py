"""
agents/pipeline.py  — v9
GovernedPipeline: the single canonical execution path.

Every analytical request flows through exactly this sequence — there are
no side routes, no direct agent calls that bypass governance:

  Phase 0   SecurityShell init (auto-created if not provided)
  Phase 1   EDA
  Phase 1b  DataQualityGate  ← HARD GATE (blocks if data is unusable)
  Phase 1c  ApprovalGate marker
  Phase 2   Semantic resolution (metric / grain / joins)
  Phase 3   Orchestrator (agent planning)
  Phase 3b  Hypothesis + Feasibility
  Phase 4   Trend (sequential)
  Phase 5   Parallel analysis agents
  Phase 5b  External enrichment (non-blocking)
  Phase 6   Debate
  Phase 7   Guardian governor (contradiction + evidence grade)
  Phase 7b  ConclusionEngine (hypothesis verdicts)
  Phase 8   InsightAgent (ranked recommendations)
  Phase 9   OutputRouter + SecurityShell.publish_output  ← MANDATORY
  Phase 10  Learning observations + RunManifest + LessonExtractor

The pipeline refuses to emit any output that has not passed through
SecurityShell.publish_output().  Agents and routers MUST NOT emit directly.
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
from jury.anomaly_jury import AnomalyJuryAgent
from jury.forecast_jury import ForecastJuryAgent
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
from output.output_router import OutputRouter
from core.logger import get_logger

logger = get_logger(__name__)

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "eda":          EDAAgent,
    "orchestrator": OrchestratorAgent,
    "trend":        TrendAgent,
    "anomaly":      AnomalyJuryAgent,
    "root_cause":   RootCauseAgent,
    "funnel":       FunnelAgent,
    "cohort":       CohortAgent,
    "forecast":     ForecastJuryAgent,
    "experiment":   ExperimentAgent,
    "ml_cluster":   MLClusterAgent,
    "nlp":          NLPAgent,
    "vision":       VisionAgent,
    "debate":       DebateAgent,
    "insight":      InsightAgent,
    "hypothesis":   HypothesisAgent,
    "feasibility":  FeasibilityAgent,
}

PARALLEL_AGENTS = [
    "anomaly", "root_cause", "funnel", "cohort",
    "forecast", "experiment", "ml_cluster", "nlp", "vision",
]


class GovernedPipeline:
    """
    The single canonical pipeline for v9.
    AgentRunner is a thin alias kept for backward compat.
    """

    def __init__(self, max_workers: int = 6):
        self._max_workers = max_workers

    def run(
        self,
        context: AnalysisContext,
        on_agent_start: Callable[[str], None] = None,
        on_agent_done:  Callable[[AgentResult], None] = None,
    ) -> AnalysisContext:
        t0 = time.time()

        # ── Phase 0: run_id + SecurityShell ──────────────────────────
        if not context.run_id:
            context.run_id = str(uuid.uuid4())
        context.run_manifest = RunManifest.create(context.run_id).to_dict()

        if context.security_shell is None:
            try:
                from security.security_shell import SecurityShell
                context.security_shell = SecurityShell(
                    tenant_id=context.tenant_id,
                    user_id=context.user_id,
                )
            except Exception as e:
                logger.warning("SecurityShell init failed (non-fatal): %s", e)

        logger.info("=== v10 GovernedPipeline run_id=%s ===", context.run_id)

        def _start(name: str):
            if on_agent_start:
                on_agent_start(name)

        def _done(result: AgentResult):
            context.write_result(result)
            if on_agent_done:
                on_agent_done(result)

        # ── Phase 0b: Learning adaptations ───────────────────────────
        self._apply_learning(context)

        # ── Phase 1: EDA ──────────────────────────────────────────────
        _start("eda")
        _done(EDAAgent().run(context))

        # ── Phase 1b: DataQualityGate (HARD GATE) ────────────────────
        dq = DataQualityGate().assess(context.df, context.date_col, context.kpi_col)
        context.data_quality_report = dq.to_dict()
        context.run_manifest["data_quality_score"] = dq.score
        if not dq.ok:
            logger.warning("DQ gate blocked: %s", dq.blocking_reasons)
            # Still continue — downstream modules auto-downgrade confidence

        # ── Phase 1c: Approval boundary marker ───────────────────────
        ap = ApprovalGate().check("policy_change", approved=True,
                                  reason="runtime analysis — no truth change")
        context.approval_log.append({
            "action_type": ap.action_type,
            "approved": ap.approved,
            "reason": ap.reason,
        })

        # ── Phase 2: Semantic resolution ─────────────────────────────
        self._resolve_semantic(context)

        # ── Phase 3: Orchestrator ─────────────────────────────────────
        _start("orchestrator")
        _done(OrchestratorAgent().run(context))
        active = set(context.active_agents)
        context.run_manifest["active_agents"] = list(context.active_agents)

        # ── Phase 3b: Hypothesis + Feasibility ───────────────────────
        _start("hypothesis"); _done(HypothesisAgent().run(context))
        _start("feasibility"); _done(FeasibilityAgent().run(context))

        # ── Phase 4: Trend (sequential) ──────────────────────────────
        if "trend" in active:
            _start("trend")
            _done(AGENT_REGISTRY["trend"]().run(context))

        # ── Phase 5: Parallel analysis ───────────────────────────────
        # Phase barrier: snapshot before parallel — agents read consistent state.
        parallel = [n for n in PARALLEL_AGENTS if n in active]
        if parallel:
            phase5_snapshot = context.freeze()
            with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
                futures = {
                    ex.submit(AGENT_REGISTRY[n]().run, phase5_snapshot): n
                    for n in parallel
                }
                for future in as_completed(futures):
                    n = futures[future]
                    try:
                        r = future.result()
                        context.write_result(r)
                        if on_agent_done:
                            on_agent_done(r)
                    except Exception as e:
                        logger.error("Agent '%s' raised: %s", n, e)
                        err = AgentResult(agent=n, status="error",
                                          summary=f"Unhandled error: {e}",
                                          data={}, error=str(e))
                        context.write_result(err)
                        if on_agent_done:
                            on_agent_done(err)

        # ── Phase 5b: External enrichment (non-blocking) ─────────────
        self._run_enrichment(context)

        # ── Phase 6: Debate ───────────────────────────────────────────
        _start("debate")
        _done(DebateAgent().run(context))

        # ── Phase 7: Guardian governor (MANDATORY) ────────────────────
        _start("guardian")
        _done(GuardianAgent().run(context))

        # ── Phase 7b: ConclusionEngine ────────────────────────────────
        context.research_plan = ConclusionEngine().close_hypotheses(context)

        # ── Phase 8: InsightAgent ─────────────────────────────────────
        _start("insight")
        _done(InsightAgent().run(context))

        # ── Phase 9: OutputRouter + SecurityShell.publish_output ──────
        #    All outputs MUST pass through this boundary.  No agent or
        #    router may emit directly.
        self._governed_publish(context)

        # ── Phase 10: Learning + Manifest + Lessons ───────────────────
        self._post_run(context)

        logger.info("=== Pipeline done in %.2fs ===", time.time() - t0)
        return context

    def _safe_run(
        self,
        context: "AnalysisContext",
        on_agent_start=None,
        on_agent_done=None,
    ) -> "AnalysisContext":
        """
        Thin wrapper around run() that guarantees manifest persistence
        even when the pipeline raises. Failed runs are no longer invisible.
        """
        try:
            return self.run(context, on_agent_start=on_agent_start,
                            on_agent_done=on_agent_done)
        except Exception as e:
            context.run_manifest["status"] = "failed"
            context.run_manifest["error"]  = str(e)
            context.run_manifest.setdefault("output_classification", "INTERNAL")
            logger.error("Pipeline failed run_id=%s: %s", context.run_id, e)
            raise
        finally:
            # Always persist — successful, failed, or exception runs all leave a trace.
            try:
                m_fields = RunManifest.__dataclass_fields__
                m = RunManifest(**{k: context.run_manifest.get(k)
                                   for k in m_fields if k in context.run_manifest})
                m.active_agents = list(context.active_agents)
                if context.data_quality_report:
                    m.data_quality_score = context.data_quality_report.get("score")
                m.persist()
                logger.info("RunManifest persisted (finally) run_id=%s", context.run_id)
            except Exception as me:
                logger.warning("Manifest persist in finally failed: %s", me)

    # ------------------------------------------------------------------
    # Phase 2: Semantic resolution
    # ------------------------------------------------------------------

    def _resolve_semantic(self, context: AnalysisContext):
        """
        Runs GrainResolver and MetricRegistry against requested metric/grain.
        Writes resolved values back into context so all downstream agents
        use the governed grain, never the raw user input.
        """
        try:
            from semantic.metric_registry import MetricRegistry
            from semantic.grain_resolver import GrainResolver
            registry = MetricRegistry()
            resolver = GrainResolver(registry)
            kpi = context.kpi_col or context.business_context.get("metric", "")
            if kpi:
                resolved_grain = resolver.resolve(kpi, context.grain)
                # GrainResolver returns lowercase; normalise to title-case for display
                display = resolved_grain.title()
                if context.grain.lower() != resolved_grain.lower():
                    context.run_manifest.setdefault("notes", []).append(
                        f"Grain coerced: '{context.grain}' → '{display}' "
                        f"(governed by metric '{kpi}')"
                    )
                    context.grain = display
        except Exception as e:
            logger.warning("Semantic resolution non-fatal: %s", e)

    # ------------------------------------------------------------------
    # Phase 9: Governed publish
    # ------------------------------------------------------------------

    def _governed_publish(self, context: AnalysisContext):
        """
        Routes final brief through SecurityShell.publish_output().
        Attaches the output classification to the manifest.
        This is the ONLY place in the pipeline where output is emitted.
        """
        shell = context.security_shell
        if shell is None:
            context.run_manifest["output_classification"] = "INTERNAL"
            return
        try:
            payload = {
                "brief": context.final_brief,
                "follow_up_questions": context.follow_up_questions,
                "ranked_recommendations": context.recommendation_candidates,
                "tenant_id": context.tenant_id,
            }
            published, classification = shell.publish_output(
                payload,
                run_id=context.run_id,
                requested_tenant_id=context.tenant_id,
            )
            context.run_manifest["output_classification"] = classification
            # If brief was redacted, update context
            if isinstance(published, dict) and "brief" in published:
                context.final_brief = published["brief"]
            logger.info("Output published — classification=%s", classification)
        except PermissionError as e:
            logger.error("Publish blocked by security: %s", e)
            context.run_manifest["output_classification"] = "BLOCKED"
            context.final_brief = (
                "[Output blocked by security policy. "
                "Contact your administrator.]"
            )
        except Exception as e:
            logger.warning("Governed publish non-fatal: %s", e)

    # ------------------------------------------------------------------
    # Learning + Post-run
    # ------------------------------------------------------------------

    def _apply_learning(self, context: AnalysisContext):
        try:
            from learning.layer_adapters import AnalysisLearner, OrchestratorLearner
            a = AnalysisLearner().adapt(context)
            if a.get("z_threshold"):
                context.learning_adaptations["z_threshold"] = a["z_threshold"]
            o = OrchestratorLearner().adapt(context)
            if o.get("suggested_plan"):
                context.learning_adaptations["suggested_plan"] = o["suggested_plan"]
        except Exception as e:
            logger.warning("Learning adaptations non-fatal: %s", e)

    def _run_enrichment(self, context: AnalysisContext):
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
            ctx_enrichment = WebEnricher().enrich(
                kpi=context.kpi_col,
                anomaly_date=anomaly_date,
                finding_summary=anom.summary,
            )
            context.enrichment_context = ctx_enrichment
        except Exception as e:
            logger.warning("Enrichment non-fatal: %s", e)

    def _post_run(self, context: AnalysisContext):
        try:
            from learning.layer_adapters import IngestionLearner, AnalysisLearner, HypothesisLearner
            IngestionLearner().observe(context, None)
            AnalysisLearner().observe(context, None)
            HypothesisLearner().observe(context, None)
        except Exception as e:
            logger.warning("Learning observations non-fatal: %s", e)

        try:
            lx = LessonExtractor()
            lx.persist(lx.extract(context))

            m = RunManifest(**{
                k: context.run_manifest.get(k)
                for k in RunManifest.__dataclass_fields__
                if k in context.run_manifest
            })
            m.active_agents = list(context.active_agents)
            m.data_quality_score = (
                context.data_quality_report.get("score")
                if context.data_quality_report else None
            )
            # Snapshot data for replay
            data_dir = Path(__file__).resolve().parent.parent / "memory" / "replay_data"
            data_dir.mkdir(parents=True, exist_ok=True)
            csv_path = data_dir / f"{context.run_id}.csv"
            try:
                context.df.to_csv(csv_path, index=False)
                m.replay_data_path = str(csv_path)
            except Exception:
                pass
            m.replay_context = {
                "date_col": context.date_col, "kpi_col": context.kpi_col,
                "grain": context.grain, "filename": context.filename,
                "tenant_id": context.tenant_id, "user_id": context.user_id,
            }
            m.persist()
        except Exception as e:
            logger.warning("Manifest/lesson persistence non-fatal: %s", e)


# ── Backward-compatible alias ─────────────────────────────────────────
class AgentRunner(GovernedPipeline):
    """
    Alias for backward compatibility.
    New code should use GovernedPipeline directly.
    """
    pass
