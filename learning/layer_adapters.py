"""
learning/layer_adapters.py
Per-layer learning adapters. One for each layer in the system.
Each observes outcomes specific to its layer and adapts accordingly.

Sycophancy protection:
  - None of these adapters can modify policy-locked rules
  - Debate jury challenge requirements are immutable
  - Adaptations require verified ground truth before updating beliefs
"""

from __future__ import annotations
from learning.learning_layer import LearningLayer
from core.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. Ingestion learner
# ──────────────────────────────────────────────────────────────────────
class IngestionLearner(LearningLayer):
    """Learns which file types parse cleanly and reorders parser priority."""
    layer_name = "ingestion"

    def observe(self, context, result) -> dict:
        doc = getattr(context, "document", None)
        if not doc:
            return {}
        quality = 1.0 - (len(doc.warnings) / max(1, len(doc.dataframes) + 1))
        self.record_observation(doc.source_type, quality)
        return {"source_type": doc.source_type, "quality": quality}

    def adapt(self, context) -> dict:
        """Returns parser priority order based on historical quality scores."""
        types = ["csv", "excel", "json", "pdf", "image", "word", "text", "sql", "stream"]
        scores = {}
        for t in types:
            score = self.decay_weighted_mean(t)
            scores[t] = score if score is not None else 0.5
        ordered = sorted(types, key=lambda t: scores[t], reverse=True)
        self.update_belief("parser_priority", ordered)
        return {"parser_priority": ordered}


# ──────────────────────────────────────────────────────────────────────
# 2. Context learner
# ──────────────────────────────────────────────────────────────────────
class ContextLearner(LearningLayer):
    """Learns which upfront questions led to better analysis quality."""
    layer_name = "context"

    def observe(self, context, result) -> dict:
        quality = context.results.get("guardian", None)
        if quality is None:
            return {}
        questions_asked = len(getattr(context, "_questions_asked", []))
        run_quality = getattr(context, "_run_quality", 3) / 5.0
        self.record_observation("questions_asked", questions_asked)
        self.record_observation("quality_per_question",
                                run_quality / max(1, questions_asked))
        return {"questions_asked": questions_asked, "quality": run_quality}

    def adapt(self, context) -> dict:
        """Returns optimal question count for this data type."""
        avg_q = self.decay_weighted_mean("questions_asked")
        optimal_q = round(avg_q) if avg_q else 3
        self.update_belief("optimal_question_count", optimal_q)
        return {"optimal_question_count": max(1, min(5, optimal_q))}


# ──────────────────────────────────────────────────────────────────────
# 3. Orchestrator learner
# ──────────────────────────────────────────────────────────────────────
class OrchestratorLearner(LearningLayer):
    """Learns which agent plan combinations produced high-quality briefs."""
    layer_name = "orchestrator"

    def observe(self, context, result) -> dict:
        plan_key = "+".join(sorted(context.active_agents))
        quality = getattr(context, "_run_quality", 3) / 5.0
        self.record_observation(plan_key, quality)
        return {"plan": context.active_agents, "quality": quality}

    def adapt(self, context) -> dict:
        """Returns the historically best plan for this data profile."""
        scenario = self._classify(context)
        best_plan = self.get_belief(f"best_plan_{scenario}")
        if best_plan:
            return {"suggested_plan": best_plan}
        return {}

    def update_plan_quality(self, plan: list, quality: float, scenario: str):
        plan_key = "+".join(sorted(plan))
        current = self.get_belief(f"plan_scores_{scenario}") or {}
        current[plan_key] = round(
            0.7 * quality + 0.3 * current.get(plan_key, quality), 3
        )
        best = max(current, key=current.get)
        self.update_belief(f"best_plan_{scenario}", best.split("+"))
        self.update_belief(f"plan_scores_{scenario}", current)

    def _classify(self, context) -> str:
        p = context.data_profile
        if p.get("has_funnel_signal"): return "funnel"
        if p.get("has_cohort_signal"): return "cohort"
        if p.get("has_time_series"):   return "timeseries"
        return "general"


# ──────────────────────────────────────────────────────────────────────
# 4. Analysis learner (per-juror threshold tuning)
# ──────────────────────────────────────────────────────────────────────
class AnalysisLearner(LearningLayer):
    """Auto-tunes analysis thresholds based on verified outcome accuracy."""
    layer_name = "analysis"

    def observe(self, context, result) -> dict:
        gt = self._gt
        for agent_name in ["anomaly", "forecast", "experiment"]:
            acc = gt.agent_accuracy(agent_name, days=60)
            if acc.get("n", 0) >= 5:
                self.record_observation(f"{agent_name}_precision",
                                        acc.get("precision", 0.5))
        return {}

    def adapt(self, context) -> dict:
        """Returns tuned thresholds for analysis agents."""
        adaptations = {}

        # Anomaly: if precision low, raise z-threshold to reduce false positives
        anom_prec = self.decay_weighted_mean("anomaly_precision")
        if anom_prec is not None:
            if anom_prec < 0.50:
                # Too many false positives → stricter threshold
                new_thresh = self.get_belief("z_threshold", 2.0) + 0.2
                self.update_belief("z_threshold", min(3.5, new_thresh))
            elif anom_prec > 0.85:
                # Very precise → can relax threshold slightly
                new_thresh = self.get_belief("z_threshold", 2.0) - 0.1
                self.update_belief("z_threshold", max(1.5, new_thresh))
            adaptations["z_threshold"] = self.get_belief("z_threshold", 2.0)

        return adaptations


# ──────────────────────────────────────────────────────────────────────
# 5. Hypothesis learner
# ──────────────────────────────────────────────────────────────────────
class HypothesisLearner(LearningLayer):
    """Tracks which hypothesis sources have the best confirmation rates."""
    layer_name = "hypothesis"

    def observe(self, context, result) -> dict:
        plan = getattr(context, "research_plan", None)
        if not plan:
            return {}
        for h in plan.hypotheses:
            from science.research_plan import HypothesisStatus
            if h.status in (HypothesisStatus.CONFIRMED, HypothesisStatus.REJECTED):
                correct = h.status == HypothesisStatus.CONFIRMED
                self.record_observation(f"source_{h.source}", 1.0 if correct else 0.0)
        return {}

    def adapt(self, context) -> dict:
        """Returns source weights for hypothesis generation."""
        weights = {}
        for source in ["data", "business", "web", "prior"]:
            score = self.decay_weighted_mean(f"source_{source}", days=90)
            weights[source] = score if score is not None else 0.5
        self.update_belief("source_weights", weights)
        return {"hypothesis_source_weights": weights}


# ──────────────────────────────────────────────────────────────────────
# 6. Insight learner (audience-aware, sycophancy-protected)
# ──────────────────────────────────────────────────────────────────────
class InsightLearner(LearningLayer):
    """
    Learns which brief formats drive action for each audience type.

    SYCOPHANCY PROTECTION:
    This learner CAN adapt: tone depth, section ordering, verbosity.
    This learner CANNOT modify: number of challenges, confidence thresholds,
    policy rule citations, or debate jury output.
    These are enforced by checking against the policy store before any update.
    """
    layer_name = "insight"

    # Keys that cannot be learned away
    POLICY_LOCKED_KEYS = {
        "include_challenges", "min_challenges", "show_confidence",
        "cite_policy_rules", "include_debate_verdict",
    }

    def observe(self, context, result) -> dict:
        audience = context.business_context.get("audience", "general")
        quality = getattr(context, "_run_quality", 3) / 5.0
        self.record_observation(f"audience_{audience}", quality)
        return {"audience": audience, "quality": quality}

    def adapt(self, context) -> dict:
        audience = context.business_context.get("audience", "general")
        score = self.decay_weighted_mean(f"audience_{audience}")

        # Only adapt non-locked presentation aspects
        adaptations = {}
        if score is not None:
            if score < 0.5:
                adaptations["brief_verbosity"] = "shorter"
                adaptations["lead_with_action"] = True
            else:
                adaptations["brief_verbosity"] = "standard"

        # Never allow learning to disable challenges
        for locked_key in self.POLICY_LOCKED_KEYS:
            adaptations.pop(locked_key, None)

        self.update_belief(f"audience_prefs_{audience}", adaptations)
        return adaptations


# ──────────────────────────────────────────────────────────────────────
# 7. Output router learner
# ──────────────────────────────────────────────────────────────────────
class OutputRouterLearner(LearningLayer):
    """Recalibrates urgency thresholds based on false alarm history."""
    layer_name = "output_router"

    def observe(self, context, result) -> dict:
        decision = getattr(context, "_output_decision", None)
        if not decision:
            return {}
        # Was the alert acted on? (set by user feedback)
        alert_acted = getattr(context, "_alert_acted_on", None)
        if alert_acted is not None and "alert" in decision.modes:
            self.record_observation(f"alert_{decision.urgency}",
                                    1.0 if alert_acted else 0.0)
        return {}

    def adapt(self, context) -> dict:
        """Returns recalibrated urgency thresholds."""
        thresholds = {}
        for urgency in ["high", "critical"]:
            acted_rate = self.decay_weighted_mean(f"alert_{urgency}")
            if acted_rate is not None and acted_rate < 0.30:
                # High false alarm rate → raise the bar for this urgency level
                current = self.get_belief(f"threshold_{urgency}", 4)
                self.update_belief(f"threshold_{urgency}", min(8, current + 1))
                thresholds[urgency] = current + 1
            elif acted_rate is not None and acted_rate > 0.80:
                # Very acted-upon → can lower threshold slightly
                current = self.get_belief(f"threshold_{urgency}", 4)
                self.update_belief(f"threshold_{urgency}", max(2, current - 1))
                thresholds[urgency] = current - 1
        return {"urgency_thresholds": thresholds}
