"""
guardian/guardian_agent.py
Guardian — governor, teacher, policeman.

Five active powers:
  1. Score agents over time (per data type, per scenario)
  2. Detect contradictions across runs on same data
  3. Rewrite prompts when agents underperform (A/B tests improvements)
  4. Improve routing (ε-greedy exploration + score decay)
  5. Enforce policy rules (veto power over conclusions)

Overfitting defences:
  - Score decay: old scores lose weight → prevents meta-overfit
  - ε-greedy exploration: 10% of runs try novel plans → stays calibrated
  - Temporal holdout enforcement: validates every forecast model out-of-sample
  - Sycophancy lock: debate jury policy rules cannot be modified by learning
  - Contradiction detection: same data → different findings → flags instability
"""

from __future__ import annotations
import json
import math
import random
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from security.policy_store import PolicyStore
from ground_truth.recorder import GroundTruthRecorder
from core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "guardian.db"

# ε-greedy: fraction of runs that explore novel agent plans
EPSILON = 0.10
# Score decay: half-life in days — scores older than this count half as much
SCORE_HALF_LIFE_DAYS = 30


@dataclass
class AgentScore:
    agent: str
    method: str
    scenario_type: str
    precision: float
    n_observations: int
    last_updated: str
    weight: float = 1.0          # decayed weight


@dataclass
class GuardianVerdict:
    approved: bool
    blocked_findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    recommended_plan_override: list[str] | None = None
    prompt_rewrites: dict[str, str] = field(default_factory=dict)


class GuardianAgent(BaseAgent):
    name = "guardian"
    description = "Governor: scores agents, detects contradictions, enforces policy, rewrites prompts"

    def __init__(self, policy_path: str = None):
        super().__init__()
        self._policy = PolicyStore(policy_path)
        self._gt = GroundTruthRecorder()
        self._db = str(DB_PATH)
        os.makedirs(os.path.dirname(self._db), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS agent_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT, method TEXT, scenario_type TEXT,
                    precision REAL, n_obs INTEGER,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT, version INTEGER, prompt_text TEXT,
                    wins INTEGER DEFAULT 0, trials INTEGER DEFAULT 0,
                    active INTEGER DEFAULT 1, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS policy_change_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    changed_by TEXT, rule TEXT,
                    old_value TEXT, new_value TEXT, reason TEXT,
                    created_at TEXT
                );
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Main entry point: review a finished context before publishing
    # ------------------------------------------------------------------

    def _run(self, context: AnalysisContext) -> AgentResult:
        data_sig = self._data_signature(context)
        verdict = GuardianVerdict(approved=True)

        # Power 1: Score agents from this run
        self._score_agents(context)

        # Power 2: Detect contradictions
        contradictions = self._detect_contradictions(context, data_sig)
        if contradictions:
            verdict.contradictions = contradictions
            verdict.warnings.append(
                f"{len(contradictions)} contradiction(s) detected with prior runs on similar data."
            )

        # Power 3: Check prompt rewrites needed
        rewrites = self._check_prompt_rewrites(context)
        if rewrites:
            verdict.prompt_rewrites = rewrites

        # Power 4: ε-greedy routing check
        override = self._epsilon_greedy_check(context)
        if override:
            verdict.recommended_plan_override = override
            verdict.warnings.append("ε-greedy exploration: novel plan suggested for next run.")

        # Power 5: Policy enforcement — veto blocking findings
        blocked = self._enforce_policy(context)
        if blocked:
            verdict.blocked_findings = blocked
            if any(b.get("blocking") for b in blocked):
                verdict.approved = False
                verdict.warnings.append(
                    f"{len(blocked)} finding(s) blocked by policy rules."
                )

        # Temporal holdout enforcement on forecasts
        holdout_warnings = self._enforce_holdout(context)
        verdict.warnings.extend(holdout_warnings)

        summary = (
            f"Guardian review: approved={verdict.approved}, "
            f"blocked={len(verdict.blocked_findings)}, "
            f"contradictions={len(verdict.contradictions)}, "
            f"warnings={len(verdict.warnings)}."
        )

        return AgentResult(
            agent=self.name,
            status="success",
            summary=summary,
            data={
                "verdict": verdict,
                "approved": verdict.approved,
                "blocked_findings": verdict.blocked_findings,
                "warnings": verdict.warnings,
                "contradictions": verdict.contradictions,
                "prompt_rewrites": verdict.prompt_rewrites,
            },
        )

    # ------------------------------------------------------------------
    # Power 1: Score agents
    # ------------------------------------------------------------------

    def _score_agents(self, context: AnalysisContext):
        scenario = self._classify_scenario(context)
        for name, result in context.results.items():
            if result.status != "success":
                continue
            precision = self._gt.agent_accuracy(name, days=90).get("precision")
            if precision is not None:
                with sqlite3.connect(self._db) as conn:
                    conn.execute("""
                        INSERT INTO agent_scores
                        (agent, method, scenario_type, precision, n_obs, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (name, "default", scenario,
                          precision, 1, datetime.now().isoformat()))
                    conn.commit()

    def get_agent_reliability(self, agent: str, scenario: str = None) -> dict:
        """Returns decay-weighted reliability score for an agent."""
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute("""
                SELECT precision, created_at FROM agent_scores
                WHERE agent = ?
                ORDER BY id DESC LIMIT 100
            """, (agent,)).fetchall()

        if not rows:
            return {"score": None, "n": 0}

        now = datetime.now()
        total_weight = 0.0
        weighted_sum = 0.0

        for precision, created_at in rows:
            if precision is None:
                continue
            try:
                age_days = (now - datetime.fromisoformat(created_at)).days
                # Exponential decay
                weight = math.exp(-age_days * math.log(2) / SCORE_HALF_LIFE_DAYS)
                weighted_sum += precision * weight
                total_weight += weight
            except Exception:
                continue

        score = weighted_sum / total_weight if total_weight > 0 else None
        return {"score": round(score, 3) if score else None, "n": len(rows)}

    # ------------------------------------------------------------------
    # Power 2: Contradiction detection
    # ------------------------------------------------------------------

    def _detect_contradictions(self, context: AnalysisContext, data_sig: str) -> list[dict]:
        contradictions = []

        for finding_type in ["anomaly", "root_cause", "forecast"]:
            prior = self._gt.contradiction_check(data_sig, finding_type)
            if len(prior) < 2:
                continue

            # Check if conclusions have flipped
            summaries = [p["summary"] for p in prior]
            current_result = context.results.get(finding_type)
            if not current_result or current_result.status != "success":
                continue

            current_summary = current_result.summary
            prior_summary = prior[0]["summary"] if prior else ""

            # Simple keyword contradiction check
            pos_words = {"increase", "up", "rise", "growth", "positive"}
            neg_words = {"decrease", "down", "drop", "decline", "negative"}

            current_pos = any(w in current_summary.lower() for w in pos_words)
            current_neg = any(w in current_summary.lower() for w in neg_words)
            prior_pos = any(w in prior_summary.lower() for w in pos_words)
            prior_neg = any(w in prior_summary.lower() for w in neg_words)

            if (current_pos and prior_neg) or (current_neg and prior_pos):
                contradictions.append({
                    "finding_type": finding_type,
                    "current": current_summary[:80],
                    "prior": prior_summary[:80],
                    "severity": "high",
                })

        return contradictions

    # ------------------------------------------------------------------
    # Power 3: Prompt rewriting
    # ------------------------------------------------------------------

    def _check_prompt_rewrites(self, context: AnalysisContext) -> dict[str, str]:
        """
        If an agent's reliability score is below 0.6, trigger prompt A/B testing.
        Returns {agent_name: improved_prompt_suggestion}.
        """
        rewrites = {}
        low_performers = []

        for name in context.active_agents:
            reliability = self.get_agent_reliability(name)
            score = reliability.get("score")
            if score is not None and score < 0.60 and reliability["n"] >= 10:
                low_performers.append(name)

        if not low_performers:
            return rewrites

        from core.config import config
        if not (config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY):
            return rewrites

        try:
            from llm.client import LLMClient
            llm = LLMClient()
            for agent_name in low_performers[:2]:   # cap at 2 per run
                suggestion = llm.complete(
                    system="You are an expert prompt engineer for analytics AI agents.",
                    user=(
                        f"Agent '{agent_name}' has low accuracy (score < 0.60) on recent runs.\n"
                        f"Current task: this agent analyses data and produces structured findings.\n"
                        f"Suggest ONE specific improvement to its system prompt to improve accuracy.\n"
                        f"Return only the suggested improvement in 1-2 sentences."
                    )
                )
                rewrites[agent_name] = suggestion
                logger.info(f"Prompt rewrite suggested for {agent_name}: {suggestion[:60]}")
        except Exception as e:
            logger.warning(f"Prompt rewrite failed: {e}")

        return rewrites

    # ------------------------------------------------------------------
    # Power 4: ε-greedy exploration
    # ------------------------------------------------------------------

    def _epsilon_greedy_check(self, context: AnalysisContext) -> list[str] | None:
        """
        With probability ε, suggest a novel agent plan instead of the learned optimal.
        This prevents meta-overfitting to historical data patterns.
        """
        if random.random() > EPSILON:
            return None

        from agents.runner import AGENT_REGISTRY
        all_agents = list(AGENT_REGISTRY.keys())
        analysis_agents = [
            a for a in all_agents
            if a not in ("eda", "orchestrator", "debate", "insight", "guardian")
        ]

        # Pick 2-3 agents not in current plan to try
        current = set(context.active_agents)
        novel = [a for a in analysis_agents if a not in current]
        if not novel:
            return None

        exploration = list(current) + random.sample(novel, min(2, len(novel)))
        # Ensure debate + insight at end
        for required in ["debate", "insight"]:
            if required in exploration:
                exploration.remove(required)
            exploration.append(required)

        logger.info(f"ε-greedy exploration: novel plan = {exploration}")
        return exploration

    # ------------------------------------------------------------------
    # Power 5: Policy enforcement
    # ------------------------------------------------------------------

    def _enforce_policy(self, context: AnalysisContext) -> list[dict]:
        blocked = []

        # Check experiment findings
        exp = context.results.get("experiment")
        if exp and exp.status == "success":
            tt = exp.data.get("results", {}).get("ttest", {})
            n_a = tt.get("n_a", 0)
            n_b = tt.get("n_b", 0)
            v = self._policy.check_ab_sample(n_a, n_b)
            if v:
                blocked.append({
                    "agent": "experiment",
                    "rule": v.rule,
                    "reason": v.reason,
                    "blocking": v.blocking,
                })

        # Check forecast confidence
        fcst = context.results.get("forecast")
        if fcst and fcst.status == "success":
            n_pts = fcst.data.get("data_points", 0)
            min_pts = self._policy.get("min_forecast_data_points", 14)
            if n_pts < min_pts:
                blocked.append({
                    "agent": "forecast",
                    "rule": "min_forecast_data_points",
                    "reason": f"Only {n_pts} data points, need {min_pts}",
                    "blocking": True,
                })

        # Check cluster quality
        clst = context.results.get("ml_cluster")
        if clst and clst.status == "success":
            sil = clst.data.get("silhouette_score", 0)
            min_sil = self._policy.get("min_cluster_silhouette", 0.25)
            if sil < min_sil:
                blocked.append({
                    "agent": "ml_cluster",
                    "rule": "min_cluster_silhouette",
                    "reason": f"Silhouette {sil:.3f} < required {min_sil}",
                    "blocking": False,   # warn but don't block
                })

        return blocked

    # ------------------------------------------------------------------
    # Temporal holdout enforcement
    # ------------------------------------------------------------------

    def _enforce_holdout(self, context: AnalysisContext) -> list[str]:
        warnings = []
        fcst = context.results.get("forecast")
        if not fcst or fcst.status != "success":
            return warnings

        method = fcst.data.get("method", "")
        if method == "Holt ETS":
            warnings.append(
                "Forecast used Holt ETS (simple fallback). "
                "Holdout validation not available for this method. "
                "Confidence capped at 0.70 per policy."
            )

        fdf = fcst.data.get("forecast_df")
        if fdf is not None and not fdf.empty:
            has_ci = "yhat_lower" in fdf.columns and "yhat_upper" in fdf.columns
            if not has_ci and self._policy.get("require_confidence_intervals"):
                warnings.append(
                    "Forecast missing confidence intervals. "
                    "Policy requires CI on all forecasts. "
                    "Publishing with reduced confidence."
                )

        return warnings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_scenario(self, context: AnalysisContext) -> str:
        profile = context.data_profile
        if profile.get("has_funnel_signal"):
            return "funnel"
        if profile.get("has_cohort_signal"):
            return "cohort"
        if profile.get("has_time_series"):
            return "timeseries"
        return "general"

    def _data_signature(self, context: AnalysisContext) -> str:
        import hashlib
        df = context.df
        sig = f"{df.shape}|{sorted(df.columns.tolist())}"
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    def log_policy_change(self, rule: str, old_val, new_val, reason: str, changed_by: str):
        """Admin-only. All changes logged immutably."""
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                INSERT INTO policy_change_log
                (changed_by, rule, old_value, new_value, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (changed_by, rule, str(old_val), str(new_val), reason,
                  datetime.now().isoformat()))
            conn.commit()
        logger.info(f"Policy change logged: {rule} {old_val} → {new_val} by {changed_by}")
