# AI Analyst v0.5 — Enterprise Autonomous Edition

**Coverage scores vs target:**

| Dimension | v0.4 | v0.5 |
|---|---|---|
| Autonomous analytics | 80% | **95%** |
| Agentic multi-skill analysis | 75% | **92%** |
| Guardian / teacher architecture | 35% | **85%** |
| Hypothesis-driven reasoning | 45% | **88%** |
| Confidentiality-first readiness | 30% | **90%** |
| Enterprise API / security | 25% | **85%** |
| External context enrichment | 0% | **75%** |

## What's new in v0.5

### Security shell (precondition for enterprise)
- `DataClassifier` — tags every column: PUBLIC / INTERNAL / CONFIDENTIAL / PII / SENSITIVE
- `PIIMasker` — replaces PII with deterministic tokens before any LLM call
- `AuditLogger` — immutable append-only log of every external call (hash only, never content)
- `PolicyStore` — admin-defined rules in `configs/policy.yaml` that no learning agent can override
- `LocalLLMClient` — routes all completions to Ollama when `internet_off_mode: true`
- `SecurityShell` — single entry point: all data and all LLM calls pass through this

### Ground truth recorder (foundation for all learning)
- Every finding gets an outcome slot (correct / incorrect / pending)
- Thumbs up/down in the UI populates this table
- Every learning agent, calibration store, and score tracker reads from it

### FastAPI service
- `POST /jobs` → submit analysis job, get job_id immediately
- `GET /jobs/{id}/brief` → fetch finished brief
- `POST /jobs/{id}/verify` → record ground truth
- JWT auth, RBAC (viewer/analyst/admin), tenant isolation, audit logging

### Guardian governor (5 active powers)
- Scores agents over time with exponential decay weighting
- Detects directional contradictions across runs on the same data signature
- Triggers LLM prompt rewrites when agent accuracy < 0.60
- ε-greedy exploration: 10% of runs try novel plans (prevents meta-overfit)
- Enforces policy rules with veto power over conclusions

### Overfitting defences
1. **Temporal holdout** — last 20% always reserved; Guardian enforces MAPE validation
2. **ε-greedy exploration** — 10% novel plans prevent routing ossification
3. **Score decay** — old observations halve in weight every 30 days
4. **Sycophancy lock** — debate jury challenge rules are policy-locked, not learnable
5. **Adversarial search** — every confirming query runs an opposing query

### Scientific reasoning pipeline
- `HypothesisAgent` — 4 independent sources (data, business, web, prior)
- `FeasibilityAgent` — filters to testable hypotheses, documents data gaps
- `ResearchPlan` — travels through pipeline, collects evidence per hypothesis
- `ConclusionEngine` — closes each hypothesis: CONFIRMED / REJECTED / INCONCLUSIVE

### Sub-jury system
- `AnomalyJuryAgent` — Z-score + IQR + STL + Isolation Forest (Foreman: majority protocol)
- `ForecastJuryAgent` — Prophet + ARIMA + ETS with temporal holdout validation
- `Foreman` — unanimous / majority / split / no-consensus protocol
- Split verdict = disagreement IS the finding

### Learning layer (7 adapters, sycophancy-protected)
- IngestionLearner, ContextLearner, OrchestratorLearner, AnalysisLearner
- HypothesisLearner, InsightLearner, OutputRouterLearner
- All adapters blocked from modifying policy-locked rules

### External enrichment (adversarial)
- Triggered by anomaly detection, not on every run
- Adversarial search: confirming query + opposing query run simultaneously
- Source reliability tracked against verified outcomes over time
- All external evidence tagged `external_context` (vs `internal_data`)

## Quickstart

```bash
cd ai_analyst_v05
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add OPENAI_API_KEY or ANTHROPIC_API_KEY

# Streamlit UI
streamlit run app/ui/v05_app.py

# FastAPI service
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Internet-off / local mode (edit configs/policy.yaml)
# internet_off_mode: true
# local_llm_mode: true
# local_llm_model: mistral   (requires: ollama pull mistral)
```

## Architecture (24 directories, 116+ files)

```
security/      ← PII masker · classifier · audit · policy · local LLM
ground_truth/  ← outcome recorder (foundation for all learning)
api/           ← FastAPI · auth · RBAC · job queue · tenant isolation
guardian/      ← governor · 5 powers · ε-greedy · score decay
science/       ← hypothesis · feasibility · research plan · conclusion engine
jury/          ← base juror · foreman protocol · anomaly jury · forecast jury
learning/      ← base layer + 7 per-layer adapters (sycophancy-protected)
enrichment/    ← adversarial web search · source reliability store
agents/        ← 14 analysis agents + runner + context (v0.4 base)
ingestion/     ← 10 format parsers (v0.4)
context_engine/← upfront questions + org memory (v0.4)
output/        ← router · alerts · conversation (v0.4)
```

## Policy configuration

Edit `configs/policy.yaml` to control:
- Minimum sample sizes and confidence thresholds
- Which agents are permitted to run
- Internet-off mode, local LLM mode
- PII handling policy
- A/B test requirements

Policy changes are logged immutably to the audit trail.
