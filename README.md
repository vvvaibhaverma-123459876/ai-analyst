# AI Analyst v7.0

AI Analyst v7.0 is the first build in this repository explicitly hardened toward the full Everything Prompt standard: governed metrics, evidence-scored hypotheses, supervisor-style governance, data-quality gating, human-approval boundaries, reproducible manifests, and ranked recommendations.

## What changed in v7.0

- **Truth spine strengthened**
  - semantic layer remains primary for metric, grain, and join governance
  - runtime manifest records semantic/policy versions per run

- **Trust spine strengthened**
  - new `quality/DataQualityGate` blocks clearly invalid analyses and lowers confidence for weak inputs
  - new approval boundary via `governance/ApprovalGate`
  - guardian now emits evidence grades and extracts run lessons
  - security shell now redacts prompt text and classifies outputs

- **Operational rigor added**
  - reproducible per-run manifest persisted under `memory/manifests/`
  - replay harness can reload manifests
  - ranked recommendations now include confidence, urgency, value, and effort

## v7.0 runtime additions

1. `runner` runs EDA, then a data-quality gate before orchestration.
2. `orchestrator` can suppress high-risk advanced agents under poor data quality.
3. `conclusion_engine` now folds in evidence grading and data-quality penalties.
4. `guardian` now returns evidence-grade and lesson metadata.
5. `insight_agent` returns ranked recommendations.
6. `runner` persists run manifests and extracted lessons.

## New folders

- `quality/`
- `governance/`
- `versioning/`
- `evaluation/`

## Important note

This build is materially closer to the Everything Prompt, but real 1.0 trust still depends on broader integration coverage, richer benchmark scenarios, and deeper policy enforcement across every endpoint. v7.0 focuses on the most important runtime gaps: data quality gating, human approval boundaries, evidence discipline, reproducibility, and decision-ranked outputs.
