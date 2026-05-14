# AI Analyst Gap-Closure Implementation Notes

This branch/package focuses on hardening the existing AI Analyst platform instead of adding another feature layer. The goal is to make the current architecture more reliable, testable, demo-ready, and safer to run locally.

## What was fixed

- Stabilized anomaly detection with robust rolling baselines, MAD-based scoring, direction/severity labels, and deterministic records.
- Fixed date/phone PII misclassification and made security classification more conservative for ordinary business metrics.
- Fixed output classification for redacted sensitive tokens and clean internal text.
- Prevented web enrichment from slowing or destabilizing offline runs; it is now opt-in via `AI_ANALYST_ENABLE_WEB_ENRICHMENT=1`.
- Fixed root-cause edge cases around empty data, invalid dates, and dimension attribution output.
- Fixed cohort analyzer edge cases around empty data and invalid dates.
- Cleaned the SQL validator contract: `validate()` returns issues, `sanitize()` returns safe SQL or raises.
- Fixed retry engine to use SQL sanitization explicitly.
- Removed duplicate `SQLGenerator` class so the deterministic multi-hop join implementation is the active implementation.
- Fixed the Streamlit follow-up chat wiring to instantiate `ConversationEngine(ctx)` and call `.chat()`.
- Fixed the broken legacy UI syntax error so repository compile checks pass.
- Tuned recommendation ranking so low-confidence urgent recommendations do not outrank safer high-confidence recommendations.
- Made heavy jury models opt-in via `AI_ANALYST_FULL_JURY=1` so default tests and local demos remain fast.
- Added CI workflow for compile and test checks.
- Removed dependency on pytest-timeout from the default pytest command.

## Validation performed

The full test suite was run after the main hardening pass and passed:

```bash
pytest -q -o addopts=''
# 353 passed
```

After the final UI/SQL/CI cleanup, run:

```bash
python -m compileall .
pytest -q
```

## Runtime flags

Optional heavier capabilities are intentionally gated:

```bash
AI_ANALYST_FULL_JURY=1              # enables heavier STL/IsolationForest/Prophet/ARIMA jurors
AI_ANALYST_ENABLE_WEB_ENRICHMENT=1  # enables external web/context enrichment
```

## Recommended commands

```bash
# UI
streamlit run app/ui/v06_app.py

# API
uvicorn api.main:app --reload

# Tests
pytest -q
```

## Honest status

This package significantly improves the reliability and demo readiness of the existing project. It does not claim that every enterprise roadmap item is fully complete. The next layer of work should focus on production deployment hardening, real connector credential validation, full multi-user auth, deeper UI polish, and richer domain templates.
