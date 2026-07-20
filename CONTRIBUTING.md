# Contributing

`main` auto-deploys to Streamlit Community Cloud (native GitHub integration —
see [docs/deployment.md](docs/deployment.md)). Treat `main` as production.

## Workflow

1. Branch off `main`.
2. Make your change, with tests where it makes sense.
3. Open a PR. `.github/workflows/ci.yml` runs the full pytest suite (`test`
   job) and a slim-dependency headless import + offline-mode test
   (`demo-smoke` job).
4. **CI must be green before merge.** Merging to `main` triggers a live
   Streamlit Cloud redeploy, so a red PR should never be merged.
5. `.github/workflows/health-check.yml` pings the live app's health endpoint
   daily and opens a GitHub issue if the deploy is down.

## Local checks before opening a PR

```bash
pip install -r requirements-ci.txt
python -m compileall .
pytest -q

# if UI-facing:
pip install -r requirements-demo.txt
streamlit run app/ui/product_app.py
```

## Scope

The public demo runs `ANALYST_MODE=offline` and must never require an API key
or make an outbound network call — `tests/test_offline_mode.py` enforces this
by patching `socket.socket.connect` to raise. If you add a new LLM-touching
module, guard it the same way the existing ones do: check
`config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY` before calling, and
degrade to a deterministic path when falsy — don't rely on the network call
itself failing gracefully.
