## What changed and why

<!-- Summarize the change and the motivation. -->

## Checklist

- [ ] `pip install -r requirements-ci.txt && pytest -q` passes
- [ ] If UI-facing: `pip install -r requirements-demo.txt && streamlit run app/ui/product_app.py` runs, clicked through connect → analyze → deep-dive
- [ ] If touching an LLM-adjacent module: confirmed no outbound call happens when `ANALYST_MODE=offline` (see `tests/test_offline_mode.py`)
- [ ] No new required env var without updating `.env.example` / the README

**Reminder:** Streamlit Community Cloud auto-redeploys on every push to
`main`. CI must be green before merging — see
[CONTRIBUTING.md](../CONTRIBUTING.md).
