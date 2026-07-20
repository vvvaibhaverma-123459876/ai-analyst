# Deployment

## Primary path: Streamlit Community Cloud

Streamlit Cloud auto-redeploys `main` natively — there is no GitHub Actions
deploy step. CI (`.github/workflows/ci.yml`) gates every push/PR: the `test`
job runs the full pytest suite, and the `demo-smoke` job installs
`requirements-demo.txt` fresh and does a headless import of the product UI to
catch missing-dependency breakage before it ever reaches the live app.
Streamlit Cloud itself watches `main` independently and redeploys on every
push that lands there.

### One-time setup (click-path)

1. [share.streamlit.io](https://share.streamlit.io) → **New app** → pick this
   GitHub repo and the `main` branch.
2. **Main file path**: `app/ui/product_app.py`
3. **Advanced settings → Python dependencies**: point it at
   `requirements-demo.txt` (not `requirements.txt` — that pulls in
   prophet/chromadb/sentence-transformers/snowflake/bigquery, which would blow
   the free tier's install time and memory budget for no benefit in offline
   mode).
4. **Advanced settings → Secrets**: add

   ```toml
   ANALYST_MODE = "offline"
   ```

   Nothing else is required — offline mode never reads `OPENAI_API_KEY` /
   `ANTHROPIC_API_KEY` even if you accidentally leave one in secrets (see
   `core/config.py`). Leaving `ANALYST_MODE` unset also works, since
   `"offline"` is the default — setting it explicitly here is just belt and
   suspenders against Streamlit Cloud's own default-secrets behavior.
5. Deploy. Once your app has a URL, set the GitHub repo variable `DEPLOY_URL`
   to it (Settings → Secrets and variables → Actions → Variables) — the daily
   health-check workflow needs it.
6. Theming is already configured in `.streamlit/config.toml` — no changes
   needed there.

### Verifying the deploy

- App health: `<your-app-url>/_stcore/health` should return `200`.
- The hero section's mode chip should read "Offline demo mode — no LLM calls,
  no API keys required".
- `.github/workflows/health-check.yml` pings this daily and opens a GitHub
  issue titled "🔴 Live demo is down" if it fails, auto-closing it once the
  check passes again.

## Also supported: Docker (container hosts)

The `Dockerfile` builds a self-contained image (`requirements-demo.txt`,
`ANALYST_MODE=offline` by default, binds to `$PORT`) for Render, Railway, Fly,
or any generic container host, if you'd rather not use Streamlit Cloud:

```bash
docker build -t ai-analyst .
docker run -p 8501:8501 -e PORT=8501 ai-analyst
```

## Running full mode (LLM-backed) locally

Full mode needs real LLM credentials and is **not** what the public demo runs
— see the README's "Capabilities: offline demo vs. full mode" table and the
"Run with real LLM access" section for the local setup (`ANALYST_MODE=full`,
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, `pip install -r requirements.txt`).
