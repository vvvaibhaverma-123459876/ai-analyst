# Production Readiness Plan

This repo is now closer to a production-style analytics platform, but the path to true production should be treated as layered hardening, not feature sprawl.

## What was hardened in this pass

- Added a modern command-center Streamlit UI at `app/ui/product_app.py`.
- Kept backwards compatibility through `app/ui/v06_app.py`.
- Added a realistic fintech onboarding demo dataset.
- Updated FastAPI job runner to use the universal ingestion engine instead of CSV-only parsing.
- Added API result endpoint: `GET /jobs/{job_id}/result`.
- Improved health endpoint capability disclosure.
- Added production-oriented Docker, Docker Compose, Makefile, and Streamlit config.
- Added UI/product strategy documentation.

## Production layers still needed before real enterprise deployment

### 1. Authentication and identity

Current auth is demo-friendly. Production should add:

- Real user store or SSO/OAuth.
- Strong password handling if password auth remains.
- Role and tenant membership stored in DB, not environment variables.
- Token revocation/session expiry controls.

### 2. Durable jobs

Current jobs are backed by SQLite and in-process background tasks. Production should add:

- Postgres job store.
- Object storage for uploads.
- Worker queue such as Celery/RQ/Arq.
- Job cancellation and retry policy.
- Progress events over WebSocket/SSE.

### 3. Data security

Before uploading real business data:

- Enforce tenant isolation in every API route.
- Add encryption at rest for uploaded files and job outputs.
- Add audit log export per tenant.
- Add configurable retention/deletion policies.
- Add secret scanning and credential redaction in logs.

### 4. Connector hardening

For Athena/Postgres/Snowflake/BigQuery:

- Add connection testing endpoint.
- Add schema browser endpoint.
- Add query preview limits.
- Add timeout and cancellation.
- Add SQL dry-run and cost guardrails.

### 5. Observability

- Structured JSON logs.
- Request IDs and run IDs across API/UI/pipeline.
- Metrics for job latency, failure rate, agent duration, and output classification.
- Error tracking.

### 6. CI/CD

- Keep full tests passing.
- Add API smoke tests.
- Add UI smoke/import test.
- Add Docker build test.
- Add release artifacts.

## Recommended deployment shape

```text
Browser
  ↓
Streamlit or React frontend
  ↓
FastAPI backend
  ↓
Worker queue
  ↓
GovernedPipeline
  ↓
Postgres metadata + object storage + audit log
```

## How to run locally

```bash
make setup-ci
make test
make run-ui
```

or with Docker:

```bash
docker compose up --build
```

Then open the UI on port `8501` and the API on port `8000`.
