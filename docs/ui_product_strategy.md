# AI Analyst Product UI Strategy

## Product principle

The UI should not feel like a notebook or a collection of scripts. It should feel like a modern analytics command center that turns raw business data into an explainable decision.

The product story is:

> Connect data → validate schema → run governed agents → understand what changed → identify why → decide what to do → export/share.

## How the UI represents the true capability

| Hidden platform capability | User-facing representation |
|---|---|
| Ingestion engine | Upload card supporting CSV, Excel, JSON, and demo datasets |
| Data quality gate | Schema validation, missing values, row/column preview |
| Semantic layer | KPI/date/grain controls before analysis |
| Governed pipeline | Live agent console showing EDA, hypothesis, trend, anomaly, root cause, forecast, guardian, insight |
| Security shell | Tenant/user controls, PII masking, governed output boundary |
| Anomaly jury | KPI trend panel with anomaly count and chart markers |
| Root-cause engine | Driver chart and ranked segment table |
| Funnel/cohort agents | Dedicated deep-dive panels when the dataset supports them |
| Forecast jury | Forecast summary and future values table |
| Guardian/debate | Trust review and agent reliability panel |
| Conversation engine | Follow-up chat grounded in the completed run |
| Audit/replay/export | Markdown/JSON report downloads and run manifest payload |

## Final app layout

1. **Hero / positioning** — instantly says this is a governed AI analyst, not a charting toy.
2. **Connect data** — demo or upload file.
3. **Validate setup** — choose date column, KPI column, time grain, business goal.
4. **Run governed analysis** — progress + live agent cards.
5. **Command center** — KPI cards, executive brief, recommended next questions.
6. **Deep dive** — trend/anomaly, period comparison, root cause, funnel, cohort, forecast.
7. **Agent console** — raw agent status and summaries for technical confidence.
8. **Ask follow-up** — conversational analysis grounded in the run.
9. **Export** — Markdown report and JSON payload.

## Design choices

- Dark glassmorphism style to feel app-like and modern.
- Strong visual hierarchy: executive decision first, diagnostics second, raw internals third.
- Capabilities are visible but not overwhelming.
- App remains usable without external LLM/API keys.
- Built around Streamlit for fast iteration, but shaped like a product shell.

## Next UI upgrades

- Add authenticated user landing page.
- Add saved analysis history sidebar.
- Add connector-specific setup flows for Athena/Postgres/Snowflake.
- Add PDF export with embedded charts.
- Add a React frontend once backend contracts stabilize.
