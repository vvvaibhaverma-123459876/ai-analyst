"""
llm/prompts.py
All prompt templates in one place.
Keeps LLM logic out of business logic files.
"""


class Prompts:

    # ------------------------------------------------------------------
    # SQL Generation
    # ------------------------------------------------------------------

    SQL_SYSTEM = """You are an expert SQL analyst. 
You write safe, read-only SELECT queries only.
Never use DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, CREATE.
Always include a LIMIT clause unless the user explicitly asks for all rows.
Return ONLY the SQL query with no explanation, no markdown fences, no preamble."""

    @staticmethod
    def sql_user(intent: dict, schema_context: str, metric_context: str) -> str:
        return f"""Write a SQL query for the following request.

USER QUESTION: {intent['raw_question']}

IDENTIFIED INTENT:
- Metric: {intent.get('metric', 'not specified')}
- Dimensions: {intent.get('dimensions', [])}
- Filters: {intent.get('filters', {})}
- Time range: {intent.get('time_range', 'not specified')}
- Analysis type: {intent.get('analysis_type', 'trend')}

SCHEMA:
{schema_context}

METRIC DEFINITIONS:
{metric_context}

Rules:
- Use only tables and columns listed in SCHEMA above.
- Apply filters from IDENTIFIED INTENT.
- Always filter by the time_range if provided.
- Add LIMIT 10000.
- Output SQL only."""

    # ------------------------------------------------------------------
    # Executive Summary
    # ------------------------------------------------------------------

    EXEC_SUMMARY_SYSTEM = """You are a senior analytics lead writing for business leadership.
Be concise, evidence-based, and action-oriented.
Use only the facts provided. Do not invent data points."""

    @staticmethod
    def exec_summary_user(payload: dict) -> str:
        return f"""Write an executive summary with these four sections:

1) What happened — state the KPI change with numbers
2) Why — evidence-based explanation from the driver data
3) Impact — business significance
4) Recommended actions — specific, actionable next steps

Facts provided:
{payload}

Format each section with a bold heading."""

    # ------------------------------------------------------------------
    # Insight Narrative
    # ------------------------------------------------------------------

    INSIGHT_SYSTEM = """You are a senior data analyst. 
Explain findings clearly to a business audience.
Use numbers from the facts. Be specific. Max 3 sentences per point."""

    @staticmethod
    def insight_user(analysis_output: dict) -> str:
        return f"""Write a brief analyst narrative (3-5 sentences) summarising these findings:

{analysis_output}

Lead with the most important finding. Mention magnitude. End with one hypothesis."""

    # ------------------------------------------------------------------
    # Follow-up Questions
    # ------------------------------------------------------------------

    FOLLOWUP_SYSTEM = """You are an analytics lead suggesting the next best questions 
to ask after an initial analysis. Questions must be specific, data-driven, and answerable."""

    @staticmethod
    def followup_user(question: str, analysis_summary: str) -> str:
        return f"""Original question: {question}

Analysis summary: {analysis_summary}

Suggest exactly 4 follow-up questions an analyst should investigate next.
Each question must be specific (mention metric, segment, or time period).
Return as a numbered list. No explanations."""

    # ------------------------------------------------------------------
    # Root Cause Narrative
    # ------------------------------------------------------------------

    ROOT_CAUSE_SYSTEM = """You are a product analytics expert specialising in root cause analysis.
Explain what likely caused a KPI change based on evidence only.
Be specific about segments, magnitudes, and timing."""

    @staticmethod
    def root_cause_user(kpi: str, delta: float, pct: float, drivers: list, anomalies: list) -> str:
        return f"""KPI: {kpi}
Overall change: {delta:+,.2f} ({pct:+.1f}%)

Top contributing segments:
{drivers}

Anomaly dates detected:
{anomalies}

Write a 3-5 sentence root cause analysis.
State which segment drove the most change, its magnitude, and a likely reason."""
