"""
core/constants.py
Project-wide constants.
"""

TIME_GRAINS = ["Daily", "Weekly", "Monthly"]

RESAMPLE_MAP = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "ME",
}

ANOMALY_METHODS = ["zscore", "iqr", "stl"]

ANALYSIS_TYPES = [
    "trend",
    "anomaly",
    "driver_attribution",
    "root_cause",
    "contribution",
    "funnel",
    "cohort",
]

LLM_PROVIDERS = ["openai", "anthropic"]

SAFE_SQL_KEYWORDS = ["SELECT", "WITH", "EXPLAIN"]
UNSAFE_SQL_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
                       "TRUNCATE", "CREATE", "REPLACE", "MERGE", "EXEC"]
