"""
core/exceptions.py
Custom exceptions used across the project.
"""


class AIAnalystError(Exception):
    """Base exception."""


class ConnectorError(AIAnalystError):
    """Raised when a data connector fails."""


class SQLGenerationError(AIAnalystError):
    """Raised when SQL cannot be generated or is invalid."""


class SQLValidationError(AIAnalystError):
    """Raised when generated SQL fails safety/format checks."""


class LLMError(AIAnalystError):
    """Raised when LLM API call fails."""


class MetadataError(AIAnalystError):
    """Raised when required metadata (metric, table) is not found."""


class AnalysisError(AIAnalystError):
    """Raised when an analysis step fails."""


class DataQualityError(AIAnalystError):
    """Raised when the DataQualityGate finds a fatal (not just low-score)
    setup problem — e.g. the chosen date column isn't actually a date — that
    would make every downstream result meaningless if the pipeline proceeded
    anyway. Distinct from the gate's regular blocking_reasons, which lower
    confidence but don't halt the run."""
