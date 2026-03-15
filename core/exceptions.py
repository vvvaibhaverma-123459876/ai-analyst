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
