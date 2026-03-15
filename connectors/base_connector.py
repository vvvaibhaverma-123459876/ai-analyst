"""
connectors/base_connector.py
Abstract base class all connectors must implement.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseConnector(ABC):

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source."""

    @abstractmethod
    def execute(self, query: str) -> pd.DataFrame:
        """Execute a query/operation and return a DataFrame."""

    @abstractmethod
    def get_schema(self) -> dict:
        """Return schema metadata: {table_name: [col_name, ...]}"""

    @abstractmethod
    def test_connection(self) -> bool:
        """Return True if connection is healthy."""
