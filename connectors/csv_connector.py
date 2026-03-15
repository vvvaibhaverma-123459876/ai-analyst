"""
connectors/csv_connector.py
CSV file connector. Supports both file path and Streamlit UploadedFile.
Extracted and extended from app.py v0.1.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, IO

from .base_connector import BaseConnector
from core.exceptions import ConnectorError
from core.logger import get_logger

logger = get_logger(__name__)


class CSVConnector(BaseConnector):

    def __init__(self, source: Union[str, Path, IO] = None):
        self._source = source
        self._df: pd.DataFrame = None

    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._source is None:
            raise ConnectorError("No CSV source provided.")
        try:
            self._df = pd.read_csv(self._source)
            logger.info(f"CSV loaded: {self._df.shape[0]} rows × {self._df.shape[1]} cols")
        except Exception as e:
            raise ConnectorError(f"Failed to read CSV: {e}") from e

    def execute(self, query: str) -> pd.DataFrame:
        """
        For CSV connector, 'query' is a pandas query expression string.
        e.g. "channel == 'organic' and signups > 100"
        Pass empty string to return the full DataFrame.
        """
        if self._df is None:
            raise ConnectorError("Not connected. Call connect() first.")
        if not query.strip():
            return self._df.copy()
        try:
            return self._df.query(query).copy()
        except Exception as e:
            raise ConnectorError(f"Query failed: {e}") from e

    def get_schema(self) -> dict:
        if self._df is None:
            raise ConnectorError("Not connected. Call connect() first.")
        return {
            "csv": [
                {
                    "column": col,
                    "dtype": str(self._df[col].dtype),
                    "sample": self._df[col].dropna().head(3).tolist(),
                }
                for col in self._df.columns
            ]
        }

    def test_connection(self) -> bool:
        return self._df is not None

    # ------------------------------------------------------------------
    # Extra helpers
    # ------------------------------------------------------------------

    def get_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            raise ConnectorError("Not connected. Call connect() first.")
        return self._df.copy()

    def load_from_uploaded_file(self, uploaded_file) -> pd.DataFrame:
        """Convenience for Streamlit st.file_uploader objects."""
        try:
            self._df = pd.read_csv(uploaded_file)
            logger.info(f"Uploaded CSV loaded: {self._df.shape}")
            return self._df.copy()
        except Exception as e:
            raise ConnectorError(f"Failed to read uploaded file: {e}") from e

    # ------------------------------------------------------------------
    # Static utility: detect best datetime column (preserved from app.py)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_datetime_column(df: pd.DataFrame) -> str | None:
        # First pass: already datetime dtype
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                return c
        # Second pass: object columns that parse well
        best_col = None
        best_valid = 0
        for c in df.columns:
            if df[c].dtype == "object":
                parsed = pd.to_datetime(df[c], errors="coerce")
                valid = parsed.notna().sum()
                if valid > best_valid and valid >= int(0.6 * len(df)):
                    best_valid = valid
                    best_col = c
        return best_col
