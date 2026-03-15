"""
connectors/bigquery_connector.py
Google BigQuery connector using google-cloud-bigquery.

Env vars:
  GOOGLE_APPLICATION_CREDENTIALS  path to service-account JSON
  BQ_PROJECT_ID
  BQ_DATASET_ID     (optional default dataset)
  BQ_LOCATION       (default: US)
"""

from __future__ import annotations
import os
import pandas as pd
from .base_connector import BaseConnector
from core.exceptions import ConnectorError
from core.logger import get_logger

logger = get_logger(__name__)


class BigQueryConnector(BaseConnector):

    def __init__(
        self,
        project_id: str  = None,
        dataset_id: str  = None,
        location: str    = None,
        credentials_path: str = None,
    ):
        self._project  = project_id or os.getenv("BQ_PROJECT_ID", "")
        self._dataset  = dataset_id or os.getenv("BQ_DATASET_ID", "")
        self._location = location   or os.getenv("BQ_LOCATION", "US")
        self._creds    = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        self._client   = None

    def connect(self) -> None:
        if not self._project:
            raise ConnectorError("BQ_PROJECT_ID must be set.")
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account

            if self._creds and os.path.exists(self._creds):
                creds = service_account.Credentials.from_service_account_file(
                    self._creds,
                    scopes=["https://www.googleapis.com/auth/bigquery"],
                )
                self._client = bigquery.Client(project=self._project, credentials=creds)
            else:
                # Use ADC (Application Default Credentials)
                self._client = bigquery.Client(project=self._project)

            # Smoke test
            self._client.query("SELECT 1").result()
            logger.info("BigQuery connected: project=%s dataset=%s", self._project, self._dataset)

        except ImportError:
            raise ConnectorError(
                "google-cloud-bigquery not installed. "
                "Run: pip install google-cloud-bigquery db-dtypes"
            )
        except Exception as e:
            raise ConnectorError(f"BigQuery connection failed: {e}") from e

    def execute(self, query: str) -> pd.DataFrame:
        if self._client is None:
            raise ConnectorError("Not connected. Call connect() first.")
        try:
            return self._client.query(query).to_dataframe()
        except Exception as e:
            raise ConnectorError(f"BigQuery query failed: {e}") from e

    def get_schema(self) -> dict:
        if self._client is None:
            raise ConnectorError("Not connected.")
        if not self._dataset:
            return {}
        dataset_ref = self._client.dataset(self._dataset)
        tables = list(self._client.list_tables(dataset_ref))
        schema: dict = {}
        for table in tables:
            t_ref = self._client.get_table(
                f"{self._project}.{self._dataset}.{table.table_id}"
            )
            schema[table.table_id] = [
                {"name": f.name, "type": f.field_type, "mode": f.mode}
                for f in t_ref.schema
            ]
        return schema

    def test_connection(self) -> bool:
        try:
            if self._client is None:
                return False
            self._client.query("SELECT 1").result()
            return True
        except Exception:
            return False

    def list_tables(self) -> list[str]:
        if not self._dataset:
            return []
        dataset_ref = self._client.dataset(self._dataset)
        return [t.table_id for t in self._client.list_tables(dataset_ref)]

    def sample(self, table: str, n: int = 5) -> pd.DataFrame:
        return self.execute(
            f"SELECT * FROM `{self._project}.{self._dataset}.{table}` LIMIT {n}"
        )
