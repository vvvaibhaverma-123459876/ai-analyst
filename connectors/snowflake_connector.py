"""
connectors/snowflake_connector.py
Snowflake connector using snowflake-sqlalchemy.

Env vars:
  SNOWFLAKE_ACCOUNT   e.g. xy12345.us-east-1
  SNOWFLAKE_USER
  SNOWFLAKE_PASSWORD
  SNOWFLAKE_DATABASE
  SNOWFLAKE_SCHEMA      (default: PUBLIC)
  SNOWFLAKE_WAREHOUSE   (default: COMPUTE_WH)
  SNOWFLAKE_ROLE        (optional)
"""

from __future__ import annotations
import os
import pandas as pd
from .base_connector import BaseConnector
from core.exceptions import ConnectorError
from core.logger import get_logger

logger = get_logger(__name__)


class SnowflakeConnector(BaseConnector):

    def __init__(
        self,
        account: str   = None,
        user: str      = None,
        password: str  = None,
        database: str  = None,
        schema: str    = None,
        warehouse: str = None,
        role: str      = None,
    ):
        self._account   = account   or os.getenv("SNOWFLAKE_ACCOUNT", "")
        self._user      = user      or os.getenv("SNOWFLAKE_USER", "")
        self._password  = password  or os.getenv("SNOWFLAKE_PASSWORD", "")
        self._database  = database  or os.getenv("SNOWFLAKE_DATABASE", "")
        self._schema    = schema    or os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
        self._warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        self._role      = role      or os.getenv("SNOWFLAKE_ROLE", "")
        self._engine    = None

    def connect(self) -> None:
        if not self._account or not self._user:
            raise ConnectorError("SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER must be set.")
        try:
            from snowflake.sqlalchemy import URL
            from sqlalchemy import create_engine, text
            url_kwargs = dict(
                account=self._account,
                user=self._user,
                password=self._password,
                database=self._database,
                schema=self._schema,
                warehouse=self._warehouse,
            )
            if self._role:
                url_kwargs["role"] = self._role
            self._engine = create_engine(URL(**url_kwargs))
            with self._engine.connect() as conn:
                conn.execute(text("SELECT CURRENT_VERSION()"))
            logger.info("Snowflake connected: db=%s schema=%s", self._database, self._schema)
        except ImportError:
            raise ConnectorError(
                "snowflake-sqlalchemy not installed. "
                "Run: pip install snowflake-sqlalchemy snowflake-connector-python"
            )
        except Exception as e:
            raise ConnectorError(f"Snowflake connection failed: {e}") from e

    def execute(self, query: str) -> pd.DataFrame:
        if self._engine is None:
            raise ConnectorError("Not connected. Call connect() first.")
        try:
            return pd.read_sql(query, self._engine)
        except Exception as e:
            raise ConnectorError(f"Snowflake query failed: {e}") from e

    def get_schema(self) -> dict:
        if self._engine is None:
            raise ConnectorError("Not connected.")
        q = f"""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM {self._database}.INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{self._schema}'
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        df = self.execute(q)
        schema: dict = {}
        for _, row in df.iterrows():
            t = row["TABLE_NAME"]
            schema.setdefault(t, []).append({
                "name": row["COLUMN_NAME"],
                "type": row["DATA_TYPE"],
                "nullable": row["IS_NULLABLE"] == "YES",
            })
        return schema

    def test_connection(self) -> bool:
        try:
            from sqlalchemy import text
            if self._engine is None:
                return False
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def list_tables(self) -> list[str]:
        df = self.execute(
            f"SELECT TABLE_NAME FROM {self._database}.INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA='{self._schema}' ORDER BY TABLE_NAME"
        )
        return df["TABLE_NAME"].tolist()
