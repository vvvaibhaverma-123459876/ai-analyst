"""
connectors/redshift_connector.py
Amazon Redshift connector using redshift_connector or SQLAlchemy+psycopg2.

Env vars:
  REDSHIFT_HOST
  REDSHIFT_PORT     (default: 5439)
  REDSHIFT_DB
  REDSHIFT_USER
  REDSHIFT_PASSWORD
  REDSHIFT_SCHEMA   (default: public)
  REDSHIFT_IAM_PROFILE  (optional, for IAM auth)
"""

from __future__ import annotations
import os
import pandas as pd
from .base_connector import BaseConnector
from core.exceptions import ConnectorError
from core.logger import get_logger

logger = get_logger(__name__)


class RedshiftConnector(BaseConnector):

    def __init__(
        self,
        host: str     = None,
        port: int     = None,
        database: str = None,
        user: str     = None,
        password: str = None,
        schema: str   = None,
    ):
        self._host     = host     or os.getenv("REDSHIFT_HOST", "")
        self._port     = port     or int(os.getenv("REDSHIFT_PORT", "5439"))
        self._database = database or os.getenv("REDSHIFT_DB", "")
        self._user     = user     or os.getenv("REDSHIFT_USER", "")
        self._password = password or os.getenv("REDSHIFT_PASSWORD", "")
        self._schema   = schema   or os.getenv("REDSHIFT_SCHEMA", "public")
        self._engine   = None

    def connect(self) -> None:
        if not self._host:
            raise ConnectorError("REDSHIFT_HOST must be set.")
        try:
            from sqlalchemy import create_engine, text
            dsn = (
                f"postgresql+psycopg2://{self._user}:{self._password}"
                f"@{self._host}:{self._port}/{self._database}"
            )
            self._engine = create_engine(
                dsn,
                pool_pre_ping=True,
                connect_args={
                    "sslmode": "require",
                    "options": f"-csearch_path={self._schema}",
                },
            )
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(
                "Redshift connected: host=%s db=%s schema=%s",
                self._host, self._database, self._schema,
            )
        except ImportError:
            raise ConnectorError(
                "psycopg2 not installed. Run: pip install psycopg2-binary"
            )
        except Exception as e:
            raise ConnectorError(f"Redshift connection failed: {e}") from e

    def execute(self, query: str) -> pd.DataFrame:
        if self._engine is None:
            raise ConnectorError("Not connected. Call connect() first.")
        try:
            return pd.read_sql(query, self._engine)
        except Exception as e:
            raise ConnectorError(f"Redshift query failed: {e}") from e

    def get_schema(self) -> dict:
        if self._engine is None:
            raise ConnectorError("Not connected.")
        q = f"""
            SELECT tablename, "column", type
            FROM pg_table_def
            WHERE schemaname = '{self._schema}'
            ORDER BY tablename, "column"
        """
        df = self.execute(q)
        schema: dict = {}
        for _, row in df.iterrows():
            t = row["tablename"]
            schema.setdefault(t, []).append({
                "name": row["column"],
                "type": row["type"],
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
            f"SELECT DISTINCT tablename FROM pg_table_def "
            f"WHERE schemaname='{self._schema}' ORDER BY tablename"
        )
        return df["tablename"].tolist()
