"""
connectors/postgres_connector.py
PostgreSQL connector using SQLAlchemy + psycopg2.

Env vars:
  DATABASE_URL=postgresql://user:pass@host:5432/dbname
  Or individual: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASS
"""

from __future__ import annotations
import os
import pandas as pd
from .base_connector import BaseConnector
from core.exceptions import ConnectorError
from core.logger import get_logger

logger = get_logger(__name__)


def _build_dsn() -> str:
    url = os.getenv("DATABASE_URL", "")
    if url and url.startswith("postgresql"):
        return url
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "analytics")
    user = os.getenv("POSTGRES_USER", "postgres")
    pw   = os.getenv("POSTGRES_PASS", "")
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"


class PostgresConnector(BaseConnector):

    def __init__(self, dsn: str = None, schema: str = "public"):
        self._dsn    = dsn or _build_dsn()
        self._schema = schema
        self._engine = None

    def connect(self) -> None:
        try:
            from sqlalchemy import create_engine, text
            self._engine = create_engine(
                self._dsn,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                connect_args={"options": f"-csearch_path={self._schema}"},
            )
            # Smoke test
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("PostgreSQL connected: schema=%s", self._schema)
        except ImportError:
            raise ConnectorError("psycopg2 not installed. Run: pip install psycopg2-binary")
        except Exception as e:
            raise ConnectorError(f"PostgreSQL connection failed: {e}") from e

    def execute(self, query: str) -> pd.DataFrame:
        if self._engine is None:
            raise ConnectorError("Not connected. Call connect() first.")
        try:
            return pd.read_sql(query, self._engine)
        except Exception as e:
            raise ConnectorError(f"Query failed: {e}") from e

    def get_schema(self) -> dict:
        if self._engine is None:
            raise ConnectorError("Not connected.")
        q = f"""
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = '{self._schema}'
            ORDER BY table_name, ordinal_position
        """
        df = pd.read_sql(q, self._engine)
        schema: dict = {}
        for _, row in df.iterrows():
            t = row["table_name"]
            schema.setdefault(t, []).append({
                "name": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
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
            f"SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema='{self._schema}' ORDER BY table_name"
        )
        return df["table_name"].tolist()

    def sample(self, table: str, n: int = 5) -> pd.DataFrame:
        return self.execute(f'SELECT * FROM "{self._schema}"."{table}" LIMIT {n}')
