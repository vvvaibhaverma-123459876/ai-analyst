"""
connectors/athena_connector.py
AWS Athena connector using PyAthena.
Requires: pip install pyathena
"""

import pandas as pd
from typing import Optional

from .base_connector import BaseConnector
from core.config import config
from core.exceptions import ConnectorError
from core.logger import get_logger

logger = get_logger(__name__)


class AthenaConnector(BaseConnector):

    def __init__(
        self,
        region: str = None,
        s3_staging_dir: str = None,
        database: str = None,
    ):
        self._region = region or config.ATHENA_REGION
        self._s3_staging_dir = s3_staging_dir or config.ATHENA_S3_STAGING_DIR
        self._database = database or config.ATHENA_DATABASE
        self._connection = None

    def connect(self) -> None:
        try:
            from pyathena import connect as athena_connect
            self._connection = athena_connect(
                region_name=self._region,
                s3_staging_dir=self._s3_staging_dir,
                schema_name=self._database,
            )
            logger.info(f"Athena connected: db={self._database}, region={self._region}")
        except ImportError:
            raise ConnectorError("pyathena not installed. Run: pip install pyathena")
        except Exception as e:
            raise ConnectorError(f"Athena connection failed: {e}") from e

    def execute(self, query: str) -> pd.DataFrame:
        if self._connection is None:
            raise ConnectorError("Not connected. Call connect() first.")
        try:
            logger.info(f"Executing Athena query:\n{query}")
            df = pd.read_sql(query, self._connection)
            logger.info(f"Query returned {len(df)} rows")
            return df
        except Exception as e:
            raise ConnectorError(f"Athena query failed: {e}") from e

    def get_schema(self) -> dict:
        if self._connection is None:
            raise ConnectorError("Not connected. Call connect() first.")
        try:
            tables_df = pd.read_sql(
                f"SHOW TABLES IN {self._database}", self._connection
            )
            schema = {}
            for table in tables_df.iloc[:, 0].tolist():
                cols_df = pd.read_sql(
                    f"DESCRIBE {self._database}.{table}", self._connection
                )
                schema[table] = cols_df.to_dict("records")
            return schema
        except Exception as e:
            raise ConnectorError(f"Schema fetch failed: {e}") from e

    def test_connection(self) -> bool:
        try:
            if self._connection is None:
                return False
            pd.read_sql("SELECT 1", self._connection)
            return True
        except Exception:
            return False
