"""
ingestion/sql_parser.py
Handles: SQL database connections (SQLite, PostgreSQL, MySQL, Athena).
Accepts a connection string or pre-built engine.
Can run a provided query or auto-discover schema + sample tables.
"""

from __future__ import annotations
import pandas as pd
from ingestion.base_parser import BaseParser, ParsedDocument
from core.logger import get_logger

logger = get_logger(__name__)

_SAMPLE_ROWS = 5000   # max rows to pull per table in auto-discovery mode


class SQLParser(BaseParser):

    def __init__(self, connection_string: str = None, engine=None, query: str = None):
        self._conn_str = connection_string
        self._engine = engine
        self._query = query

    def can_parse(self, filename: str, mime_type: str = "") -> bool:
        return (filename.lower().endswith((".db", ".sqlite", ".sqlite3")) or
                "sql" in filename.lower() or
                "database" in mime_type)

    def parse(self, source=None, filename: str = "") -> ParsedDocument:
        doc = ParsedDocument(source_name=filename or "sql_source", source_type="sql")
        try:
            engine = self._get_engine(source, doc)
            if engine is None:
                return doc

            if self._query:
                df = pd.read_sql(self._query, engine)
                doc.dataframes.append(df)
                doc.table_names.append("query_result")
                doc.metadata["query"] = self._query
                logger.info(f"SQL query returned {len(df)} rows")
            else:
                self._auto_discover(engine, doc)

        except Exception as e:
            doc.add_warning(f"SQL parse error: {e}")
        return doc

    def _get_engine(self, source, doc):
        try:
            from sqlalchemy import create_engine
            if self._engine:
                return self._engine
            conn_str = self._conn_str or str(source)
            return create_engine(conn_str)
        except ImportError:
            doc.add_warning("sqlalchemy not installed. Run: pip install sqlalchemy")
        except Exception as e:
            doc.add_warning(f"DB connection failed: {e}")
        return None

    def _auto_discover(self, engine, doc: ParsedDocument):
        try:
            from sqlalchemy import inspect
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            doc.metadata["tables_available"] = tables

            for table in tables[:10]:     # cap at 10 tables
                try:
                    df = pd.read_sql(f"SELECT * FROM {table} LIMIT {_SAMPLE_ROWS}", engine)
                    if not df.empty:
                        doc.dataframes.append(df)
                        doc.table_names.append(table)
                        logger.info(f"Table '{table}': {df.shape}")
                except Exception as e:
                    doc.add_warning(f"Table '{table}' sample failed: {e}")
        except Exception as e:
            doc.add_warning(f"Schema discovery failed: {e}")
