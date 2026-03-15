"""
memory/vector_store.py
Semantic vector memory for org intelligence.

Replaces SQLite keyword search in OrgMemory with dense vector retrieval.
Works in two modes:
  1. chromadb (local, default) — zero-infra, persisted on disk
  2. pgvector  (Postgres)      — production multi-tenant, set VECTOR_BACKEND=pgvector

Design principles:
  - Drop-in upgrade: OrgMemory delegates to VectorStore for all semantic queries
  - Falls back to keyword scan if embeddings unavailable (no API key)
  - Embeddings generated via OpenAI text-embedding-3-small or a local
    sentence-transformers model (internet_off_mode)
  - All writes are dual-write: both SQLite (for audit/export) and vector DB
"""

from __future__ import annotations
import json
import os
import hashlib
from pathlib import Path
from typing import Any
from core.logger import get_logger
from core.config import config

logger = get_logger(__name__)

VECTOR_DIR = Path(__file__).resolve().parent.parent / "memory" / "vectors"
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chromadb")   # chromadb | pgvector
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


class EmbeddingProvider:
    """Generates embeddings. Auto-selects provider based on config."""

    def __init__(self):
        self._local = config.__dict__.get("internet_off_mode", False) or \
                      os.getenv("INTERNET_OFF_MODE", "false").lower() == "true"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            if self._local:
                return self._embed_local(texts)
            if config.OPENAI_API_KEY:
                return self._embed_openai(texts)
            if config.ANTHROPIC_API_KEY:
                return self._embed_local(texts)     # anthropic has no embeddings API yet
            return self._embed_local(texts)
        except Exception as e:
            logger.warning(f"Embedding failed, using zeros: {e}")
            return [[0.0] * 384 for _ in texts]    # dim matches all-MiniLM-L6-v2

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in resp.data]

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(texts, show_progress_bar=False)
            return [e.tolist() for e in embeddings]
        except ImportError:
            logger.warning("sentence-transformers not installed; returning zero vectors")
            return [[0.0] * 384 for _ in texts]


class ChromaVectorStore:
    """Local ChromaDB-backed vector store."""

    COLLECTIONS = {
        "insights":    "insight_history",
        "corrections": "user_corrections",
        "patterns":    "analysis_patterns",
        "context":     "business_context",
    }

    def __init__(self):
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        self._embedder = EmbeddingProvider()
        self._client = None
        self._collections: dict[str, Any] = {}
        self._init()

    def _init(self):
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(VECTOR_DIR))
            for name in self.COLLECTIONS:
                self._collections[name] = self._client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
            logger.info("ChromaDB vector store initialised at %s", VECTOR_DIR)
        except ImportError:
            logger.warning("chromadb not installed — vector store disabled. "
                           "Run: pip install chromadb")
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}")

    def _available(self) -> bool:
        return self._client is not None

    def upsert(self, collection: str, doc_id: str, text: str, metadata: dict = None):
        if not self._available() or collection not in self._collections:
            return
        try:
            embedding = self._embedder.embed([text])[0]
            self._collections[collection].upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}],
            )
        except Exception as e:
            logger.warning(f"VectorStore upsert failed: {e}")

    def query(self, collection: str, query_text: str, n: int = 5,
              where: dict = None) -> list[dict]:
        if not self._available() or collection not in self._collections:
            return []
        try:
            embedding = self._embedder.embed([query_text])[0]
            kwargs: dict = dict(query_embeddings=[embedding], n_results=min(n, 10))
            if where:
                kwargs["where"] = where
            result = self._collections[collection].query(**kwargs)
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            return [
                {"text": d, "metadata": m, "score": round(1 - dist, 4)}
                for d, m, dist in zip(docs, metas, distances)
            ]
        except Exception as e:
            logger.warning(f"VectorStore query failed: {e}")
            return []

    def delete(self, collection: str, doc_id: str):
        if not self._available() or collection not in self._collections:
            return
        try:
            self._collections[collection].delete(ids=[doc_id])
        except Exception as e:
            logger.warning(f"VectorStore delete failed: {e}")


class PgVectorStore:
    """PostgreSQL + pgvector backend for production multi-tenant deployments."""

    def __init__(self):
        self._embedder = EmbeddingProvider()
        self._engine = None
        self._init()

    def _init(self):
        dsn = os.getenv("DATABASE_URL", "")
        if not dsn:
            logger.warning("DATABASE_URL not set — PgVectorStore disabled")
            return
        try:
            from sqlalchemy import create_engine, text
            self._engine = create_engine(dsn, pool_pre_ping=True)
            with self._engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS vector_docs (
                        id TEXT PRIMARY KEY,
                        collection TEXT NOT NULL,
                        tenant_id TEXT NOT NULL DEFAULT 'default',
                        embedding vector(384),
                        document TEXT,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT now()
                    )
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_vector_docs_collection
                    ON vector_docs(collection, tenant_id)
                """))
                conn.commit()
            logger.info("PgVectorStore initialised")
        except Exception as e:
            logger.warning(f"PgVectorStore init failed: {e}")
            self._engine = None

    def _available(self) -> bool:
        return self._engine is not None

    def upsert(self, collection: str, doc_id: str, text: str,
               metadata: dict = None, tenant_id: str = "default"):
        if not self._available():
            return
        try:
            from sqlalchemy import text as sql_text
            embedding = self._embedder.embed([text])[0]
            with self._engine.connect() as conn:
                conn.execute(sql_text("""
                    INSERT INTO vector_docs(id, collection, tenant_id, embedding, document, metadata)
                    VALUES (:id, :col, :tid, :emb, :doc, :meta)
                    ON CONFLICT(id) DO UPDATE SET
                        embedding=EXCLUDED.embedding,
                        document=EXCLUDED.document,
                        metadata=EXCLUDED.metadata
                """), {
                    "id": f"{tenant_id}:{doc_id}",
                    "col": collection, "tid": tenant_id,
                    "emb": str(embedding),
                    "doc": text, "meta": json.dumps(metadata or {}),
                })
                conn.commit()
        except Exception as e:
            logger.warning(f"PgVector upsert failed: {e}")

    def query(self, collection: str, query_text: str, n: int = 5,
              tenant_id: str = "default") -> list[dict]:
        if not self._available():
            return []
        try:
            from sqlalchemy import text as sql_text
            embedding = self._embedder.embed([query_text])[0]
            with self._engine.connect() as conn:
                rows = conn.execute(sql_text("""
                    SELECT document, metadata,
                           1 - (embedding <=> CAST(:emb AS vector)) AS score
                    FROM vector_docs
                    WHERE collection = :col AND tenant_id = :tid
                    ORDER BY embedding <=> CAST(:emb AS vector)
                    LIMIT :n
                """), {
                    "emb": str(embedding), "col": collection,
                    "tid": tenant_id, "n": n,
                }).fetchall()
            return [
                {"text": r[0], "metadata": json.loads(r[1]), "score": round(float(r[2]), 4)}
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"PgVector query failed: {e}")
            return []

    def delete(self, collection: str, doc_id: str, tenant_id: str = "default"):
        if not self._available():
            return
        try:
            from sqlalchemy import text as sql_text
            with self._engine.connect() as conn:
                conn.execute(sql_text(
                    "DELETE FROM vector_docs WHERE id = :id AND collection = :col"
                ), {"id": f"{tenant_id}:{doc_id}", "col": collection})
                conn.commit()
        except Exception as e:
            logger.warning(f"PgVector delete failed: {e}")


def get_vector_store():
    """Factory: return the configured vector store backend."""
    if VECTOR_BACKEND == "pgvector":
        store = PgVectorStore()
        if store._available():
            return store
        logger.warning("PgVector unavailable, falling back to ChromaDB")
    return ChromaVectorStore()


# Module-level singleton
_store: ChromaVectorStore | PgVectorStore | None = None


def vector_store() -> ChromaVectorStore | PgVectorStore:
    global _store
    if _store is None:
        _store = get_vector_store()
    return _store
