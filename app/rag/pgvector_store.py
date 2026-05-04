"""
Postgres + pgvector backend (Extension).

A drop-in replacement for the Chroma backend exposing the same
``count`` / ``upsert`` / ``query`` async API. It:

* Lazy-creates the ``vector`` extension and the
  ``settings.pgvector_table`` table on first use.
* Stores ``id`` (text PK), ``embedding`` (``vector(768)``), ``document``
  (text), ``metadata`` (jsonb).
* Queries use cosine distance (``<=>``) and return the same shape as
  Chroma (``ids``/``distances``/``documents``/``metadatas`` are lists
  of lists, one per query embedding) so ``search_docs.py`` doesn't need
  to know which backend it's talking to.

The connection pool is opened lazily on first call and reused. Pool
sizing is conservative because this backend is meant to slot into the
same single-process FastAPI worker the rest of the app runs in.

To use it::

    pip install -e ".[pgvector]"
    docker compose -f docker-compose.pg.yml up -d
    PGVECTOR_DATABASE_URL=postgresql://helix:helix@localhost:5432/helix_srop \\
      VECTOR_STORE_BACKEND=pgvector \\
      python -m app.rag.ingest

Switching the running app over is just two env vars; no schema migration
is needed because session/message storage stays where it is (SQLite or
its own Postgres in ``DATABASE_URL``).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from app.settings import settings

log = structlog.get_logger()

_EMBEDDING_DIM = 768
_pool: Any | None = None
_init_lock = asyncio.Lock()
_initialized = False


def _connection_url() -> str:
    url = settings.pgvector_database_url.strip()
    if not url:
        url = settings.database_url
    if url.startswith("postgresql+asyncpg://"):
        url = "postgresql://" + url[len("postgresql+asyncpg://") :]
    if not url.startswith("postgresql://"):
        raise RuntimeError(
            f"pgvector backend needs a postgresql:// URL, got: {url[:30]}..."
        )
    return url


async def _ensure_pool() -> Any:
    global _pool, _initialized
    if _pool is None:
        try:
            import asyncpg
        except ImportError as exc:
            raise RuntimeError(
                "pgvector backend selected but asyncpg is not installed. "
                'Run `pip install -e ".[pgvector]"`.'
            ) from exc
        _pool = await asyncpg.create_pool(
            dsn=_connection_url(),
            min_size=1,
            max_size=4,
            command_timeout=30,
        )
    if not _initialized:
        async with _init_lock:
            if not _initialized:
                async with _pool.acquire() as conn:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    await conn.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {settings.pgvector_table} (
                            id        TEXT PRIMARY KEY,
                            embedding VECTOR({_EMBEDDING_DIM}),
                            document  TEXT NOT NULL,
                            metadata  JSONB NOT NULL DEFAULT '{{}}'::jsonb
                        );
                        """
                    )
                    await conn.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {settings.pgvector_table}_embedding_idx
                        ON {settings.pgvector_table}
                        USING hnsw (embedding vector_cosine_ops);
                        """
                    )
                _initialized = True
                log.info("pgvector_initialized", table=settings.pgvector_table)
    return _pool


def _format_vec(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.7f}" for v in vec) + "]"


def reset_collection_cache() -> None:
    """Drop the cached pool so the next call re-opens it.

    Used by tests or by config-reload paths. Closing the pool is
    fire-and-forget — if it can't be awaited (no running loop), the
    pool is just dropped.
    """
    global _pool, _initialized
    if _pool is not None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_pool.close())
            else:
                loop.run_until_complete(_pool.close())
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
    _pool = None
    _initialized = False


async def upsert(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    if not ids:
        return
    pool = await _ensure_pool()
    rows = [
        (
            cid,
            _format_vec(emb),
            doc,
            json.dumps(meta or {}),
        )
        for cid, emb, doc, meta in zip(ids, embeddings, documents, metadatas)
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            f"""
            INSERT INTO {settings.pgvector_table} (id, embedding, document, metadata)
            VALUES ($1, $2::vector, $3, $4::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                document  = EXCLUDED.document,
                metadata  = EXCLUDED.metadata;
            """,
            rows,
        )


async def query(
    query_embedding: list[float],
    k: int,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pool = await _ensure_pool()
    qvec = _format_vec(query_embedding)

    where_sql = ""
    params: list[Any] = [qvec, k]
    if where:
        clauses = []
        for i, (key, value) in enumerate(where.items(), start=3):
            clauses.append(f"metadata->>${i - 1} = ${i}")
            params.extend([key, str(value)])
        if clauses:
            where_sql = "WHERE " + " AND ".join(clauses)

    sql = f"""
        SELECT id, document, metadata, embedding <=> $1::vector AS distance
        FROM {settings.pgvector_table}
        {where_sql}
        ORDER BY embedding <=> $1::vector
        LIMIT $2;
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    ids = [r["id"] for r in rows]
    distances = [float(r["distance"]) for r in rows]
    documents = [r["document"] for r in rows]
    metadatas = [
        json.loads(r["metadata"]) if isinstance(r["metadata"], str) else dict(r["metadata"] or {})
        for r in rows
    ]
    return {
        "ids": [ids],
        "distances": [distances],
        "documents": [documents],
        "metadatas": [metadatas],
    }


async def count() -> int:
    pool = await _ensure_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT COUNT(*) AS n FROM {settings.pgvector_table};")
    return int(row["n"]) if row else 0
