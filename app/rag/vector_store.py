"""
Vector-store dispatcher.

Two backends:

* ``chroma`` — local file-based persistent store. Default; no extra
  services to run, no extra deps. See ``app/rag/chroma_store.py``.
* ``pgvector`` — Postgres + pgvector extension. Drop-in alternative for
  multi-process / multi-host deployments. See
  ``app/rag/pgvector_store.py``. Requires ``pip install -e ".[pgvector]"``.

The dispatcher exposes the same async API both backends implement —
``upsert`` / ``query`` / ``count`` / ``reset_collection_cache`` — so
``search_docs.py``, ``ingest.py``, and the eval harness don't need to
know which backend is selected. The choice is made by
``settings.vector_store_backend`` and resolved on every call (not cached
to a module global) so test fixtures and demos can flip it via env vars.

For backwards-compat with code that pokes at the chroma collection
directly (the eval harness does this to resolve chunk_id → source), we
keep the legacy ``_get_collection_sync`` symbol as a re-export of the
chroma backend's helper.
"""
from __future__ import annotations

from typing import Any

from app.rag import chroma_store
from app.rag.chroma_store import _get_collection_sync  # noqa: F401 - re-export for eval
from app.settings import settings


def _backend_module() -> Any:
    backend = (settings.vector_store_backend or "chroma").lower()
    if backend == "pgvector":
        from app.rag import pgvector_store  # local import — optional dep

        return pgvector_store
    return chroma_store


def reset_collection_cache() -> None:
    """Reset both backends' caches (no-op for the unselected one)."""
    chroma_store.reset_collection_cache()
    try:
        from app.rag import pgvector_store

        pgvector_store.reset_collection_cache()
    except Exception:  # noqa: BLE001 - pgvector deps may be missing; best-effort
        pass


async def upsert(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    return await _backend_module().upsert(ids, embeddings, documents, metadatas)


async def query(
    query_embedding: list[float],
    k: int,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await _backend_module().query(query_embedding, k, where)


async def count() -> int:
    return await _backend_module().count()
