"""
Chroma persistent client wrapper.

All raw chroma calls are sync — they are funneled through this module so
that we have exactly one place that wraps them with ``asyncio.to_thread``
when called from async code. Collections are cached on the module to
avoid repeated client instantiation.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.settings import settings

_COLLECTION_NAME = "helix_docs"
_collection: Any | None = None


def _get_collection_sync() -> Any:
    global _collection
    if _collection is not None:
        return _collection
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    _collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def reset_collection_cache() -> None:
    """Drop the cached collection (used by tests that switch persist dirs)."""
    global _collection
    _collection = None


def upsert_sync(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    coll = _get_collection_sync()
    coll.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_sync(
    query_embedding: list[float],
    k: int,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    coll = _get_collection_sync()
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": k,
    }
    if where:
        kwargs["where"] = where
    return coll.query(**kwargs)


def count_sync() -> int:
    return _get_collection_sync().count()


async def upsert(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    await asyncio.to_thread(upsert_sync, ids, embeddings, documents, metadatas)


async def query(
    query_embedding: list[float],
    k: int,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(query_sync, query_embedding, k, where)


async def count() -> int:
    return await asyncio.to_thread(count_sync)
