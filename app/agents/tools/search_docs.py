"""
``search_docs`` tool — used by ``KnowledgeAgent``.

Embeds the user query with the same backend used at ingest time, queries the
Chroma collection, converts cosine distance to a [0, 1] score, and returns
ordered chunks. Chunk IDs are part of the result so the agent can cite
sources, and the pipeline can record them on the trace.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.rag.embeddings import embed_query
from app.rag.vector_store import query as vs_query


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _distance_to_score(distance: float) -> float:
    """Cosine distance in [0, 2] → similarity in [0, 1]. Clamp for safety."""
    score = 1.0 - float(distance)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return round(score, 4)


async def search_docs(
    query: str, k: int = 5, product_area: str | None = None
) -> list[DocChunk]:
    """Search the vector store for top-k relevant chunks.

    Args:
        query: natural language query from the user.
        k: number of chunks to return (default 5).
        product_area: optional metadata filter, e.g. ``"security"``.

    Returns:
        A list of ``DocChunk`` ordered by descending similarity score.
        Empty list if the vector store is empty or no chunks match.
    """
    if not query or not query.strip():
        return []
    query_vec = await asyncio.to_thread(embed_query, query, "auto")

    where: dict[str, Any] | None = None
    if product_area:
        where = {"product_area": product_area}

    raw = await vs_query(query_vec, k=k, where=where)

    ids_batch = raw.get("ids") or [[]]
    distances_batch = raw.get("distances") or [[]]
    documents_batch = raw.get("documents") or [[]]
    metadatas_batch = raw.get("metadatas") or [[]]

    ids = ids_batch[0] if ids_batch else []
    distances = distances_batch[0] if distances_batch else []
    documents = documents_batch[0] if documents_batch else []
    metadatas = metadatas_batch[0] if metadatas_batch else []

    chunks: list[DocChunk] = []
    for chunk_id, distance, doc, meta in zip(ids, distances, documents, metadatas):
        chunks.append(
            DocChunk(
                chunk_id=chunk_id,
                score=_distance_to_score(distance),
                content=doc or "",
                metadata=dict(meta or {}),
            )
        )
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks


def format_chunks_for_agent(chunks: list[DocChunk]) -> str:
    """Pretty-print chunks for inclusion in an LLM prompt / tool result."""
    if not chunks:
        return "No relevant documentation chunks were found."
    parts: list[str] = []
    for c in chunks:
        source = c.metadata.get("source", "unknown")
        parts.append(
            f"[{c.chunk_id}] (score={c.score:.2f}, source={source})\n{c.content}"
        )
    return "\n\n---\n\n".join(parts)
