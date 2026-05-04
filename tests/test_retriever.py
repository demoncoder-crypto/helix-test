"""
Unit tests for RAG retrieval and the chunker.

The retriever test runs against a temporary Chroma directory ingested with
the local (deterministic, no-API-key) embedding backend. This keeps the
test self-contained: no secrets, no network.
"""
from __future__ import annotations

import pytest


def test_chunker_produces_non_empty_chunks() -> None:
    from app.rag.chunker import chunk_markdown

    text = (
        "# Doc\n\n"
        "Intro paragraph.\n\n"
        "## Section A\n\nSome content for A.\n\n"
        "### Subsection A.1\n\nMore A content.\n\n"
        "## Section B\n\nContent for B is here."
    )
    chunks = chunk_markdown(text, chunk_size=200, overlap=1)
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)


def test_chunk_ids_are_stable() -> None:
    from app.rag.chunker import make_chunk_id

    a = make_chunk_id("deploy-keys.md", 0)
    b = make_chunk_id("deploy-keys.md", 0)
    c = make_chunk_id("deploy-keys.md", 1)
    assert a == b
    assert a != c
    assert a.startswith("chunk_")


def test_extract_frontmatter_parses_yaml() -> None:
    from app.rag.chunker import extract_frontmatter

    text = (
        "---\n"
        "title: Deploy Keys\n"
        "product_area: security\n"
        "---\n"
        "# Body\n"
        "content here\n"
    )
    meta, body = extract_frontmatter(text)
    assert meta == {"title": "Deploy Keys", "product_area": "security"}
    assert body.startswith("# Body")


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids(seed_vector_store) -> None:
    """search_docs must return chunk IDs and scores in [0, 1]."""
    from app.agents.tools.search_docs import search_docs

    results = await search_docs("how do I rotate a deploy key", k=3)
    assert len(results) > 0
    assert all(r.chunk_id and r.chunk_id.startswith("chunk_") for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert all(r.content for r in results)
