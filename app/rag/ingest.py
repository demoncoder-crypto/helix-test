"""
RAG ingest CLI.

Walks a directory of markdown files, extracts YAML frontmatter, splits each
file into heading-aware chunks, embeds the chunks, and upserts them into
the Chroma vector store.

Stable chunk IDs (``sha256(file::chunk_index)``) make re-ingest idempotent
— the spec explicitly checks this. Embedding batches are kept small to
stay under provider rate limits.

Usage::

    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 800 --reset
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.chunker import chunk_markdown, extract_frontmatter, make_chunk_id
from app.rag.embeddings import embed_texts
from app.rag.vector_store import upsert as vs_upsert


@dataclass
class IngestStats:
    files: int = 0
    chunks: int = 0


def _flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be primitives. Stringify lists/dicts."""
    flat: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
    return flat


async def _embed_in_batches(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Run embeddings in batches; embeddings are sync so we offload to a thread."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = await asyncio.to_thread(embed_texts, batch, "retrieval_document", "auto")
        out.extend(vectors)
    return out


async def ingest_directory(
    docs_path: Path, chunk_size: int = 800, chunk_overlap: int = 1
) -> IngestStats:
    """Ingest every ``*.md`` under ``docs_path`` into the vector store."""
    md_files = sorted(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")
    stats = IngestStats(files=len(md_files))

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        frontmatter, body = extract_frontmatter(text)
        chunks = chunk_markdown(body, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            print(f"  {file_path.name}: 0 chunks (skipped)")
            continue

        source = file_path.name
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(source, idx)
            ids.append(chunk_id)
            base_meta = _flatten_metadata(frontmatter)
            base_meta.update(
                {
                    "source": source,
                    "chunk_index": idx,
                    "title": base_meta.get("title", file_path.stem),
                    "product_area": base_meta.get("product_area", "general"),
                }
            )
            metadatas.append(base_meta)

        embeddings = await _embed_in_batches(chunks)
        await vs_upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        stats.chunks += len(chunks)
        print(f"  {file_path.name}: {len(chunks)} chunks -> upserted")

    print(f"Ingest complete. {stats.files} files, {stats.chunks} chunks.")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=1)
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"Path does not exist: {args.path}")

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
