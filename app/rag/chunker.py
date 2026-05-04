"""
Markdown chunker + frontmatter extractor.

Strategy: heading-aware splitting (split on `## ` / `### `), with sentence-level
sub-splitting for any section that exceeds `max_chars`. The Helix corpus uses
clean Markdown structure; this preserves section coherence while bounding
chunk size.

All functions are pure — no I/O, fully unit-testable.
"""
from __future__ import annotations

import hashlib
import re

import yaml

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_SPLIT_RE = re.compile(r"\n(?=#{2,3} )")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_frontmatter(text: str) -> tuple[dict, str]:
    """Return ``(metadata_dict, body_without_frontmatter)``.

    Falls back to ``({}, text)`` when no frontmatter block is present or
    when the YAML cannot be parsed (a malformed file should still be ingested).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, text[match.end():]


def _split_sentences(text: str, max_chars: int, overlap_sentences: int = 1) -> list[str]:
    """Pack sentences into chunks no larger than ``max_chars`` with a small overlap."""
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current).strip())
            keep = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(keep)
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]


def chunk_markdown(text: str, chunk_size: int = 800, overlap: int = 1) -> list[str]:
    """Split markdown into heading-aligned chunks.

    Args:
        text: raw markdown (frontmatter already stripped is fine but not required).
        chunk_size: soft maximum characters per chunk; sections longer than
            this get sub-split sentence-by-sentence.
        overlap: number of trailing sentences to repeat at the start of the
            next chunk when sub-splitting (helps preserve context).
    """
    body = text.strip()
    if not body:
        return []

    sections = _HEADING_SPLIT_RE.split(body)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(
                _split_sentences(section, max_chars=chunk_size, overlap_sentences=overlap)
            )
    return [c for c in chunks if c.strip()]


def make_chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic chunk ID — re-ingest must not duplicate."""
    raw = f"{source}::{chunk_index}".encode()
    return "chunk_" + hashlib.sha256(raw).hexdigest()[:16]
