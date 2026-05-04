"""
KnowledgeAgent — answers product / docs questions using RAG.

This sub-agent has exactly one tool: ``search_docs``. Its system instruction
forces it to call the tool, ground its answer in the returned chunks, and
cite chunk IDs in square brackets. If retrieval comes back empty, it must
say so rather than hallucinate.
"""
from __future__ import annotations

from typing import Any

from app.agents.tools.search_docs import format_chunks_for_agent, search_docs
from app.settings import settings

KNOWLEDGE_INSTRUCTION = """
You are the Helix Knowledge Agent. You answer questions about Helix product
features and documentation.

PROCESS (mandatory, in order):
1. Call the `search_docs` tool with the user's query to retrieve top-k chunks.
2. If the result is empty or contains no relevant chunks, reply exactly:
   "I don't have documentation on that yet." Do not invent an answer.
3. Otherwise, write a concise answer (2-5 sentences) using ONLY the
   information in the retrieved chunks.
4. Always cite chunk IDs inline in square brackets, e.g.:
   "Rotate the key under Settings → Security [chunk_abc123]."
   Cite every distinct chunk you used.

NEVER answer from prior knowledge. NEVER skip step 1.
"""


async def search_docs_tool(query: str, k: int = 5) -> dict[str, Any]:
    """Search Helix product documentation.

    Args:
        query: Natural language question from the user.
        k: How many top chunks to retrieve. Default 5.

    Returns:
        A dict with ``chunks`` (formatted text the model should ground on)
        and ``chunk_ids`` (list of stable IDs, used for citations and
        traces).
    """
    chunks = await search_docs(query=query, k=k)
    return {
        "chunks": format_chunks_for_agent(chunks),
        "chunk_ids": [c.chunk_id for c in chunks],
        "count": len(chunks),
    }


def build_knowledge_agent() -> Any:
    """Lazy import of google-adk so the rest of the app can be loaded
    (and tested with mocks) without the heavy dependency available.
    """
    from google.adk.agents import LlmAgent

    return LlmAgent(
        name="knowledge_agent",
        model=settings.adk_model,
        description=(
            "Answers questions about Helix product features and documentation "
            "by searching the docs corpus and citing chunk IDs."
        ),
        instruction=KNOWLEDGE_INSTRUCTION,
        tools=[search_docs_tool],
    )
