"""
AccountAgent — handles questions about the user's account (builds, plan,
usage). Has two tools and a strict instruction telling it to call the right
one based on intent.

The user's ``user_id`` and ``plan_tier`` are *injected by the orchestrator*
into this agent's instruction at runtime (Pattern 3 from the ADK guide), so
the agent can pass them as tool arguments without re-asking.
"""
from __future__ import annotations

from typing import Any

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.settings import settings

ACCOUNT_INSTRUCTION_TEMPLATE = """
You are the Helix Account Agent. You answer questions about the current
user's account, builds, and usage.

The current user's identity has already been resolved:
- user_id: {user_id}
- plan_tier: {plan_tier}

Always pass these exact values to your tools. NEVER ask the user to
re-confirm them.

Tool selection:
- For build history / "recent builds" / "last failed builds" → call
  `get_recent_builds_tool(user_id={user_id_repr}, limit=N)` where N
  defaults to 5 unless the user asked for a different number.
- For plan, usage, limits, storage, or account status → call
  `get_account_status_tool(user_id={user_id_repr}, plan_tier={plan_tier_repr})`.

After the tool returns, summarize the result in a clear, short answer.
If the user asked about builds, list build_id, status, branch, and
duration in a readable format (no JSON dump).
"""


async def get_recent_builds_tool(user_id: str, limit: int = 5) -> dict[str, Any]:
    """Return the user's most recent CI builds, newest first.

    Args:
        user_id: The user's ID (already known from session context).
        limit: How many builds to return. Default 5, max 20.
    """
    capped = max(1, min(20, limit))
    builds = await get_recent_builds(user_id=user_id, limit=capped)
    return {"builds": builds, "count": len(builds)}


async def get_account_status_tool(user_id: str, plan_tier: str = "free") -> dict[str, Any]:
    """Return the user's plan, concurrent build usage, and storage usage.

    Args:
        user_id: The user's ID.
        plan_tier: Their plan tier (already known from session context).
    """
    return await get_account_status(user_id=user_id, plan_tier=plan_tier)


def build_account_agent(user_id: str, plan_tier: str) -> Any:
    """Build a fresh AccountAgent with the user context baked into the
    instruction. Created per turn — instruction has live state.
    """
    from google.adk.agents import LlmAgent

    instruction = ACCOUNT_INSTRUCTION_TEMPLATE.format(
        user_id=user_id,
        plan_tier=plan_tier,
        user_id_repr=repr(user_id),
        plan_tier_repr=repr(plan_tier),
    )
    return LlmAgent(
        name="account_agent",
        model=settings.adk_model,
        description=(
            "Answers questions about the current user's Helix account: "
            "recent builds, plan tier, and usage limits."
        ),
        instruction=instruction,
        tools=[get_recent_builds_tool, get_account_status_tool],
    )
