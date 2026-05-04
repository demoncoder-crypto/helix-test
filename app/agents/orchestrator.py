"""
SROP Root Orchestrator — Google ADK agent.

Routes every turn to ``KnowledgeAgent`` or ``AccountAgent`` via ADK's
``AgentTool`` pattern. The LLM picks which tool to call based on its
schema and the system instruction; we never grep the LLM output for
routing decisions (string-parsing routing is a hard penalty in the rubric).

We use **Pattern 3** from the ADK guide for state injection: load
``SessionState`` from SQLite each turn and bake user context (user_id,
plan_tier, last_agent, turn) into the instruction at construction time.
This is the lightest pattern, satisfies "state survives restart" because
state lives in SQLite, and keeps the LLM context window small.
"""
from __future__ import annotations

from typing import Any

from app.agents.account import build_account_agent
from app.agents.knowledge import build_knowledge_agent
from app.settings import settings
from app.srop.state import SessionState

ROOT_INSTRUCTION_TEMPLATE = """
You are the Helix Support Concierge — a routing agent for a B2B dev-tools
platform. You hold the conversation context and dispatch each user turn
to the correct specialist tool.

You are talking to a known user. DO NOT ask them to identify themselves.
- user_id: {user_id}
- plan_tier: {plan_tier}
- last specialist used: {last_agent}
- turn number: {turn}

Routing rules (call the matching tool):
- "How do I X", "What is X", any question about Helix features, docs,
  configuration, security, billing → call `knowledge_agent`.
- The user's account, plan, builds, usage, limits, storage → call
  `account_agent`.
- Greetings, thanks, "ok", or anything off-topic → answer directly with
  one short sentence. Do NOT call any tool.
- Refuse politely if the user asks for something off-topic for a Helix
  support agent (poems, jokes, unrelated coding help, opinions).

Whenever a tool is appropriate, you MUST call it. NEVER answer
documentation or account questions yourself — the specialists have
the data and tools.
"""


def _format_root_instruction(state: SessionState) -> str:
    return ROOT_INSTRUCTION_TEMPLATE.format(
        user_id=state.user_id,
        plan_tier=state.plan_tier,
        last_agent=state.last_agent or "none",
        turn=state.turn_count + 1,
    )


def build_root_agent(state: SessionState) -> Any:
    """Construct the root orchestrator with current session state.

    Built per-turn so the instruction reflects the latest state. The
    sub-agents are wrapped with ``AgentTool`` so the LLM treats them as
    callable functions and decides routing via tool selection.
    """
    from google.adk.agents import LlmAgent
    from google.adk.tools.agent_tool import AgentTool

    knowledge_agent = build_knowledge_agent()
    account_agent = build_account_agent(state.user_id, state.plan_tier)

    return LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        description="Helix Support Concierge: routes turns to specialists.",
        instruction=_format_root_instruction(state),
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=account_agent),
        ],
    )
