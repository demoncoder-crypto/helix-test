"""
Session state schema — persisted in sessions.state (JSON column).

Only store what the agent CANNOT re-derive from message history.
Keep it small — every turn loads and saves this.

TODO: extend if your design requires additional fields.
"""
from typing import Literal

from pydantic import BaseModel


class SessionState(BaseModel):
    user_id: str
    plan_tier: Literal["free", "pro", "enterprise"] = "free"
    last_agent: Literal["knowledge", "account", "smalltalk"] | None = None
    turn_count: int = 0

    def to_db_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_db_dict(cls, data: dict) -> "SessionState":
        return cls.model_validate(data)
