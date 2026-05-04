"""
POST /v1/sessions — create a session.

Upserts the User row (so plan_tier is updated if it changed since last visit),
creates a Session row with an initial serialized SessionState, and returns the
generated session_id. All I/O is async via SQLAlchemy 2.x async.
"""
import uuid
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session as SessionModel
from app.db.models import User
from app.db.session import get_db
from app.srop.state import SessionState

router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    plan_tier: Literal["free", "pro", "enterprise"] = "free"


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """Create a new session bound to a user. Upsert the user record."""
    user_result = await db.execute(select(User).where(User.user_id == body.user_id))
    user = user_result.scalar_one_or_none()
    if user is None:
        user = User(user_id=body.user_id, plan_tier=body.plan_tier)
        db.add(user)
    else:
        user.plan_tier = body.plan_tier

    session_id = str(uuid.uuid4())
    initial_state = SessionState(user_id=body.user_id, plan_tier=body.plan_tier)
    session = SessionModel(
        session_id=session_id,
        user_id=body.user_id,
        state=initial_state.to_db_dict(),
    )
    db.add(session)
    await db.commit()

    return CreateSessionResponse(session_id=session_id, user_id=body.user_id)
