"""
POST /v1/sessions — create a session.
"""
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str
    plan_tier: str = "free"


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str


from app.db import models
from app.srop.state import SessionState
from sqlalchemy.dialects.sqlite import insert

@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """
    Create a new session. Upsert the user if not seen before.
    Initialize SessionState and persist to DB.
    """
    # 1. Upsert User
    # In SQLite, we use the insert().on_conflict_do_update() or similar.
    # For now, let's just do a simple check and insert.
    stmt = insert(models.User).values(
        user_id=body.user_id,
        plan_tier=body.plan_tier
    ).on_conflict_do_update(
        index_elements=["user_id"],
        set_={"plan_tier": body.plan_tier}
    )
    await db.execute(stmt)

    # 2. Create Session
    session_id = str(uuid.uuid4())
    initial_state = SessionState(
        user_id=body.user_id,
        plan_tier=body.plan_tier
    )
    
    new_session = models.Session(
        session_id=session_id,
        user_id=body.user_id,
        state=initial_state.to_db_dict()
    )
    db.add(new_session)
    await db.commit()

    return CreateSessionResponse(
        session_id=session_id,
        user_id=body.user_id
    )
