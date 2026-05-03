"""
GET /v1/traces/{trace_id} — return the structured trace for one pipeline turn.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

router = APIRouter(tags=["traces"])


class ToolCallRecord(BaseModel):
    tool_name: str
    args: dict
    result: dict | str | None


class TraceResponse(BaseModel):
    trace_id: str
    session_id: str
    routed_to: str
    tool_calls: list[ToolCallRecord]
    retrieved_chunk_ids: list[str]
    latency_ms: int


from app.db import models
from fastapi import HTTPException
from sqlalchemy import select

@router.get("/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(
    trace_id: str,
    db: AsyncSession = Depends(get_db),
) -> TraceResponse:
    """Return trace for one turn. 404 if not found."""
    stmt = select(models.AgentTrace).where(models.AgentTrace.trace_id == trace_id)
    result = await db.execute(stmt)
    trace = result.scalar_one_or_none()
    
    if not trace:
        raise HTTPException(status_code=404, detail="TRACE_NOT_FOUND")
    
    return TraceResponse(
        trace_id=trace.trace_id,
        session_id=trace.session_id,
        routed_to=trace.routed_to,
        tool_calls=[
            ToolCallRecord(
                tool_name=tc.get("tool_name"),
                args=tc.get("args"),
                result=tc.get("result")
            ) for tc in trace.tool_calls
        ],
        retrieved_chunk_ids=trace.retrieved_chunk_ids,
        latency_ms=trace.latency_ms
    )
