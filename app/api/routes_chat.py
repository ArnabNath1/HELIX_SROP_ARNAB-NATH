"""
POST /v1/chat/{session_id} — send a user message, get assistant reply.
"""
import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import UpstreamTimeoutError
from app.db.models import Session as DbSession, IdempotencyKey
from app.db.session import get_db
from app.srop import pipeline
import json

import structlog
log = structlog.get_logger()

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    reply: str
    routed_to: str
    trace_id: str


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    accept: str | None = Header(None),
) -> ChatResponse | StreamingResponse:
    """
    Run one turn of the SROP pipeline.

    Error responses:
    - 404 SESSION_NOT_FOUND  — session_id does not exist
    - 504 UPSTREAM_TIMEOUT   — LLM did not respond in time
    """
    # 0. Idempotency Check
    if idempotency_key:
        stmt = select(IdempotencyKey).where(IdempotencyKey.key == idempotency_key)
        res = await db.execute(stmt)
        existing = res.scalar_one_or_none()
        if existing:
            log.info("idempotency_hit", key=idempotency_key)
            return ChatResponse(**existing.response_body)

    # Pre-flight: verify session exists before entering the pipeline
    stmt = select(DbSession).where(DbSession.session_id == session_id)
    result = await db.execute(stmt)
    db_session = result.scalar_one_or_none()
    if db_session is None:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")

    # 1. Handle Streaming
    if accept == "text/event-stream":
        async def event_generator():
            async for event in pipeline.run_stream(session_id, body.content, db):
                # Format event for SSE
                # We can't easily serialize the ADK event to JSON directly if it has complex types,
                # but we can send the text parts or a simple summary.
                if event.content and event.content.parts:
                    txt = event.content.parts[0].text or ""
                    if txt:
                        yield f"data: {json.dumps({'text': txt})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        pr = await pipeline.run(session_id, body.content, db)
    except ValueError as exc:
        log.error("pipeline_value_error", error=str(exc))
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND") from exc
    except UpstreamTimeoutError as exc:
        log.error("pipeline_timeout", error=str(exc))
        raise HTTPException(status_code=504, detail="UPSTREAM_TIMEOUT") from exc
    except asyncio.TimeoutError as exc:
        log.error("pipeline_asyncio_timeout")
        raise HTTPException(status_code=504, detail="UPSTREAM_TIMEOUT") from exc
    except Exception as exc:
        log.error("pipeline_unexpected_error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    resp = ChatResponse(
        reply=pr.content,
        routed_to=pr.routed_to,
        trace_id=pr.trace_id,
    )

    # 2. Store Idempotency Key
    if idempotency_key:
        new_idemp = IdempotencyKey(
            key=idempotency_key,
            user_id=db_session.user_id,
            response_body=resp.model_dump()
        )
        db.add(new_idemp)
        await db.commit()

    return resp
