"""
SROP entrypoint — called by the message route.

Design:
  State persistence strategy: **Pattern 3** — SessionState stored as JSON
  in the `sessions.state` column.  On every turn we:
    1. Load SessionState from DB  →  build dynamic root agent with it injected
    2. Build recent-history string from stored Message rows for conversational
       context (last 10 messages, newest first)
    3. Run ADK with a fresh InMemoryRunner (stateless per turn)
    4. Extract routed_to and tool_calls from ADK event stream
    5. Persist updated SessionState + new messages + trace to DB

  Why Pattern 3?
  - Simplest: no custom BaseSessionService implementation required
  - Survives process restarts trivially (state is in SQLite)
  - The small context overhead (< 500 tokens) is worth the simplicity

  Timeout: asyncio.wait_for wraps the full event iteration coroutine.
  If the LLM doesn't respond within llm_timeout_seconds, we raise
  UpstreamTimeoutError which the route converts to 504.
"""
import asyncio
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog
from google.genai import types as genai_types
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import build_root_agent
from app.api.errors import UpstreamTimeoutError
from app.db.models import AgentTrace, Message, Session, User
from app.settings import settings
from app.srop.state import SessionState

log = structlog.get_logger()

# Regex to extract chunk IDs embedded in search_docs output
_CHUNK_ID_RE = re.compile(r"\[(chunk_[a-f0-9]+)\]")

from app.agents.tools.support_tools import db_session_var

# Number of recent messages to inject as history (user+assistant pairs)
_HISTORY_TURNS = 5


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


def _format_history(messages: list[Message]) -> str:
    """Format recent DB messages as a conversation string for the agent instruction."""
    if not messages:
        return ""
    lines: list[str] = []
    for msg in messages[-(_HISTORY_TURNS * 2) :]:
        prefix = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{prefix}: {msg.content}")
    return "\n".join(lines)


    return final_content, routed_to, tool_calls, retrieved_chunk_ids


async def _stream_events(
    runner: Any,
    user_id: str,
    session_id: str,
    new_message: genai_types.Content,
):
    """Generator that yields ADK events and finally returns the same tuple as _collect_events."""
    routed_to = "srop_root"
    tool_calls: list[dict] = []
    retrieved_chunk_ids: list[str] = []
    final_content = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    ):
        # Yield the event for SSE streaming
        yield event

        # --- Standard extraction logic ---
        for fc in event.get_function_calls() or []:
            tool_calls.append({"tool_name": fc.name, "args": dict(fc.args) if fc.args else {}, "result": None})
        for fr in event.get_function_responses() or []:
            resp_str = str(fr.response) if fr.response else ""
            for tc in reversed(tool_calls):
                if tc["tool_name"] == fr.name and tc["result"] is None:
                    tc["result"] = resp_str[:2000]
                    break
            retrieved_chunk_ids.extend(_CHUNK_ID_RE.findall(resp_str))
        if event.is_final_response():
            if event.content and event.content.parts:
                final_content = event.content.parts[0].text or ""
            routed_to = event.author or "srop_root"

    # Store results on the runner object or similar so the caller can access them after iteration?
    # Actually, we'll just yield a special final object or use a wrapper.
    yield (final_content, routed_to, tool_calls, retrieved_chunk_ids)


async def _collect_events(
    runner: Any,
    user_id: str,
    session_id: str,
    new_message: genai_types.Content,
) -> tuple[str, str, list[dict], list[str]]:
    """Legacy wrapper for synchronous collection."""
    final_res = None
    async for item in _stream_events(runner, user_id, session_id, new_message):
        if isinstance(item, tuple):
            final_res = item
    return final_res  # type: ignore


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    """
    Run one pipeline turn for a session.

    Args:
        session_id: The session to run.
        user_message: The user's latest message.
        db: Async SQLAlchemy session (injected by FastAPI).

    Returns:
        PipelineResult with the reply, routing decision, and trace ID.

    Raises:
        ValueError: If the session doesn't exist.
        UpstreamTimeoutError: If the LLM exceeds the configured timeout.
    """
    trace_id = str(uuid.uuid4())
    start_time = time.monotonic()

    structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)
    log.info("pipeline_started", user_message_len=len(user_message))
    
    # Bind DB session for tools
    db_token = db_session_var.set(db)

    # ── 1. Load session ───────────────────────────────────────────────────────
    stmt = select(Session).where(Session.session_id == session_id)
    result = await db.execute(stmt)
    db_session = result.scalar_one_or_none()
    if not db_session:
        raise ValueError("Session not found")

    # ── 2. Load recent message history ───────────────────────────────────────
    hist_stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    hist_result = await db.execute(hist_stmt)
    db_messages = list(hist_result.scalars().all())
    recent_history = _format_history(db_messages)

    # ── 3. Re-hydrate session state ───────────────────────────────────────────
    state = SessionState.from_db_dict(db_session.state)

    # ── 4. Build dynamic agent (Pattern 3 — state injected into instruction) ─
    dynamic_agent = build_root_agent(state, recent_history)

    # ── 5. Set up ADK runner and create session ───────────────────────────────
    from google.adk.runners import InMemoryRunner

    runner = InMemoryRunner(agent=dynamic_agent)
    # Create a fresh ADK session (state lives in our DB; ADK session is ephemeral)
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=db_session.user_id,
        session_id=session_id,
    )

    new_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message)],
    )

    # ── 6. Run agent with timeout ─────────────────────────────────────────────
    try:
        final_content, routed_to, tool_calls, retrieved_chunk_ids = (
            await asyncio.wait_for(
                _collect_events(runner, db_session.user_id, session_id, new_message),
                timeout=settings.llm_timeout_seconds,
            )
        )
    except asyncio.TimeoutError:
        log.error("pipeline_timeout", timeout_s=settings.llm_timeout_seconds)
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        )

    # Normalise routed_to to a known enum value
    known_agents = {"knowledge_agent", "account_agent", "support_agent"}
    if routed_to in known_agents:
        normalised = routed_to.replace("_agent", "")  # "knowledge", "account", "support"
    else:
        normalised = "smalltalk"

    log.info(
        "pipeline_completed",
        routed_to=normalised,
        tool_call_count=len(tool_calls),
        chunk_count=len(retrieved_chunk_ids),
    )

    # ── 7. Persist messages ───────────────────────────────────────────────────
    now = datetime.now(timezone.utc)

    user_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=user_message,
        created_at=now,
    )
    db.add(user_msg)

    assistant_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=final_content,
        trace_id=trace_id,
        created_at=now,
    )
    db.add(assistant_msg)

    # ── 8. Update session state ───────────────────────────────────────────────
    state.turn_count += 1
    state.last_agent = normalised if normalised in ("knowledge", "account") else None
    db_session.state = state.to_db_dict()
    db_session.updated_at = now

    # ── 9. Write trace ────────────────────────────────────────────────────────
    latency_ms = int((time.monotonic() - start_time) * 1000)
    trace = AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        routed_to=normalised,
        tool_calls=tool_calls,
        retrieved_chunk_ids=list(dict.fromkeys(retrieved_chunk_ids)),  # dedup order-preserving
        latency_ms=latency_ms,
        created_at=now,
    )
    db.add(trace)

    await db.commit()
    db_session_var.reset(db_token)

    structlog.contextvars.unbind_contextvars("session_id", "trace_id")
    return PipelineResult(
        content=final_content,
        routed_to=normalised,
        trace_id=trace_id,
    )


async def run_stream(session_id: str, user_message: str, db: AsyncSession):
    """Streaming version of the pipeline."""
    trace_id = str(uuid.uuid4())
    start_time = time.monotonic()
    
    # 1-3. Standard Setup
    stmt = select(Session).where(Session.session_id == session_id)
    result = await db.execute(stmt)
    db_session = result.scalar_one_or_none()
    if not db_session:
        raise ValueError("Session not found")
    
    state = SessionState.from_db_dict(db_session.state)
    hist_stmt = select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    hist_result = await db.execute(hist_stmt)
    db_messages = list(hist_result.scalars().all())
    recent_history = _format_history(db_messages)
    
    dynamic_agent = build_root_agent(state, recent_history)
    
    from google.adk.runners import InMemoryRunner
    runner = InMemoryRunner(agent=dynamic_agent)
    await runner.session_service.create_session(app_name=runner.app_name, user_id=db_session.user_id, session_id=session_id)

    new_message = genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)])
    
    db_token = db_session_var.set(db)
    final_tuple = None
    
    try:
        async for item in _stream_events(runner, db_session.user_id, session_id, new_message):
            if isinstance(item, tuple):
                final_tuple = item
            else:
                # Yield raw event for the API to format as SSE
                yield item
    finally:
        db_session_var.reset(db_token)

    if final_tuple:
        final_content, routed_to, tool_calls, retrieved_chunk_ids = final_tuple
        # Standard Persistence (copied from run)
        now = datetime.now(timezone.utc)
        user_msg = Message(message_id=str(uuid.uuid4()), session_id=session_id, role="user", content=user_message, created_at=now)
        db.add(user_msg)
        assistant_msg = Message(message_id=str(uuid.uuid4()), session_id=session_id, role="assistant", content=final_content, trace_id=trace_id, created_at=now)
        db.add(assistant_msg)
        
        state.turn_count += 1
        known_agents = {"knowledge_agent", "account_agent", "support_agent"}
        normalised = routed_to.replace("_agent", "") if routed_to in known_agents else "smalltalk"
        state.last_agent = normalised if normalised in ("knowledge", "account", "support") else None
        db_session.state = state.to_db_dict()
        
        latency_ms = int((time.monotonic() - start_time) * 1000)
        trace = AgentTrace(trace_id=trace_id, session_id=session_id, routed_to=normalised, tool_calls=tool_calls, retrieved_chunk_ids=list(dict.fromkeys(retrieved_chunk_ids)), latency_ms=latency_ms, created_at=now)
        db.add(trace)
        await db.commit()
