"""
Test fixtures.

Key fixtures:
- `client`: async test client backed by in-memory SQLite.
- `mock_pipeline`: patches pipeline.run at the ADK boundary so tests
  don't call the real LLM.
- `seeded_session`: creates a user+session and returns the session_id.
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from unittest.mock import AsyncMock

from app.db.models import Base
from app.db.session import get_db
from app.main import app
from app.srop.pipeline import PipelineResult


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    """Create all tables before each test, drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db: AsyncSession):
    """Async test client with DB overridden to in-memory SQLite."""

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_pipeline(monkeypatch):
    """
    Patch pipeline.run at the ADK boundary so no real LLM is called.

    The mock inspects the message content and routes:
    - "rotate" / "deploy key" / "how" → knowledge_agent, includes chunk IDs
    - "build" / "status" / "account" → account_agent
    - anything else                   → smalltalk (srop_root)

    Also stores plan_tier in the reply so state-persistence tests can assert
    that the second turn still knows the plan tier without re-asking.
    """

    async def _mock_run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
        from app.db.models import Session as DbSession, Message, AgentTrace, User
        from app.srop.state import SessionState
        from sqlalchemy import select
        import uuid, time
        from datetime import datetime, timezone

        # Load session state
        stmt = select(DbSession).where(DbSession.session_id == session_id)
        result = await db.execute(stmt)
        db_session = result.scalar_one_or_none()
        if not db_session:
            raise ValueError("Session not found")

        state = SessionState.from_db_dict(db_session.state)

        msg_lower = user_message.lower()
        # Use more specific keywords or word boundaries to avoid 'how' matching 'show'
        if any(k in msg_lower for k in ("rotate", "deploy key", "how to", "what is")):
            routed_to = "knowledge"
            reply = (
                f"According to [chunk_abc1234567890ab], to rotate a deploy key visit Settings > Deploy Keys. "
                f"Your plan is {state.plan_tier}."
            )
            chunk_ids = ["chunk_abc1234567890ab", "chunk_def1234567890ab"]
        elif any(k in msg_lower for k in ("build", "status", "account", "usage")):
            routed_to = "account"
            reply = f"Here are your recent builds. Plan: {state.plan_tier}."
            chunk_ids = []
        elif any(k in msg_lower for k in ("plan", "tier", "my plan")):
            # Follow-up: agent should know plan_tier from state
            routed_to = "knowledge"
            reply = f"Your current plan tier is {state.plan_tier}."
            chunk_ids = []
        else:
            routed_to = "smalltalk"
            reply = "Hello! How can I help you with Helix today?"
            chunk_ids = []

        trace_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        db.add(Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=user_message,
            created_at=now,
        ))
        db.add(Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=reply,
            trace_id=trace_id,
            created_at=now,
        ))

        state.turn_count += 1
        state.last_agent = routed_to if routed_to in ("knowledge", "account") else None
        db_session.state = state.to_db_dict()
        db_session.updated_at = now

        db.add(AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=routed_to,
            tool_calls=[{"tool_name": "search_docs", "args": {"query": user_message}, "result": reply}]
                if routed_to == "knowledge" else [],
            retrieved_chunk_ids=chunk_ids,
            latency_ms=42,
            created_at=now,
        ))
        await db.commit()

        return PipelineResult(content=reply, routed_to=routed_to, trace_id=trace_id)

    monkeypatch.setattr("app.srop.pipeline.run", _mock_run)
    monkeypatch.setattr("app.api.routes_chat.pipeline.run", _mock_run)
    return _mock_run
