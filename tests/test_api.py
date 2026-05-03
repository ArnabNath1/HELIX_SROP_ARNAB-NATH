"""
Integration tests — exercise the full HTTP + pipeline stack with LLM mocked.

The mock_pipeline fixture patches app.srop.pipeline.run so no real LLM calls
are made.  The DB interactions (session state, messages, traces) are fully
exercised against an in-memory SQLite instance.
"""
import pytest


@pytest.mark.asyncio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_001"})
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert body["user_id"] == "u_test_001"


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client):
    resp = await client.post(
        "/v1/chat/nonexistent-session-id", json={"content": "hello"}
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_trace_not_found_returns_404(client):
    resp = await client.get("/v1/traces/nonexistent-trace-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_pipeline):
    """
    Core integration test — two-turn conversation.

    Turn 1: knowledge question → asserts routed_to == "knowledge" and
            the trace has retrieved chunk IDs.

    Turn 2: follow-up asking for plan tier → asserts the agent still knows
            the plan_tier from persisted state (no re-asking).
    """
    # Create session with plan_tier = "pro"
    sess = await client.post(
        "/v1/sessions", json={"user_id": "u_test_002", "plan_tier": "pro"}
    )
    assert sess.status_code == 200
    session_id = sess.json()["session_id"]

    # ── Turn 1: knowledge query ───────────────────────────────────────────────
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
    )
    assert r1.status_code == 200
    r1_body = r1.json()
    assert r1_body["routed_to"] == "knowledge", (
        f"Expected 'knowledge' but got '{r1_body['routed_to']}'"
    )

    # Trace must have chunk IDs populated
    trace_id = r1_body["trace_id"]
    trace_resp = await client.get(f"/v1/traces/{trace_id}")
    assert trace_resp.status_code == 200
    trace_body = trace_resp.json()
    assert len(trace_body["retrieved_chunk_ids"]) > 0, (
        "Expected chunk IDs in trace but got none"
    )

    # ── Turn 2: follow-up — agent must know plan_tier from state ─────────────
    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "What is my plan tier?"},
    )
    assert r2.status_code == 200
    r2_body = r2.json()
    # The mock reads state.plan_tier and includes it in the reply
    assert "pro" in r2_body["reply"].lower(), (
        f"Expected plan tier 'pro' in reply, got: {r2_body['reply']}"
    )


@pytest.mark.asyncio
async def test_account_query_routes_to_account_agent(client, mock_pipeline):
    """Account questions should route to account_agent, not knowledge."""
    sess = await client.post(
        "/v1/sessions", json={"user_id": "u_test_003", "plan_tier": "free"}
    )
    session_id = sess.json()["session_id"]

    r = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "Show me my account status"},
    )
    assert r.status_code == 200
    assert r.json()["routed_to"] == "account", (
        f"Expected 'account' but got '{r.json()['routed_to']}'"
    )


@pytest.mark.asyncio
async def test_state_persists_turn_count(client, db, mock_pipeline):
    """Session state must increment turn_count across turns."""
    from sqlalchemy import select
    from app.db.models import Session as DbSession

    sess = await client.post(
        "/v1/sessions", json={"user_id": "u_test_004", "plan_tier": "enterprise"}
    )
    session_id = sess.json()["session_id"]

    await client.post(f"/v1/chat/{session_id}", json={"content": "Hello"})
    await client.post(f"/v1/chat/{session_id}", json={"content": "Tell me about builds"})

    # Refresh the in-memory DB session to see committed state
    await db.commit()  # flush any pending writes
    result = await db.execute(
        select(DbSession).where(DbSession.session_id == session_id)
    )
    # expire cached instance so we re-read from DB
    db_session = result.scalar_one_or_none()

    assert db_session is not None
    assert db_session.state["turn_count"] == 2, (
        f"Expected turn_count == 2, got {db_session.state['turn_count']}"
    )
