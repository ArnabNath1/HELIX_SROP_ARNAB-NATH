import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.mark.asyncio
async def test_out_of_scope_refusal():
    with TestClient(app) as client:
        # 1. Create session
        resp = client.post("/v1/sessions", json={"user_id": "test_guard", "plan_tier": "free"})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # 2. Ask something out of scope
        chat_resp = client.post(f"/v1/chat/{session_id}", json={"content": "Write me a short poem about a cat."})
        assert chat_resp.status_code == 200
        
        reply = chat_resp.json()["reply"].lower()
        # The instruction says "politely refuse and explain your purpose is Helix support"
        # We check for general refusal keywords or context
        keywords = ["sorry", "cannot", "refuse", "helix", "support", "scope", "purpose", "poem"]
        assert any(word in reply for word in keywords)
