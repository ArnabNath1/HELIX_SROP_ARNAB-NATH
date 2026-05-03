"""
Account tools — used by AccountAgent.

Returns JSON-serializable dicts (not dataclasses) so ADK can forward the
result to the LLM.  Mock data is used; the wiring is what matters.
"""
from datetime import datetime, timezone


async def get_recent_builds(user_id: str, limit: int = 5) -> dict:
    """
    Return the most recent CI/CD builds for a user, newest first.

    Args:
        user_id: The Helix user ID to look up builds for.
        limit: Maximum number of builds to return (default 5).

    Returns:
        A dict with 'user_id' and 'builds' (list of build summaries).
    """
    now = datetime.now(timezone.utc).isoformat()
    statuses = ["passed", "failed", "passed", "cancelled", "passed"]
    builds = [
        {
            "build_id": f"build_{user_id}_{i:03d}",
            "pipeline": "main-deployment",
            "status": statuses[i % len(statuses)],
            "branch": "main" if i % 3 != 1 else "feature/my-branch",
            "started_at": now,
            "duration_seconds": 120 + i * 15,
        }
        for i in range(limit)
    ]
    return {"user_id": user_id, "builds": builds}


async def get_account_status(user_id: str) -> dict:
    """
    Return current account status including plan tier and resource usage.

    Args:
        user_id: The Helix user ID to look up.

    Returns:
        A dict with plan, concurrent build usage, and storage usage.
    """
    return {
        "user_id": user_id,
        "plan_tier": "pro",
        "concurrent_builds_used": 2,
        "concurrent_builds_limit": 10,
        "storage_used_gb": 15.4,
        "storage_limit_gb": 100.0,
        "active_runners": 3,
    }
