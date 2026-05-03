"""
SROP Root Orchestrator — Google ADK agent.

Routes every user turn to KnowledgeAgent or AccountAgent via ADK's AgentTool.
The LLM decides which tool to call — no string-parsing of output.

Intent → sub-agent:
  knowledge:  "how do I X", "what is X", docs questions
  account:    "show my builds", "my account status", usage questions
  smalltalk:  greetings, thanks — root agent handles inline (no tool call)

State injection strategy: **Pattern 3** — build a fresh LlmAgent each turn
with user context (user_id, plan_tier, last_agent, turn_count, recent history)
baked into the instruction.  State persists in our SQLite DB; ADK is stateless.
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents.knowledge import knowledge_agent
from app.agents.account import ACCOUNT_INSTRUCTION
from app.agents.support import support_agent
from app.agents.tools.account_tools import get_recent_builds, get_account_status
from app.settings import settings
from app.srop.state import SessionState

ROOT_INSTRUCTION = """\
You are the Helix Support Concierge — a routing orchestrator.
Route every user message to exactly the right specialist tool.

Intent → tool mapping:
- Questions about HOW to do something, WHAT a feature is, or any product docs \
→ call knowledge_agent
- Questions about THEIR account, builds, status, usage, billing → call account_agent (you MUST include the user_id in your request to it).
- Requests to talk to a human, open a ticket, or complex issues that need escalation → call support_agent
- Greetings, thanks, polite closing → respond briefly inline (no tool call)
- OUT-OF-SCOPE: If the user asks for poems, jokes, stories, code unrelated to Helix, or anything not about product docs or their account → politely refuse and explain your purpose is Helix support.

Rules:
1. NEVER answer knowledge or account questions yourself — always delegate.
2. NEVER ask the user for their user_id or plan_tier; it is in the context below.
3. Always call a tool when intent matches — do not guess or hallucinate.
4. When a sub-agent provides an answer with citations (e.g., [chunk_123]), you MUST preserve those citations in your final response.
"""


def build_root_agent(state: SessionState, recent_history: str = "") -> LlmAgent:
    """
    Build the root orchestrator with the current session context injected.

    Creates a fresh LlmAgent each turn so the instruction reflects the current
    user_id, plan_tier, last_agent, and turn_count from persisted state.

    Args:
        state: Current session state loaded from the DB.
        recent_history: Last few turns formatted as a string (may be empty).
    """
    context_block = f"""\

=== CURRENT USER CONTEXT (do NOT ask the user for any of this) ===
user_id:      {state.user_id}
plan_tier:    {state.plan_tier}
turn_number:  {state.turn_count + 1}
last_agent:   {state.last_agent or 'none (first turn)'}

=== RECENT CONVERSATION HISTORY ===
{recent_history.strip() if recent_history else '(this is the first turn)'}
"""

    instruction = ROOT_INSTRUCTION + context_block

    # Pattern 3: Inject user context into the sub-agent's instruction too
    dynamic_account_agent = LlmAgent(
        name="account_agent",
        model=settings.adk_model,
        instruction=ACCOUNT_INSTRUCTION + f"\n\nCURRENT USER_ID: {state.user_id}\nPLAN_TIER: {state.plan_tier}",
        tools=[get_recent_builds, get_account_status],
    )

    return LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        instruction=instruction,
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=dynamic_account_agent),
            AgentTool(agent=support_agent),
        ],
    )
