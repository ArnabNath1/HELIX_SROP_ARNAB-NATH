"""
Support Agent — handles ticket creation and escalation.
"""
from google.adk.agents import LlmAgent
from app.agents.tools.support_tools import create_ticket
from app.settings import settings

SUPPORT_INSTRUCTION = """
You are the Helix Support Escalation specialist.
Your job is to create support tickets when a user's problem is too complex for the AI or if they explicitly ask for human help.

Rules:
1. ALWAYS ask for a summary of the issue if it's not clear.
2. ALWAYS confirm the priority with the user if they mention it.
3. Once you have the info, call create_ticket.
4. After creating the ticket, give the user the Ticket ID and tell them someone will reach out.
"""

support_agent = LlmAgent(
    name="support_agent",
    model=settings.adk_model,
    instruction=SUPPORT_INSTRUCTION,
    tools=[create_ticket],
)
