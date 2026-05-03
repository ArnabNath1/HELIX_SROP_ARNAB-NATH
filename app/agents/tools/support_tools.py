import uuid
from datetime import datetime
from contextvars import ContextVar
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Ticket

# ContextVar to hold the database session for the current request
db_session_var: ContextVar[AsyncSession] = ContextVar("db_session")

async def create_ticket(user_id: str, summary: str, priority: str = "med") -> str:
    """
    Escalate a complex issue by creating a support ticket.
    
    Args:
        user_id: The Helix user ID.
        summary: A clear summary of the issue to escalate.
        priority: Priority of the ticket ('low', 'med', 'high').
        
    Returns:
        The generated ticket ID (e.g., 'TICKET-12345').
    """
    db = db_session_var.get()
    ticket_id = f"TICKET-{uuid.uuid4().hex[:8].upper()}"
    
    new_ticket = Ticket(
        ticket_id=ticket_id,
        user_id=user_id,
        summary=summary,
        priority=priority,
        created_at=datetime.utcnow()
    )
    
    db.add(new_ticket)
    # We don't commit here; the pipeline will commit the whole session at the end
    # or we can commit here if we want immediate persistence.
    await db.flush() 
    
    return ticket_id
