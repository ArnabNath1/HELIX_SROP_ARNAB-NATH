from google.adk.agents import LlmAgent
from app.agents.tools.search_docs import search_docs
from app.settings import settings

KNOWLEDGE_INSTRUCTION = """
You are a Helix Product Knowledge Specialist.
Your goal is to answer technical questions about the Helix platform using the provided documentation chunks.

Guidelines:
1. Use ONLY the provided context chunks to answer.
2. If the answer is not in the context, say "I don't have documentation on that." Do not hallucinate.
3. ALWAYS cite the chunk_id for every fact you state (e.g., "According to [chunk_abc123]...").
4. If multiple chunks are relevant, synthesize them into a coherent answer.
"""

knowledge_agent = LlmAgent(
    name="knowledge_agent",
    model=settings.adk_model,
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[search_docs],
)
