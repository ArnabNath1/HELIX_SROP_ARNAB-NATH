"""
search_docs — RAG retrieval tool used by KnowledgeAgent.

Returns a formatted string that the LLM can read, with chunk IDs embedded
so the agent can cite them.  The pipeline parses chunk IDs from the function
response for trace recording.
"""
import structlog

import chromadb
from google import genai
from dataclasses import dataclass

from app.settings import settings

log = structlog.get_logger()


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict  # e.g. {"product_area": "security", "source": "deploy-keys.md"}


async def search_docs(query: str, k: int = 5) -> str:
    """
    Search Helix product documentation for chunks relevant to the query.
    Uses LLM reranking to pick the best results.
    """
    client = genai.Client(api_key=settings.google_api_key)

    # 1. Retrieval (Top 10)
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=query,
        config={
            "task_type": "retrieval_query",
        }
    )
    query_embedding = result.embeddings[0].values

    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = chroma_client.get_or_create_collection(name="helix_docs")

    # Get more than k for reranking
    results = collection.query(query_embeddings=[query_embedding], n_results=min(k * 2, 10))

    if not results["ids"] or not results["ids"][0]:
        return "No relevant documentation found for this query."

    chunks: list[DocChunk] = []
    for chunk_id, distance, doc, meta in zip(
        results["ids"][0], results["distances"][0], results["documents"][0], results["metadatas"][0]
    ):
        score = round(1.0 - distance, 4)
        chunks.append(DocChunk(chunk_id=chunk_id, score=score, content=doc, metadata=meta))

    # 2. Reranking (LLM-as-judge)
    # We ask the LLM to rank the chunks by ID
    candidate_text = "\n".join([f"ID: {c.chunk_id}\nContent: {c.content[:200]}..." for c in chunks])
    rerank_prompt = f"""
    You are an expert search reranker. Given the query '{query}' and the following candidates, 
    list the IDs of the top {k} most relevant chunks, best first. 
    Output ONLY the IDs separated by commas, nothing else.
    
    Candidates:
    {candidate_text}
    """
    
    try:
        rerank_res = client.models.generate_content(
            model=settings.adk_model,
            contents=rerank_prompt
        )
        best_ids = [cid.strip() for cid in rerank_res.text.split(",") if cid.strip()]
        # Filter chunks based on LLM's choice
        final_chunks = []
        for bid in best_ids:
            matching = [c for c in chunks if c.chunk_id == bid]
            if matching:
                final_chunks.append(matching[0])
        # Fallback if LLM failed
        if not final_chunks:
            final_chunks = chunks[:k]
        else:
            final_chunks = final_chunks[:k]
    except Exception as e:
        log.error("rerank_error", error=str(e))
        final_chunks = chunks[:k]

    log.info(
        "search_docs_results",
        query=query,
        k=k,
        num_results=len(final_chunks),
        top_score=final_chunks[0].score if final_chunks else None,
        reranked=True
    )

    parts: list[str] = []
    for chunk in final_chunks:
        source = chunk.metadata.get("source", "unknown")
        parts.append(
            f"[{chunk.chunk_id}] (score: {chunk.score:.2f}, source: {source})\n"
            f"{chunk.content}"
        )

    return "\n\n---\n\n".join(parts)
