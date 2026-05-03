"""
Unit tests for RAG retrieval.

test_search_docs_returns_results_with_chunk_ids requires a seeded vector store.
Run `uv run python -m app.rag.ingest --path docs/` first.

test_chunker_produces_non_empty_chunks runs without any external services.
"""
import pytest


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids():
    """
    search_docs must return a non-empty string containing chunk IDs and scores.

    Requires: vector store seeded via `python -m app.rag.ingest --path docs/`.
    """
    import re
    from app.agents.tools.search_docs import search_docs

    result = await search_docs("how to rotate a deploy key", k=3)

    assert isinstance(result, str), "search_docs must return a string"
    assert len(result) > 0, "search_docs returned an empty string"
    assert "No relevant documentation found" not in result, (
        "Vector store appears empty — run ingest first"
    )

    # Each result block must contain a chunk ID in [chunk_<hex>] format
    chunk_ids = re.findall(r"\[(chunk_[a-f0-9]+)\]", result)
    assert len(chunk_ids) > 0, f"No chunk IDs found in output: {result[:200]}"

    # Scores must appear and be parseable as floats in [0, 1]
    scores = re.findall(r"score: ([0-9.]+)", result)
    assert len(scores) > 0, "No scores found in search_docs output"
    for s in scores:
        score_val = float(s)
        assert 0.0 <= score_val <= 1.0, f"Score {score_val} out of [0, 1] range"


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings for typical markdown input."""
    from app.rag.ingest import chunk_markdown

    text = (
        "# Deploy Keys\n\n"
        "Deploy keys are SSH keys that grant read or read/write access to a single repository.\n\n"
        "## Rotating a Deploy Key\n\n"
        "To rotate a deploy key, navigate to Settings > Deploy Keys and click Rotate.\n"
        "You will be prompted to confirm the rotation.  The old key is immediately revoked.\n\n"
        "## Creating a Deploy Key\n\n"
        "Go to Settings > Deploy Keys > Add Key.  Paste your public SSH key and click Save.\n"
    )

    chunks = chunk_markdown(text, chunk_size=200, overlap=32)

    assert len(chunks) > 0, "Chunker produced no chunks"
    assert all(c.strip() for c in chunks), "Chunker produced at least one empty chunk"
    # All chunk content should be non-trivial
    assert all(len(c) > 10 for c in chunks), "Some chunks are suspiciously short"
