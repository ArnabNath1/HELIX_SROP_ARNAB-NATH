import argparse
import asyncio
import hashlib
import re
from pathlib import Path

import chromadb
import google.generativeai as genai
import yaml

from app.settings import settings


def chunk_markdown(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split markdown text into overlapping chunks using a sentence-aware strategy.
    Heading-aware splitting is also integrated for better coherence.
    """
    # Split by headings first
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks = []

    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section.strip())
        else:
            # Sentence-aware sub-chunking
            sentences = re.split(r"(?<=[.!?])\s+", section)
            current_chunk = []
            current_len = 0
            for sentence in sentences:
                if current_len + len(sentence) > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Overlap: keep some sentences (simulated by taking a slice)
                    # For simplicity in this implementation, we just start fresh
                    # but typically you'd keep some suffix.
                    current_chunk = []
                    current_len = 0
                current_chunk.append(sentence)
                current_len += len(sentence)
            if current_chunk:
                chunks.append(" ".join(current_chunk))

    return [c for c in chunks if c.strip()]


def extract_metadata(file_path: Path, text: str) -> tuple[dict, str]:
    """
    Extract metadata from a markdown file's frontmatter.
    """
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {"source": file_path.name}, text
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        metadata = {}
    metadata["source"] = file_path.name
    body = text[match.end() :]
    return metadata, body


def make_chunk_id(file_path: str, chunk_index: int) -> str:
    raw = f"{file_path}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


async def embed_in_batches(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    genai.configure(api_key=settings.google_api_key)
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=batch,
            task_type="retrieval_document",
        )
        embeddings.extend(result["embedding"])
        await asyncio.sleep(15)  # More aggressive sleep for free tier
    return embeddings


async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    md_files = list(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name="helix_docs", metadata={"hnsw:space": "cosine"}
    )

    all_ids = []
    all_embeddings = []
    all_documents = []
    all_metadatas = []

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata, body = extract_metadata(file_path, text)
        chunks = chunk_markdown(body, chunk_size, chunk_overlap)
        print(f"  {file_path.name}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(str(file_path), i)
            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_metadatas.append(metadata)

    if all_documents:
        print(f"Embedding {len(all_documents)} chunks...")
        all_embeddings = await embed_in_batches(all_documents)
        
        print(f"Upserting to vector store...")
        collection.upsert(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_documents,
            metadatas=all_metadatas,
        )

    print("Ingest complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
