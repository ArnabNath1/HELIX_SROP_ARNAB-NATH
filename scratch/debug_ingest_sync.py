from pathlib import Path
import os
from app.settings import settings
import google.generativeai as genai
import chromadb

def debug():
    print("Starting debug...")
    docs_path = Path("docs")
    md_files = list(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} files")
    
    print(f"Chroma persist dir: {settings.chroma_persist_dir}")
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    print("Connected to Chroma")
    
    collection = client.get_or_create_collection(
        name="helix_docs", metadata={"hnsw:space": "cosine"}
    )
    print("Got collection")
    
    print(f"API Key: {settings.google_api_key[:5]}...")
    genai.configure(api_key=settings.google_api_key)
    print("Configured GenAI")
    
    print("Embedding a test string...")
    # genai.embed_content is sync? 
    # Actually google-generativeai is mostly sync unless using async methods.
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content="test",
        task_type="retrieval_query",
    )
    print("Embedding result obtained")
    print(f"Embedding length: {len(result['embedding'])}")

if __name__ == "__main__":
    debug()
