"""
embedder.py
-----------
Optimized for speed:
1. Parallel embedding with ThreadPoolExecutor
2. Single shared ChromaDB client (no reconnecting)
3. Larger batch sizes for ChromaDB writes
4. Retry logic for failed embeddings
"""

import requests
import chromadb
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "qwen3-embedding:4b"
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "vision_rag"
BATCH_SIZE = 50        # larger = fewer ChromaDB write calls
EMBED_WORKERS = 8     # parallel embedding threads
MAX_RETRIES = 2        # retry failed embeddings


# --- Single shared client (avoids reconnecting on every call) ---
_client = None
_client_lock = Lock()

def _get_client() -> chromadb.PersistentClient:
    global _client
    with _client_lock:
        if _client is None:
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def get_embedding(text: str) -> list:
    """
    Embeds a single text string via Ollama with retry logic.
    """
    payload = {"model": EMBED_MODEL, "prompt": text}

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Embedding failed after {MAX_RETRIES} retries: {e}")


def _embed_chunk(item: tuple) -> tuple:
    """
    Embeds a single chunk. Returns (index, embedding).
    Designed for parallel execution.
    """
    index, text = item
    embedding = get_embedding(text)
    return index, embedding


def get_embeddings_parallel(texts: list) -> list:
    """
    Embeds all texts in parallel using ThreadPoolExecutor.
    Returns embeddings in the same order as input texts.
    """
    embeddings = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
        futures = {
            executor.submit(_embed_chunk, (i, text)): i
            for i, text in enumerate(texts)
        }
        for future in as_completed(futures):
            index, embedding = future.result()
            embeddings[index] = embedding

    return embeddings


def get_or_create_collection() -> chromadb.Collection:
    """Returns the ChromaDB collection, creating it if needed."""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def embed_and_store(chunks: list):
    """
    Embeds all chunks in parallel and stores in ChromaDB in large batches.
    """
    if not chunks:
        print("[embedder] No chunks to embed.")
        return

    collection = get_or_create_collection()
    total = len(chunks)

    print(f"[embedder] Embedding {total} chunks ({EMBED_WORKERS} parallel workers)...")

    # Extract all texts
    texts = [c["text"] for c in chunks]

    # Embed all in parallel
    embeddings = get_embeddings_parallel(texts)

    # Build metadata list
    metadatas = []
    for c in chunks:
        meta = dict(c["metadata"])
        meta["source"] = c["source"]
        metadatas.append(meta)

    ids = [f"chunk_{i}" for i in range(total)]

    # Store in ChromaDB in large batches
    print(f"[embedder] Storing {total} chunks in ChromaDB...")
    for batch_start in range(0, total, BATCH_SIZE):
        end = min(batch_start + BATCH_SIZE, total)
        collection.add(
            ids=ids[batch_start:end],
            embeddings=embeddings[batch_start:end],
            documents=texts[batch_start:end],
            metadatas=metadatas[batch_start:end]
        )
        print(f"[embedder] Stored {end}/{total} chunks...")

    print(f"[embedder] Done. {total} chunks stored in '{CHROMA_DB_PATH}'")


def load_collection() -> chromadb.Collection:
    """Loads existing ChromaDB collection using the shared client."""
    client = _get_client()
    return client.get_collection(name=COLLECTION_NAME)