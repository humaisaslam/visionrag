"""
retriever.py
------------
Responsible for:
1. Embedding the user query
2. Searching ChromaDB for the most relevant chunks
3. Returning chunks with text, source, page_num, and distance
"""

from modules.embedder import get_embedding, load_collection

TOP_N_RESULTS = 5


def retrieve(query: str, top_n: int = TOP_N_RESULTS) -> list:
    """
    Embeds the query and retrieves the most relevant chunks from ChromaDB.

    Returns:
        List of dicts with: text, source, page_num, metadata, distance
    """
    print(f"[retriever] Embedding query: '{query}'")
    query_embedding = get_embedding(query)
    collection = load_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_n, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved_chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page_num": meta.get("page_num", "?"),
            "metadata": meta,
            "distance": round(dist, 4)
        })

    print(f"[retriever] Retrieved {len(retrieved_chunks)} chunks.")
    for i, chunk in enumerate(retrieved_chunks):
        preview = chunk["text"][:80].replace("\n", " ")
        print(f"  [{i+1}] (page={chunk['page_num']}, src={chunk['source']}, dist={chunk['distance']}) {preview}...")

    return retrieved_chunks


def format_context(chunks: list) -> str:
    """
    Formats chunks into a labeled context string with page numbers.
    Used by stream_answer() in app.py.
    """
    parts = []
    for i, chunk in enumerate(chunks):
        page = chunk.get("page_num", "?")
        label = "Text" if chunk["source"] == "pdf_text" else "Image/Diagram"
        parts.append(f"[Page {page} — {label}]\n{chunk['text']}")
    return "\n\n".join(parts)