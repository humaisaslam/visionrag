"""
answerer.py
"""

import requests
import json
import re


OLLAMA_URL = "http://localhost:11434/api/generate"
ANSWER_MODEL = "llava-phi3"


SYSTEM_PROMPT = """You are a helpful document assistant. Answer questions using only the provided document context.

Rules:
- Only use information from the context
- Be clear and direct
- No filler phrases like "Great question!" or "I hope this helps!"
- Never use inline citation markers like [1], [^1^], [^2^] — these are forbidden
- If the answer is not in the document, say so briefly
- At the very end of your answer, always add a references block in this exact format:

References:
Page: 3, 7, 12
"""


def detect_question_type(query: str) -> str:
    q = query.lower().strip()
    if any(w in q for w in ["summarize", "summary", "overview"]):
        return "SUMMARY"
    elif any(w in q for w in ["difference", "compare", "vs", "versus"]):
        return "COMPARISON"
    elif any(w in q for w in ["how many", "how much", "percentage", "total"]):
        return "DATA"
    elif any(w in q for w in ["list", "what are", "types of", "examples of"]):
        return "LIST"
    elif any(w in q for w in ["how does", "how do", "why", "explain"]):
        return "EXPLANATION"
    elif any(w in q for w in ["diagram", "figure", "chart", "graph", "image", "table"]):
        return "VISUAL"
    else:
        return "FACTUAL"


def strip_citations(text: str) -> str:
    # Remove all citation variants: [^2^], [^2], [2], ^2^, ^2
    text = re.sub(r'\[\^?\d+\^?\]', '', text)
    text = re.sub(r'\^\d+\^?', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[\^?\]', '', text)
    text = re.sub(r' +', ' ', text).strip()
    return text


def extract_snippet(text: str, query: str, max_chars: int = 250) -> str:
    """
    Finds the most relevant sentence in a chunk for display as a source snippet.
    Falls back to the first max_chars if no keyword match found.
    """
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))

    for kw in keywords:
        for sentence in sentences:
            if kw in sentence.lower():
                snippet = sentence.strip()
                return snippet[:max_chars] + ("..." if len(snippet) > max_chars else "")

    return text[:max_chars].strip() + ("..." if len(text) > max_chars else "")


def build_prompt(query: str, chunks: list[dict], mode: str = "Normal") -> str:
    """
    Builds the prompt with clearly labeled page context blocks.
    chunks: list of dicts with keys 'text', 'page_num' (and optionally 'caption')
    """
    if mode == "Short":
        length = "Answer in 2 sentences maximum. Be direct and concise."
    else:
        length = "Give a clear, complete answer. Use bullet points or steps where helpful."

    context_parts = []
    for chunk in chunks:
        page = chunk.get("page_num", "?")
        text = chunk.get("text", chunk.get("caption", ""))
        context_parts.append(f"[Page {page}]\n{text}")

    context = "\n\n".join(context_parts)

    return f"""Document context:
---
{context}
---

Question: {query}

{length}
IMPORTANT: Do NOT use inline citation markers like [1], [^1^], [^2^] anywhere in your answer.
At the end of your answer write a references block exactly like this:

References:
Page: <comma separated page numbers used>

Answer:"""


def answer(query: str, chunks: list[dict], mode: str = "Normal") -> dict:
    """
    Answers a query using provided chunks (each with 'text'/'caption' and 'page_num').

    Returns:
        {
            "answer": str,
            "sources": [{"page_num": int, "snippet": str}]
        }

    Note: also accepts a plain context string for backward compatibility.
    """
    # Backward compatibility: if a plain string is passed, wrap it
    if isinstance(chunks, str):
        chunks = [{"text": chunks, "page_num": "?"}]

    prompt = build_prompt(query, chunks, mode)
    qtype = detect_question_type(query)
    num_predict = 150 if mode == "Short" else 1024

    payload = {
        "model": ANSWER_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "20m",
        "options": {
            "temperature": 0.2,
            "repeat_penalty": 1.1,
            "num_predict": num_predict,
            "num_ctx": 4096,
        }
    }

    print(f"\n[answerer] [{qtype}] [{mode}] {ANSWER_MODEL}\n" + "-" * 50)
    full_response = ""

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    print(token, end="", flush=True)
                    full_response += token
                    if chunk.get("done", False):
                        break
        full_response = strip_citations(full_response)
    except Exception as e:
        print(f"\n[answerer] ERROR: {e}")

    print("\n" + "-" * 50)

    # Build source list with page number + relevant snippet
    sources = []
    seen_pages = set()
    for c in chunks:
        page = c.get("page_num", "?")
        if page in seen_pages:
            continue
        seen_pages.add(page)
        text = c.get("text", c.get("caption", ""))
        sources.append({
            "page_num": page,
            "snippet": extract_snippet(text, query)
        })

    return {
        "answer": full_response,
        "sources": sources
    }