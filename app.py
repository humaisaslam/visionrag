"""
app.py — Vision RAG
"""


import streamlit as st
import os
import json
import shutil
import requests
import chromadb
import re
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3') 
st.set_page_config(page_title="Vision RAG", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 900px; }
    section[data-testid="stSidebar"] { min-width: 270px; max-width: 270px; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* PDF badge */
    .pdf-badge {
        background: #1e293b;
        border-radius: 6px;
        padding: 0.35rem 0.75rem;
        font-size: 0.78rem;
        color: #94a3b8;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    /* References box */
    .ref-box {
        background: #0f172a;
        border-left: 3px solid #3b82f6;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 0.82rem;
        color: #93c5fd;
        margin-top: 0.25rem;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

from modules.embedder import embed_and_store, CHROMA_DB_PATH, COLLECTION_NAME, get_embedding
from modules.answerer import build_prompt, ANSWER_MODEL, SYSTEM_PROMPT, detect_question_type, strip_citations

PROFILES = {
    "Short":  {"top_n": 3, "threshold": 0.60, "fallback_threshold": 0.80, "num_predict": 256},
    "Normal": {"top_n": 6, "threshold": 0.65, "fallback_threshold": 0.85, "num_predict": -1},
}

for key, val in [
    ("chat_history", []),
    ("ingested_pdfs", {}),
    ("db_loaded", False),
    ("response_mode", "Normal"),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── ChromaDB ──
@st.cache_resource
def get_chroma_client():
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collection():
    try:
        return get_chroma_client().get_collection(name=COLLECTION_NAME)
    except Exception:
        return None


def get_db_count():
    col = get_collection()
    return col.count() if col else 0


def retrieve_chunks(query, top_n, threshold, fallback):
    col = get_collection()
    if not col or col.count() == 0:
        return []
    embedding = get_embedding(query)
    results = col.query(
        query_embeddings=[embedding],
        n_results=min(top_n, col.count()),
        include=["documents", "metadatas", "distances"]
    )
    chunks = [
        {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page_num": meta.get("page_num", "?"),
            "distance": round(dist, 4)
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
        if dist <= threshold
    ]
    if not chunks:
        chunks = [
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page_num": meta.get("page_num", "?"),
                "distance": round(dist, 4)
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
            if dist <= fallback
        ]
    return chunks


def wipe_knowledge_base():
    import gc, time
    try:
        try:
            get_chroma_client().delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        get_chroma_client.clear()
        gc.collect()
        time.sleep(1)
        if os.path.exists(CHROMA_DB_PATH):
            try:
                shutil.rmtree(CHROMA_DB_PATH)
            except Exception:
                for root, dirs, files in os.walk(CHROMA_DB_PATH, topdown=False):
                    for f in files:
                        try: os.remove(os.path.join(root, f))
                        except: pass
                    for d in dirs:
                        try: os.rmdir(os.path.join(root, d))
                        except: pass
                try: os.rmdir(CHROMA_DB_PATH)
                except: pass
        for k in ["db_loaded", "ingested_pdfs", "chat_history"]:
            st.session_state[k] = False if k == "db_loaded" else ([] if k == "chat_history" else {})
        return True
    except Exception as e:
        st.error(f"Wipe failed: {e}")
        return False


def stream_answer(query, chunks, mode, num_predict):
    prompt = build_prompt(query, chunks, mode)
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
            "num_ctx": 8192,
        }
    }
    try:
        with requests.post("http://localhost:11434/api/generate", json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                    if chunk.get("done", False):
                        break
    except requests.exceptions.ConnectionError:
        yield "❌ Cannot connect to Ollama. Run `ollama serve`."
    except Exception as e:
        yield f"❌ Error: {str(e)}"



def clean_response(text: str) -> str:
    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}', '', text)
    # Remove markdown headers
    text = re.sub(r'#{1,6}\s*', '', text)
    # Remove bullet points
    text = re.sub(r'(?m)^\s*[-\*]\s+', '', text)
    # Remove numbered lists
    text = re.sub(r'(?m)^\s*\d+\.\s+', '', text)
    # Remove URLs (http/https)
    text = re.sub(r'https?://\S+', '', text)
    # Remove "Retrieved from:" lines
    text = re.sub(r'(?i)retrieved from[:\s]*.*', '', text)
    # Remove leftover __ markdown underlines
    text = re.sub(r'__+', '', text)
    # Collapse blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def parse_references(raw: str) -> tuple[str, str]:
    """
    Splits model output into (answer_body, references_text).
    Strips the References block from the answer and returns it separately.
    """
    pattern = re.compile(
        r'(References\s*[:\-]?\s*\n?\s*Page\s*[:\-]\s*[\d,\s]+)',
        re.IGNORECASE
    )
    match = pattern.search(raw)
    if match:
        refs = match.group(1).strip()
        # Normalise to "References:\nPage: X, Y, Z"
        page_nums = re.search(r'[\d,\s]+$', refs)
        if page_nums:
            refs = f"References:\nPage: {page_nums.group().strip()}"
        answer = raw[:match.start()].strip()
        return answer, refs
    return raw.strip(), ""


def render_references(refs: str):
    if refs:
        st.markdown(f'<div class="ref-box">📄 {refs.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)


# ════════════════════════════════
# SIDEBAR
# ════════════════════════════════
with st.sidebar:
    st.markdown("## 🧭 Vision RAG")
    st.caption("Offline PDF Q&A · Powered by Ollama")
    st.divider()

    # ── KB status ──
    db_count = get_db_count()
    if db_count > 0:
        st.success(f"✅ Knowledge base ready\n\n{db_count} chunks indexed")
        st.session_state.db_loaded = True
    else:
        st.warning("⚠️ No knowledge base.\nIngest a PDF below.")
        st.session_state.db_loaded = False

    st.divider()

    # ── Response mode ──
    st.markdown("**Response length**")
    mode = st.radio(
        "mode", ["Short", "Normal"],
        index=["Short", "Normal"].index(st.session_state.response_mode),
        label_visibility="collapsed",
        horizontal=True
    )
    st.session_state.response_mode = mode
    st.caption("2 sentences" if mode == "Short" else "Full detailed answer")

    st.divider()

    # ── Ingest ──
    st.markdown("**Ingest PDF**")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    clear_existing = st.checkbox("Clear existing KB first", value=False)

    if st.button("⚡ Start Ingestion", use_container_width=True, type="primary"):
        if not uploaded_file:
            st.error("Please upload a PDF first.")
        else:
            from modules.pdf_extractor import extract_from_pdf
            from modules.image_captioner import caption_all_images
            from modules.chunker import chunk_all

            os.makedirs("data", exist_ok=True)
            pdf_path = os.path.join("data", uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if clear_existing:
                wipe_knowledge_base()

            progress = st.progress(0, text="Starting...")
            try:
                progress.progress(10, text="Extracting text & images...")
                extracted = extract_from_pdf(pdf_path)
                text, images = extracted["text"], extracted["image_paths"]

                progress.progress(35, text=f"Captioning {len(images)} images...")
                captions = caption_all_images(images) if images else []

                progress.progress(60, text="Chunking content...")
                chunks = chunk_all(text, captions)

                progress.progress(75, text=f"Embedding {len(chunks)} chunks...")
                embed_and_store(chunks)
                progress.progress(100, text="✅ Done!")

                st.session_state.ingested_pdfs[uploaded_file.name] = {
                    "chunks": len(chunks),
                    "images": images
                }
                st.session_state.db_loaded = True
                st.success(f"✅ Ingested — {len(chunks)} chunks stored.")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
                progress.empty()

    st.divider()

    # ── Actions ──
    st.markdown("**Actions**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with c2:
        if st.button("💣 Wipe DB", use_container_width=True):
            if wipe_knowledge_base():
                st.rerun()

    st.divider()
    st.caption(f"🤖 Model: `{ANSWER_MODEL}`")
    st.caption("Make sure `ollama serve` is running")


# ════════════════════════════════
# MAIN
# ════════════════════════════════
st.markdown("## 💬 Chat with your PDF")

# ── PDF index (collapsed) ──
if st.session_state.ingested_pdfs:
    for name, data in st.session_state.ingested_pdfs.items():
        with st.expander(f"📁 {name}  ·  {data['chunks']} chunks  ·  {len(data['images'])} images", expanded=False):
            if data["images"]:
                cols = st.columns(4)
                for idx, img_path in enumerate(data["images"]):
                    if os.path.exists(img_path):
                        with cols[idx % 4]:
                            st.image(img_path, use_container_width=True)
            else:
                st.caption("No images extracted.")
    st.divider()

# ── Chat history ──
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("references"):
            render_references(msg["references"])

# ── Chat input ──
query = st.chat_input("Ask a question about your PDF...")

if query:
    if not st.session_state.db_loaded:
        st.error("No knowledge base found. Please ingest a PDF first.")
    else:
        mode = st.session_state.response_mode
        profile = PROFILES[mode]

        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        references = ""

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                chunks = retrieve_chunks(
                    query,
                    top_n=profile["top_n"],
                    threshold=profile["threshold"],
                    fallback=profile["fallback_threshold"]
                )

            if not chunks:
                answer_body = "I couldn't find relevant content for this question. Try rephrasing."
                st.markdown(answer_body)
            else:
                source_types = set(c["source"] for c in chunks)
                source_label = " · ".join("Text" if s == "pdf_text" else "Image" for s in source_types)
                qtype = detect_question_type(query)
                st.caption(f"{qtype} · {source_label} · {len(chunks)} chunks · {mode}")

                placeholder = st.empty()
                raw = ""
                for token in stream_answer(query, chunks, mode, profile["num_predict"]):
                    raw += token
                    placeholder.markdown(strip_citations(raw) + "\u25cc")

                raw = strip_citations(raw)
                raw = clean_response(raw)
                answer_body, references = parse_references(raw)
                # Replace streaming placeholder with clean final answer
                placeholder.markdown(answer_body)
                if references:
                    render_references(references)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_body,
            "references": references
        })