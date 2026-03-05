# Vision RAG — Offline Multimodal PDF Q&A

## What is Vision RAG?

Vision RAG is a fully **offline** Retrieval-Augmented Generation system that lets you chat with your PDF documents — including charts, diagrams, tables, and images inside them.

Unlike cloud-based document Q&A tools, **your data never leaves your machine.** No OpenAI API keys. No internet required after setup.

---

##  Project Structure

```
vision-rag/
│
├── app.py                  # Streamlit UI — main entry point
│
└── modules/
    ├── pdf_extractor.py    # PDF text + image extraction (PyMuPDF)
    ├── image_captioner.py  # Vision captioning via LLaVA-Phi3 (parallel)
    ├── chunker.py          # Text splitting + caption chunking with page metadata
    ├── embedder.py         # Parallel embedding + ChromaDB storage
    ├── retriever.py        # Query embedding + semantic search
    └── answerer.py         # Prompt builder + LLM response streamer
```

---


### Python Dependencies

```bash
pip install streamlit chromadb pymupdf requests
```

### Ollama Models

Pull the required models before first use:

```bash
# LLM for answering + vision captioning
ollama pull llava-phi3

# Embedding model
ollama pull qwen3-embedding:4b
```

---

##  Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/vision-rag.git
cd vision-rag
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start Ollama**
```bash
ollama serve
```

**4. Pull the models**
```bash
ollama pull llava-phi3
ollama pull qwen3-embedding:4b
```

**5. Run the app**
```bash
streamlit run app.py
```

**6. Open your browser** at `http://localhost:8501`

##  Configuration

Key settings are at the top of each module:

| File | Setting | Default | Description |
|------|---------|---------|-------------|
| `embedder.py` | `EMBED_MODEL` | `qwen3-embedding:4b` | Ollama embedding model |
| `embedder.py` | `EMBED_WORKERS` | `8` | Parallel embedding threads |
| `embedder.py` | `BATCH_SIZE` | `50` | ChromaDB write batch size |
| `answerer.py` | `ANSWER_MODEL` | `llava-phi3` | LLM for generating answers |
| `chunker.py` | `CHUNK_SIZE` | `800` | Characters per text chunk |
| `chunker.py` | `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `image_captioner.py` | `MIN_IMAGE_SIZE_BYTES` | `10,000` | Skip images smaller than this (logos/icons) |
| `image_captioner.py` | `MAX_WORKERS` | `3` | Parallel captioning threads |

---

##  Module Overview

### `pdf_extractor.py`
Uses **PyMuPDF (fitz)** to extract page-by-page text and all embedded images. Images are saved to `data/images/` with filenames like `page3_img2.jpeg` for page tracking.

### `image_captioner.py`
Sends each image to **LLaVA-Phi3** via Ollama's `/api/chat` endpoint with a detailed structured prompt. Captions include image type, content, labels, key takeaway, and figure numbers. Tiny images (< 10KB) are skipped automatically.

### `chunker.py`
Splits extracted text into overlapping chunks with page number metadata. Handles three cases: form-feed page breaks, `Page N` markers in text, and no page info. Image captions are treated as individual chunks.

### `embedder.py`
Embeds all chunks in parallel using **ThreadPoolExecutor** (8 workers by default). Stores embeddings in **ChromaDB** with cosine similarity space. Uses a single shared client for efficiency.

### `retriever.py`
Embeds the user query and searches ChromaDB for the nearest chunks. Returns text, source type, page number, and similarity distance.

### `answerer.py`
Builds a structured prompt with labeled page context blocks and streams the LLM response token-by-token. Detects question type (Summary, Comparison, Data, List, Explanation, Visual, Factual) to adapt the prompt style.

---

## Author

**Muhammad Humais Aslam**
AI Engineer · Automation Builder
[LinkedIn](https://linkedin.com/in/humaisaslam) · [GitHub](https://github.com/yourusername) · humaisaslam@gmail.com




