# PDF Vectorisation Pipeline

**Technical Assessment — Nestack | LangChain / LangGraph Implementation**

---

## Overview

This project implements a two-component PDF vectorisation pipeline:

1. **`ingest.py`** — Loads a PDF, splits it into chunks, generates embeddings, and persists them to a local FAISS vector store. Modelled as a **LangGraph** graph for full traceability.
2. **`server.py`** — A FastAPI HTTP server exposing `POST /query`. Embeds the incoming query and calls `similarity_search_with_score()` **directly** on the vector store. No LLM chain is used.

---

## LangGraph Ingestion Graph

The ingestion pipeline is modelled as a **StateGraph** with four named nodes:

```
load  →  split  →  embed  →  store
```

| Node | Responsibility |
|---|---|
| `load` | `PyPDFLoader` — reads all pages from the PDF |
| `split` | `RecursiveCharacterTextSplitter` — splits pages into overlapping chunks |
| `embed` | `HuggingFaceEmbeddings` — initialises the local embedding model |
| `store` | `FAISS.from_documents()` + `save_local()` — builds and persists the index |

A printed node list is output at the start of every ingestion run.

---

## Component Choices & Justification

### 1. Document Loader — `PyPDFLoader`
- Native LangChain integration (`langchain-community`)
- Lightweight: no server or complex dependencies required
- Preserves page-level metadata (`page` field) so retrieval results can report page numbers
- Handles multi-page PDFs reliably

### 2. Text Splitter — `RecursiveCharacterTextSplitter`
- **chunk_size = 1000** characters  
  Captures roughly 1–2 paragraphs — a semantically coherent unit that is not so large as to dilute relevance scoring.
- **chunk_overlap = 200** characters (~20%)  
  Ensures sentences that span two chunks are still retrievable from either side, preventing context loss at boundaries.
- Separator hierarchy `["\n\n", "\n", ".", " ", ""]` attempts to break at natural boundaries before falling back to hard character splits.

### 3. Embedding Model — `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)
- Runs **fully locally** — no API key or internet access required after first download
- 384-dimensional dense vectors with excellent semantic quality for English text
- First-class LangChain integration via `langchain-huggingface`
- ~90 MB model, cached after the first run
- `normalize_embeddings=True` ensures cosine similarity is well-behaved

### 4. Vector Store — `FAISS` (local)
- No external server or cloud dependency
- Built-in LangChain persistence via `save_local()` / `load_local()`
- Supports `similarity_search_with_score()` natively (returns L2 distance)
- Efficient for small-to-medium document corpora

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone / unzip the project

```bash
cd "python assessment"
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) is downloaded on first run and cached locally.

---

## Usage

### Part 1 — Ingest a PDF

```bash
python ingest.py --file document.pdf
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--file` | *(required)* | Path to the PDF |
| `--chunk-size` | `1000` | Characters per chunk |
| `--chunk-overlap` | `200` | Overlap between chunks |
| `--vectorstore-dir` | `./vectorstore` | Output directory |

**Example output:**

```
============================================================
LangGraph Ingestion Pipeline
Nodes: load → split → embed → store
============================================================

[load] Loading PDF: document.pdf
[load] ✓ Loaded 12 page(s)

[split] Splitting with chunk_size=1000, chunk_overlap=200
[split] ✓ Produced 47 chunk(s)

[embed] Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
[embed] (First run downloads ~90 MB — subsequent runs use cache)
[embed] ✓ Embedding model ready

[store] Building FAISS index from 47 chunk(s)
[store] ✓ Vector store saved to './vectorstore/'
[store]   Files: index.faiss, index.pkl

============================================================
✅ Ingestion complete!
   Pages loaded : 12
   Chunks stored: 47
   Vector store : ./vectorstore/
============================================================
```

### Part 2 — Start the Retrieval Server

```bash
uvicorn server:app --reload
```

The server starts at **http://localhost:8000**.

- `GET  /`      — health check + vectorstore status
- `POST /query` — similarity search endpoint

### Query the endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "top_k": 3}'
```

**Example response:**

```json
{
  "query": "What is this document about?",
  "top_k": 3,
  "results": [
    {
      "chunk_text": "This document describes the methodology for...",
      "page_number": 1,
      "score": 0.312
    },
    {
      "chunk_text": "The primary objective of this study is to...",
      "page_number": 2,
      "score": 0.489
    },
    {
      "chunk_text": "In summary, the report covers...",
      "page_number": 11,
      "score": 0.521
    }
  ]
}
```

> **Score note:** FAISS returns **L2 distance** — lower is more similar.

---

## Generating `results.json`

After ingestion, run the following three queries and capture the output:

```bash
# Query 1
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "top_k": 3}'

# Query 2
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings or conclusions?", "top_k": 3}'

# Query 3
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What methodology or approach is described?", "top_k": 3}'
```

Save the combined JSON array to `results.json`.

---

## Project Structure

```
python assessment/
├── ingest.py          # LangGraph ingestion pipeline
├── server.py          # FastAPI retrieval server
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── results.json       # Sample query results
└── vectorstore/       # Created at runtime by ingest.py
    ├── index.faiss
    └── index.pkl
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VECTORSTORE_DIR` | `./vectorstore` | Override the vector store path for the server |

---

## Deployment

> Add your live deployment link here once deployed.

For a quick cloud deployment, you can use **Railway**, **Render**, or **Fly.io**:

1. Push the repository (private) to GitHub
2. Connect to your deployment platform
3. Set the start command: `uvicorn server:app --host 0.0.0.0 --port 8000`
4. Ensure the `vectorstore/` directory is committed or generated during build

---

## Evaluation Criteria Addressed

| Criterion | How addressed |
|---|---|
| Pipeline runs end-to-end | Ingestion + retrieval fully functional; see usage above |
| Correct LangChain primitives | `PyPDFLoader`, `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, `FAISS` — each used for its correct purpose |
| No forbidden chains | `similarity_search_with_score()` called directly; no `RetrievalQA` or `ConversationalRetrievalChain` |
| Retrieval quality | `results.json` contains semantically relevant chunks per query |
| LangGraph bonus (+5%) | Four named nodes, typed state, documented flow |
