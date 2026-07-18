"""
server.py — PDF Vectorisation Retrieval API

Usage:
    uvicorn server:app --reload [--host 0.0.0.0] [--port 8000]

Endpoint:
    POST /query
    Body : { "query": str, "top_k": int }
    Response: [{ "chunk_text": str, "page_number": int | null, "score": float }]

NOTE: No LLM chain is used. Retrieval calls similarity_search_with_score()
      directly on the FAISS vector store.
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------

_vectorstore: Optional[FAISS] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None


# ---------------------------------------------------------------------------
# Lifespan — load vector store at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vectorstore, _embeddings

    print(f"[startup] Loading embedding model: {EMBEDDING_MODEL}")
    _embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    index_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
    if not os.path.isfile(index_path):
        print(
            f"[startup] WARNING: Vector store not found at '{VECTORSTORE_DIR}'.\n"
            "          Run ingestion first: python ingest.py --file document.pdf"
        )
    else:
        print(f"[startup] Loading FAISS index from '{VECTORSTORE_DIR}'")
        _vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            _embeddings,
            allow_dangerous_deserialization=True,
        )
        print("[startup] ✓ Vector store ready")

    yield  # Server runs here

    print("[shutdown] Cleaning up resources")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PDF Vectorisation Retrieval API",
    description=(
        "Retrieves semantically similar text chunks from an ingested PDF "
        "using FAISS similarity search. No LLM chain involved."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query string", min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, description="Number of results to return", ge=1, le=50)


class ChunkResult(BaseModel):
    chunk_text: str
    page_number: Optional[int]
    score: float


class QueryResponse(BaseModel):
    query: str
    top_k: int
    results: List[ChunkResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def root():
    """Returns API status and whether the vector store is loaded."""
    return {
        "status": "ok",
        "vectorstore_loaded": _vectorstore is not None,
        "vectorstore_dir": VECTORSTORE_DIR,
        "embedding_model": EMBEDDING_MODEL,
    }


@app.post("/query", response_model=QueryResponse, summary="Similarity search")
def query_vectorstore(request: QueryRequest):
    """
    Embed the incoming query using the same HuggingFace model used during
    ingestion, then call similarity_search_with_score() on the FAISS vector
    store and return the raw top-k chunks.

    No RetrievalQA, ConversationalRetrievalChain, or LLM is used here.
    """
    if _vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector store not loaded. "
                "Run ingestion first: python ingest.py --file document.pdf"
            ),
        )

    # Direct similarity search — no LLM chain
    results_with_scores = _vectorstore.similarity_search_with_score(
        request.query,
        k=request.top_k,
    )

    chunk_results: List[ChunkResult] = []
    for doc, score in results_with_scores:
        # Page numbers are stored as 0-indexed in metadata; convert to 1-indexed
        raw_page = doc.metadata.get("page")
        page_number = int(raw_page) + 1 if raw_page is not None else None

        chunk_results.append(
            ChunkResult(
                chunk_text=doc.page_content,
                page_number=page_number,
                score=float(score),
            )
        )

    return QueryResponse(
        query=request.query,
        top_k=request.top_k,
        results=chunk_results,
    )


# ---------------------------------------------------------------------------
# Run directly (development only — prefer uvicorn for production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
