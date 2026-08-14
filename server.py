"""
server.py — PDF Vectorisation Retrieval API

Usage:
    uvicorn server:app --reload [--host 0.0.0.0] [--port 8000]

Endpoints:
    GET  /        — Health check & vectorstore metadata
    POST /query   — Similarity search endpoint
    POST /reload  — Hot-reload FAISS index from disk
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Enforce PyTorch backend for Hugging Face Transformers
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, HTTPException, status, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from utils import verify_index_checksum, calculate_index_checksum, l2_to_cosine_similarity

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("server")

# Global state
_vectorstore: Optional[FAISS] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None

# Security Dependency
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key_header_value: Optional[str] = Security(api_key_header)):
    if settings.API_KEY:
        if not api_key_header_value or api_key_header_value != settings.API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key"
            )
    return api_key_header_value


def load_vectorstore_from_disk() -> bool:
    """Helper to verify checksum and load FAISS vector store into memory."""
    global _vectorstore, _embeddings

    index_path = os.path.join(settings.VECTORSTORE_DIR, "index.faiss")
    if not os.path.isfile(index_path):
        logger.warning(f"Vector store index not found at '{settings.VECTORSTORE_DIR}'. Ingestion required.")
        # Do not wipe the existing vectorstore if it's already serving requests
        return False

    if not verify_index_checksum(settings.VECTORSTORE_DIR):
        logger.error(f"Checksum mismatch or corrupted index files in '{settings.VECTORSTORE_DIR}'.")
        # Keep the existing loaded vectorstore if one exists
        return False

    try:
        if _embeddings is None:
            logger.info(f"Initializing embedding model: '{settings.EMBEDDING_MODEL}' on device '{settings.DEVICE}'")
            _embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": settings.DEVICE},
                encode_kwargs={"normalize_embeddings": True},
            )

        logger.info(f"Loading FAISS index from '{settings.VECTORSTORE_DIR}'")
        new_store = FAISS.load_local(
            settings.VECTORSTORE_DIR,
            _embeddings,
            allow_dangerous_deserialization=True,
        )
        _vectorstore = new_store
        logger.info("✓ FAISS vector store successfully loaded into memory.")
        return True
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        # Keep the existing loaded vectorstore if one exists
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: initialize embeddings and load vector store at startup."""
    logger.info("Starting API server lifecycle...")
    load_vectorstore_from_disk()
    yield
    logger.info("Shutting down API server...")


app = FastAPI(
    title="PDF Vectorisation Retrieval API",
    description="Exposes semantic search endpoints over ingested PDF vectors.",
    version="1.1.0",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="The search query string",
        min_length=1,
        max_length=settings.MAX_QUERY_LENGTH
    )
    top_k: int = Field(
        settings.DEFAULT_TOP_K,
        description="Number of top results to return",
        ge=1,
        le=50
    )


class ChunkResult(BaseModel):
    chunk_text: str
    page_number: Optional[int] = None
    filename: Optional[str] = None
    score: float = Field(..., description="Cosine similarity score between 0.0 and 1.0")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    top_k: int
    results: List[ChunkResult]


class StatusResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    total_vectors: Optional[int] = None
    embedding_model: str
    device: str
    checksum: Optional[str] = None


# Endpoints
@app.get("/", response_model=StatusResponse, summary="Health & status check")
def health_check():
    """Returns API health status and vector store operational metadata."""
    total_vectors = None
    if _vectorstore is not None and hasattr(_vectorstore, "index"):
        total_vectors = _vectorstore.index.ntotal

    return StatusResponse(
        status="ok",
        vectorstore_loaded=_vectorstore is not None,
        total_vectors=total_vectors,
        embedding_model=settings.EMBEDDING_MODEL,
        device=settings.DEVICE,
        checksum=calculate_index_checksum(settings.VECTORSTORE_DIR),
    )


@app.post("/reload", summary="Hot-reload vector store from disk", dependencies=[Depends(verify_api_key)])
def reload_vectorstore():
    """Dynamically reloads the FAISS index from disk without restarting the server."""
    success = load_vectorstore_from_disk()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload vector store from disk. Check server logs."
        )
    return {"message": "Vector store reloaded successfully."}


@app.post("/query", response_model=QueryResponse, summary="Similarity search", dependencies=[Depends(verify_api_key)])
def query_vectorstore(request: QueryRequest):
    """
    Perform direct vector similarity search on the FAISS store.
    Converts raw L2 distance to normalized Cosine Similarity [0.0, 1.0].
    """
    if _vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not loaded. Run ingestion first: python ingest.py --file document.pdf"
        )

    try:
        results_with_scores = _vectorstore.similarity_search_with_score(
            request.query,
            k=request.top_k,
        )

        chunk_results: List[ChunkResult] = []
        for doc, raw_score in results_with_scores:
            # Extract metadata fields
            raw_page = doc.metadata.get("page")
            page_number = int(raw_page) if raw_page is not None else None
            filename = doc.metadata.get("filename", os.path.basename(doc.metadata.get("source", "")))

            # Normalize L2 distance to Cosine Similarity score [0, 1]
            cosine_score = l2_to_cosine_similarity(raw_score)

            chunk_results.append(
                ChunkResult(
                    chunk_text=doc.page_content,
                    page_number=page_number,
                    filename=filename if filename else None,
                    score=cosine_score,
                    metadata=doc.metadata,
                )
            )

        return QueryResponse(
            query=request.query,
            top_k=request.top_k,
            results=chunk_results,
        )
    except Exception as e:
        logger.error(f"Error executing search query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing similarity search: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
