"""
ingest.py — PDF Vectorisation Pipeline (LangGraph implementation)

Usage:
    python ingest.py [--file document.pdf | --dir ./pdf_dir] [--chunk-size 1000] [--chunk-overlap 200] [--vectorstore-dir ./vectorstore]

LangGraph node flow:
    load -> validate -> split -> embed_and_store
"""

import argparse
import glob
import hashlib
import logging
import os
import sqlite3
import sys
import time
from typing import List, TypedDict, Dict, Any

# Enforce PyTorch backend for Hugging Face Transformers
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_community.document_loaders import PyMuPDFLoader
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END

from config import settings
from utils import clean_text, save_index_checksum

# Set up structured logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# State schema for the LangGraph pipeline (Serializable primitives only)
# ---------------------------------------------------------------------------

class SerializedDocument(TypedDict):
    page_content: str
    metadata: Dict[str, Any]


class PipelineState(TypedDict):
    """Shared serializable state passed between LangGraph nodes."""
    file_paths: List[str]
    chunk_size: int
    chunk_overlap: int
    vectorstore_dir: str
    embedding_model_name: str
    documents: List[SerializedDocument]
    chunks: List[SerializedDocument]
    status: str
    checksum: str
    error: str
    overwrite: bool


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# SQLite Document Registry Helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str):
    """Ensure the registry directory and database table exist."""
    db_dir = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            file_hash TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            ingested_at INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Node 1 — load
# ---------------------------------------------------------------------------

def load_node(state: PipelineState) -> PipelineState:
    """Load PDF files using PyMuPDFLoader or PyPDFLoader and extract pages."""
    file_paths = state["file_paths"]
    logger.info(f"[load] Starting PDF loading for {len(file_paths)} file(s)...")

    db_path = settings.DATABASE_PATH
    init_db(db_path)

    # Read already ingested file hashes from SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT file_hash FROM ingested_files")
    existing_hashes = {row[0] for row in cursor.fetchall()}
    conn.close()

    overwrite = state.get("overwrite", False)
    loaded_docs: List[SerializedDocument] = []

    for file_path in file_paths:
        if not os.path.isfile(file_path):
            logger.warning(f"[load] File not found: {file_path}")
            continue

        # Compute SHA-256 hash of the file for metadata tracing
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while c := f.read(8192):
                hasher.update(c)
        file_hash = hasher.hexdigest()[:12]
        filename = os.path.basename(file_path)

        if file_hash in existing_hashes and not overwrite:
            logger.info(f"[load] Skipped file (already ingested): '{filename}'")
            continue

        try:
            if HAS_PYMUPDF:
                loader = PyMuPDFLoader(file_path)
            else:
                loader = PyPDFLoader(file_path)

            raw_documents = loader.load()
            logger.info(f"[load] Loaded {len(raw_documents)} page(s) from '{filename}'")

            for doc in raw_documents:
                # Standardize page metadata
                raw_page = doc.metadata.get("page", doc.metadata.get("page_number", 0))
                try:
                    page_num = int(raw_page) + 1  # Convert 0-index to 1-index if needed
                except (ValueError, TypeError):
                    page_num = 1

                metadata = {
                    "source": file_path,
                    "filename": filename,
                    "file_hash": file_hash,
                    "page": page_num,
                    "total_pages": len(raw_documents),
                }

                cleaned_content = clean_text(doc.page_content)
                if cleaned_content:
                    loaded_docs.append({
                        "page_content": cleaned_content,
                        "metadata": metadata
                    })
        except Exception as e:
            logger.error(f"[load] Error reading '{file_path}': {e}")

    logger.info(f"[load] ✓ Total valid pages loaded: {len(loaded_docs)}")
    return {**state, "documents": loaded_docs, "status": "loaded"}


# ---------------------------------------------------------------------------
# Node 2 — validate
# ---------------------------------------------------------------------------

def validate_node(state: PipelineState) -> PipelineState:
    """Validate that extracted documents contain readable text."""
    docs = state["documents"]
    if not docs:
        logger.error("[validate] ❌ No valid text documents could be extracted.")
        return {**state, "status": "failed", "error": "No documents extracted"}

    logger.info(f"[validate] ✓ Validated {len(docs)} document pages.")
    return {**state, "status": "validated"}


def should_continue_after_validation(state: PipelineState) -> str:
    """Conditional edge router: proceed to split if validated, else terminate."""
    if state["status"] == "failed":
        return "end"
    return "split"


# ---------------------------------------------------------------------------
# Node 3 — split
# ---------------------------------------------------------------------------

def split_node(state: PipelineState) -> PipelineState:
    """Split page documents into chunks with structure-aware splitter."""
    chunk_size = state["chunk_size"]
    chunk_overlap = state["chunk_overlap"]
    logger.info(f"[split] Splitting with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    langchain_docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in state["documents"]
    ]

    raw_chunks = splitter.split_documents(langchain_docs)

    serialized_chunks: List[SerializedDocument] = []
    timestamp = int(time.time())

    for idx, chunk in enumerate(raw_chunks):
        meta = dict(chunk.metadata)
        meta["chunk_id"] = idx
        meta["chunk_size"] = len(chunk.page_content)
        meta["created_at"] = timestamp

        serialized_chunks.append({
            "page_content": chunk.page_content,
            "metadata": meta
        })

    logger.info(f"[split] ✓ Produced {len(serialized_chunks)} chunk(s)")
    return {**state, "chunks": serialized_chunks, "status": "split"}


# ---------------------------------------------------------------------------
# Node 4 — embed_and_store
# ---------------------------------------------------------------------------

def embed_and_store_node(state: PipelineState) -> PipelineState:
    """Embed text chunks using HuggingFaceEmbeddings and persist to FAISS index."""
    chunks = state["chunks"]
    vectorstore_dir = state["vectorstore_dir"]
    model_name = state["embedding_model_name"]
    overwrite = state.get("overwrite", False)

    logger.info(f"[embed_and_store] Loading embedding model: '{model_name}' on device '{settings.DEVICE}'")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": settings.DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    langchain_chunks = [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in chunks
    ]

    logger.info(f"[embed_and_store] Building FAISS vector store from {len(langchain_chunks)} chunk(s)...")

    # If index already exists in target dir and not overwrite, append to it; otherwise create new
    index_path = os.path.join(vectorstore_dir, "index.faiss")
    if os.path.isfile(index_path) and not overwrite:
        logger.info(f"[embed_and_store] Existing vector store found at '{vectorstore_dir}'. Checking for duplicates.")
        existing_store = FAISS.load_local(
            vectorstore_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Avoid duplicate ingestion by checking file hashes already in existing_store
        existing_file_hashes = set()
        for doc_id, doc in existing_store.docstore._dict.items():
            f_hash = doc.metadata.get("file_hash")
            if f_hash:
                existing_file_hashes.add(f_hash)

        # Filter out chunks whose file is already in the existing_store
        new_chunks = []
        skipped_filenames = set()
        for chunk in langchain_chunks:
            f_hash = chunk.metadata.get("file_hash")
            if f_hash in existing_file_hashes:
                skipped_filenames.add(chunk.metadata.get("filename", "unknown"))
            else:
                new_chunks.append(chunk)

        if skipped_filenames:
            logger.info(f"[embed_and_store] Skipped already ingested files: {', '.join(skipped_filenames)}")

        if new_chunks:
            existing_store.add_documents(new_chunks)
            vectorstore = existing_store
            logger.info(f"[embed_and_store] Merged {len(new_chunks)} new chunk(s) into the store.")
        else:
            vectorstore = existing_store
            logger.info("[embed_and_store] No new documents to add (all were duplicates).")
    else:
        if overwrite:
            logger.info(f"[embed_and_store] Overwriting existing vector store at '{vectorstore_dir}'.")
        vectorstore = FAISS.from_documents(
            documents=langchain_chunks,
            embedding=embeddings
        )

    os.makedirs(vectorstore_dir, exist_ok=True)
    vectorstore.save_local(vectorstore_dir)
    checksum = save_index_checksum(vectorstore_dir)

    logger.info(f"[embed_and_store] ✓ Vector store saved to '{vectorstore_dir}/'")
    logger.info(f"[embed_and_store]   Checksum: {checksum}")

    # Register successfully ingested files in SQLite DB registry
    db_path = settings.DATABASE_PATH
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if overwrite:
        cursor.execute("DELETE FROM ingested_files")

    registered_hashes = set()
    for chunk in langchain_chunks:
        f_hash = chunk.metadata.get("file_hash")
        f_name = chunk.metadata.get("filename", "unknown")
        f_path = chunk.metadata.get("source", "")
        if f_hash and f_hash not in registered_hashes:
            registered_hashes.add(f_hash)
            try:
                f_size = os.path.getsize(f_path) if os.path.isfile(f_path) else 0
            except Exception:
                f_size = 0
            cursor.execute(
                "INSERT OR IGNORE INTO ingested_files (file_hash, filename, file_size, ingested_at) VALUES (?, ?, ?, ?)",
                (f_hash, f_name, f_size, int(time.time()))
            )
    conn.commit()
    conn.close()
    logger.info(f"[embed_and_store] ✓ Registered {len(registered_hashes)} file(s) in SQLite registry.")

    return {**state, "status": "completed", "checksum": checksum or ""}


# ---------------------------------------------------------------------------
# Build & compile the LangGraph Pipeline
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Assemble the ingestion graph with validation routing."""
    graph = StateGraph(PipelineState)

    graph.add_node("load", load_node)
    graph.add_node("validate", validate_node)
    graph.add_node("split", split_node)
    graph.add_node("embed_and_store", embed_and_store_node)

    graph.set_entry_point("load")
    graph.add_edge("load", "validate")

    graph.add_conditional_edges(
        "validate",
        should_continue_after_validation,
        {
            "split": "split",
            "end": END
        }
    )

    graph.add_edge("split", "embed_and_store")
    graph.add_edge("embed_and_store", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# CLI Entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into FAISS vector store via LangGraph."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single PDF file")
    group.add_argument("--dir", help="Path to a directory containing PDF files")

    parser.add_argument("--chunk-size", type=int, default=settings.CHUNK_SIZE, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=settings.CHUNK_OVERLAP, help="Chunk overlap")
    parser.add_argument("--vectorstore-dir", default=settings.VECTORSTORE_DIR, help="Output directory")
    parser.add_argument("--model", default=settings.EMBEDDING_MODEL, help="HuggingFace embedding model name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing vector store if it exists")

    args = parser.parse_args()

    file_paths = []
    if args.file:
        file_paths.append(args.file)
    elif args.dir:
        pdf_pattern = os.path.join(args.dir, "**", "*.pdf")
        file_paths = glob.glob(pdf_pattern, recursive=True)
        if not file_paths:
            logger.error(f"No PDF files found in directory '{args.dir}'")
            sys.exit(1)

    print("=" * 60)
    print("LangGraph PDF Ingestion Pipeline")
    print("Nodes: load -> validate -> split -> embed_and_store")
    print(f"Files to process: {len(file_paths)}")
    print("=" * 60)

    initial_state: PipelineState = {
        "file_paths": file_paths,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "vectorstore_dir": args.vectorstore_dir,
        "embedding_model_name": args.model,
        "documents": [],
        "chunks": [],
        "status": "initialized",
        "checksum": "",
        "error": "",
        "overwrite": args.overwrite
    }

    pipeline = build_graph()
    final_state = pipeline.invoke(initial_state)

    print("\n" + "=" * 60)
    if final_state.get("status") == "completed":
        print("✅ Ingestion complete!")
        print(f"   Pages processed : {len(final_state['documents'])}")
        print(f"   Chunks created   : {len(final_state['chunks'])}")
        print(f"   Vector store dir : {args.vectorstore_dir}/")
        print(f"   Index Checksum   : {final_state.get('checksum')}")
    else:
        print(f"❌ Ingestion failed: {final_state.get('error')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
