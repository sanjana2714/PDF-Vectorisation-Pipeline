"""
ingest.py — PDF Vectorisation Pipeline (LangGraph implementation)

Usage:
    python ingest.py --file document.pdf [--chunk-size 1000] [--chunk-overlap 200] [--vectorstore-dir ./vectorstore]

LangGraph node flow:
    load → split → embed → store
"""

import argparse
import os
import sys
from typing import TypedDict, List

from langchain_community.document_loaders import PyPDFLoader  # noqa: community still ships PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# State schema for the LangGraph pipeline
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    """Shared state passed between LangGraph nodes."""
    file_path: str
    chunk_size: int
    chunk_overlap: int
    vectorstore_dir: str
    documents: List[Document]
    chunks: List[Document]
    embeddings: HuggingFaceEmbeddings
    vectorstore: FAISS


# ---------------------------------------------------------------------------
# Node 1 — load
# ---------------------------------------------------------------------------

def load_node(state: PipelineState) -> PipelineState:
    """Load the PDF using PyPDFLoader and attach pages to state."""
    file_path = state["file_path"]
    print(f"\n[load] Loading PDF: {file_path}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"[load] ✓ Loaded {len(documents)} page(s)")
    return {**state, "documents": documents}


# ---------------------------------------------------------------------------
# Node 2 — split
# ---------------------------------------------------------------------------

def split_node(state: PipelineState) -> PipelineState:
    """Split documents into chunks using RecursiveCharacterTextSplitter.

    chunk_size=1000  → captures a meaningful semantic unit (≈ 1–2 paragraphs)
                        without being so large that retrieved chunks become noisy.
    chunk_overlap=200 → ~20% overlap preserves context at chunk boundaries so
                         that sentences split across two chunks are still
                         retrievable from either side.
    """
    print(f"\n[split] Splitting with chunk_size={state['chunk_size']}, "
          f"chunk_overlap={state['chunk_overlap']}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=state["chunk_size"],
        chunk_overlap=state["chunk_overlap"],
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(state["documents"])
    print(f"[split] ✓ Produced {len(chunks)} chunk(s)")
    return {**state, "chunks": chunks}


# ---------------------------------------------------------------------------
# Node 3 — embed
# ---------------------------------------------------------------------------

def embed_node(state: PipelineState) -> PipelineState:
    """Initialise the HuggingFace embedding model.

    Model: sentence-transformers/all-MiniLM-L6-v2
      - 384-dimensional dense vectors
      - Excellent semantic quality for English text
      - Runs locally — no API key required
      - ~90 MB download on first run (cached thereafter)
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\n[embed] Loading embedding model: {model_name}")
    print("[embed] (First run downloads ~90 MB — subsequent runs use cache)")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Smoke-test with a single string to catch issues early
    _ = embeddings.embed_query("smoke test")
    print("[embed] ✓ Embedding model ready")
    return {**state, "embeddings": embeddings}


# ---------------------------------------------------------------------------
# Node 4 — store
# ---------------------------------------------------------------------------

def store_node(state: PipelineState) -> PipelineState:
    """Create a FAISS vector store from chunks and persist it to disk."""
    vectorstore_dir = state["vectorstore_dir"]
    print(f"\n[store] Building FAISS index from {len(state['chunks'])} chunk(s)")

    vectorstore = FAISS.from_documents(
        documents=state["chunks"],
        embedding=state["embeddings"],
    )

    os.makedirs(vectorstore_dir, exist_ok=True)
    vectorstore.save_local(vectorstore_dir)

    print(f"[store] ✓ Vector store saved to '{vectorstore_dir}/'")
    print(f"[store]   Files: index.faiss, index.pkl")
    return {**state, "vectorstore": vectorstore}


# ---------------------------------------------------------------------------
# Build & compile the LangGraph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Assemble the ingestion graph: load → split → embed → store."""
    graph = StateGraph(PipelineState)

    graph.add_node("load", load_node)
    graph.add_node("split", split_node)
    graph.add_node("embed", embed_node)
    graph.add_node("store", store_node)

    graph.set_entry_point("load")
    graph.add_edge("load", "split")
    graph.add_edge("split", "embed")
    graph.add_edge("embed", "store")
    graph.add_edge("store", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into a FAISS vector store via LangGraph."
    )
    parser.add_argument("--file", required=True, help="Path to the PDF file")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200)")
    parser.add_argument("--vectorstore-dir", default="./vectorstore", help="Directory to save the vector store")
    args = parser.parse_args()

    pipeline = build_graph()

    # Print the graph node list (as required by the assessment)
    print("=" * 60)
    print("LangGraph Ingestion Pipeline")
    print("Nodes: load -> split -> embed -> store")
    print("=" * 60)

    initial_state: PipelineState = {
        "file_path": args.file,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "vectorstore_dir": args.vectorstore_dir,
        "documents": [],
        "chunks": [],
        "embeddings": None,
        "vectorstore": None,
    }

    final_state = pipeline.invoke(initial_state)

    print("\n" + "=" * 60)
    print("✅ Ingestion complete!")
    print(f"   Pages loaded : {len(final_state['documents'])}")
    print(f"   Chunks stored: {len(final_state['chunks'])}")
    print(f"   Vector store : {args.vectorstore_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
