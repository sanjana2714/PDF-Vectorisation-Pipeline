"""
tests/test_ingest.py — Unit and integration tests for LangGraph ingestion pipeline.
"""

import os
import sys
import shutil
import tempfile
import pytest

# Add project root to sys.path to allow running pytest directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import clean_text, l2_to_cosine_similarity, save_index_checksum, verify_index_checksum
from ingest import build_graph, PipelineState


def test_clean_text_utility():
    raw = "Header text\n\nPage 1 of 12\n\nSome body text here.\n\n\n\nFooter text"
    cleaned = clean_text(raw)
    assert "Page 1 of 12" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Some body text here." in cleaned


def test_l2_to_cosine_similarity_conversion():
    # L2_squared = 0.0 -> Cosine Sim = 1.0
    assert l2_to_cosine_similarity(0.0) == 1.0
    # L2_squared = 2.0 (orthogonal vectors) -> Cosine Sim = 0.0
    assert l2_to_cosine_similarity(2.0) == 0.0
    # L2_squared > 2.0 bounds at 0.0
    assert l2_to_cosine_similarity(2.5) == 0.0


def test_ingest_pipeline_with_sample_pdf():
    # Use existing sample PDF 'langchain.pdf' if present
    pdf_path = "langchain.pdf"
    if not os.path.isfile(pdf_path):
        pytest.skip("langchain.pdf not found in test workspace.")

    temp_dir = tempfile.mkdtemp()
    try:
        pipeline = build_graph()
        initial_state: PipelineState = {
            "file_paths": [pdf_path],
            "chunk_size": 500,
            "chunk_overlap": 50,
            "vectorstore_dir": temp_dir,
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "documents": [],
            "chunks": [],
            "status": "initialized",
            "checksum": "",
            "error": "",
            "overwrite": True
        }

        final_state = pipeline.invoke(initial_state)

        assert final_state["status"] == "completed"
        assert len(final_state["documents"]) > 0
        assert len(final_state["chunks"]) > 0
        assert os.path.isfile(os.path.join(temp_dir, "index.faiss"))
        assert os.path.isfile(os.path.join(temp_dir, "index.pkl"))
        assert os.path.isfile(os.path.join(temp_dir, "index.checksum"))
        assert verify_index_checksum(temp_dir) is True
    finally:
        shutil.rmtree(temp_dir)
