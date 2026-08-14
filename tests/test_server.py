"""
tests/test_server.py — Integration tests for FastAPI server endpoints.
"""

import os
import sys
# Add project root to sys.path to allow running pytest directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from server import app, load_vectorstore_from_disk
import pytest

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "embedding_model" in data
    assert "device" in data


def test_query_validation_empty_query():
    response = client.post("/query", json={"query": "", "top_k": 3})
    assert response.status_code == 422  # Unprocessable Entity (Validation Error)


def test_query_validation_oversized_query():
    large_query = "a" * 2005
    response = client.post("/query", json={"query": large_query, "top_k": 3})
    assert response.status_code == 422


def test_query_execution_and_score_range():
    # If vectorstore loaded successfully, test search execution
    load_vectorstore_from_disk()
    response = client.post("/query", json={"query": "What is LangChain?", "top_k": 2})

    if response.status_code == 503:
        pytest.skip("Vector store not ingested/loaded on disk.")

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is LangChain?"
    assert len(data["results"]) <= 2

    for res in data["results"]:
        assert "chunk_text" in res
        assert 0.0 <= res["score"] <= 1.0


def test_reload_endpoint():
    response = client.post("/reload")
    # Will return 200 if store exists, or 500 if store isn't present
    assert response.status_code in (200, 500)


def test_api_key_unauthorized():
    from config import settings
    settings.API_KEY = "supersecret"
    try:
        response = client.post("/query", json={"query": "test", "top_k": 3})
        assert response.status_code == 401
        
        response_reload = client.post("/reload")
        assert response_reload.status_code == 401
    finally:
        settings.API_KEY = None


def test_api_key_authorized():
    from config import settings
    settings.API_KEY = "supersecret"
    try:
        response = client.post("/query", json={"query": "test", "top_k": 3}, headers={"X-API-Key": "supersecret"})
        assert response.status_code != 401
        
        response_reload = client.post("/reload", headers={"X-API-Key": "supersecret"})
        assert response_reload.status_code != 401
    finally:
        settings.API_KEY = None
