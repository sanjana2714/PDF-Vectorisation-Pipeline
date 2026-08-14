"""
config.py — Centralized configuration management for PDF Vectorisation Pipeline.
"""

import os

# Enforce PyTorch backend for Hugging Face Transformers & prevent Keras 3 TensorFlow conflicts
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_default_device() -> str:
    """Detect CUDA or MPS availability if PyTorch is installed, otherwise fallback to cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Settings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Embedding & Hardware
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEVICE: str = get_default_device()
    
    # Vectorstore Persistence
    VECTORSTORE_DIR: str = "./vectorstore"
    
    # Text Chunking Default Parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # API Server Configuration
    MAX_QUERY_LENGTH: int = 2000
    DEFAULT_TOP_K: int = 5
    CORS_ORIGINS: List[str] = ["*"]
    LOG_LEVEL: str = "INFO"
    API_KEY: Optional[str] = None
    DATABASE_PATH: str = "./vectorstore/metadata.db"


settings = Settings()
