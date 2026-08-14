"""
utils.py — Helper utilities for vector store checksum verification, text cleaning, and score normalization.
"""

import hashlib
import os
import re
from typing import Optional


def calculate_index_checksum(vectorstore_dir: str) -> Optional[str]:
    """Calculate combined SHA-256 checksum for index.faiss and index.pkl."""
    faiss_path = os.path.join(vectorstore_dir, "index.faiss")
    pkl_path = os.path.join(vectorstore_dir, "index.pkl")

    if not (os.path.isfile(faiss_path) and os.path.isfile(pkl_path)):
        return None

    hasher = hashlib.sha256()
    for file_path in sorted([faiss_path, pkl_path]):
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

    return hasher.hexdigest()


def save_index_checksum(vectorstore_dir: str) -> Optional[str]:
    """Calculate and write index.checksum file into the vectorstore directory."""
    checksum = calculate_index_checksum(vectorstore_dir)
    if checksum:
        checksum_path = os.path.join(vectorstore_dir, "index.checksum")
        with open(checksum_path, "w", encoding="utf-8") as f:
            f.write(checksum)
    return checksum


def verify_index_checksum(vectorstore_dir: str) -> bool:
    """Verify that current index files match saved index.checksum."""
    checksum_path = os.path.join(vectorstore_dir, "index.checksum")
    if not os.path.isfile(checksum_path):
        # If no checksum file exists yet, return True if index files exist (backward compatibility)
        faiss_path = os.path.join(vectorstore_dir, "index.faiss")
        pkl_path = os.path.join(vectorstore_dir, "index.pkl")
        return os.path.isfile(faiss_path) and os.path.isfile(pkl_path)

    with open(checksum_path, "r", encoding="utf-8") as f:
        saved_checksum = f.read().strip()

    current_checksum = calculate_index_checksum(vectorstore_dir)
    return saved_checksum == current_checksum and saved_checksum != ""


def clean_text(text: str) -> str:
    """Clean extracted document text by stripping header/footer noise and excessive whitespace."""
    if not text:
        return ""

    # Normalize carriage returns and non-breaking spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")

    # Remove standalone page numbers or "Page X of Y" artifacts
    text = re.sub(r"(?i)^\s*page\s+\d+(\s+of\s+\d+)?\s*$", "", text, flags=re.MULTILINE)

    # Collapse multiple consecutive blank lines (more than 2 newlines) into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def l2_to_cosine_similarity(l2_distance: float) -> float:
    """
    Convert FAISS L2 distance score for normalized vectors to Cosine Similarity score [0, 1].
    L2^2 = 2 * (1 - CosineSimilarity) => CosineSimilarity = 1 - (L2^2 / 2)
    Note: FAISS similarity_search_with_score returns the squared L2 distance (L2^2).
    """
    cosine_sim = 1.0 - float(l2_distance) / 2.0
    return max(0.0, min(1.0, round(cosine_sim, 4)))
