"""
Shared sentence-transformers embedder singleton.

Loaded lazily (first call pays the model-load cost) and reused across
services/collaboration.py (gap evidence-quality) and services/similarity.py
(existing-work detection) so the model is only ever loaded once per process.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder
