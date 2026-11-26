"""Module-level SentenceTransformer singleton.

Loading the model takes ~2–8 s and allocates ~1 GB of memory.
Constructing it once per process (not once per request) eliminates
repeated load cost on the retrieval hot path (D-08).

Usage:
    from embeddings import get_embedder
    model = get_embedder()          # first call loads; subsequent calls return cached instance
    vec = model.encode("query: " + text)
"""
from __future__ import annotations

import threading
from typing import Optional

_lock = threading.Lock()
_model: Optional[object] = None  # SentenceTransformer, typed loosely to avoid import at module level


def get_embedder(model_name: str = "intfloat/e5-base-v2"):
    """Return the process-wide SentenceTransformer instance, loading it on first call."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(model_name)
    return _model
