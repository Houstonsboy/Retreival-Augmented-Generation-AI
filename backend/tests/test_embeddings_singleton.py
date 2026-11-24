"""Unit test: get_embedder() returns the same object across two calls (singleton)."""
import importlib
import sys


def test_embedder_singleton():
    # Reload embeddings to clear any cached state from other tests
    if "embeddings" in sys.modules:
        mod = sys.modules["embeddings"]
        mod._model = None  # reset singleton

    from embeddings import get_embedder

    m1 = get_embedder()
    m2 = get_embedder()
    assert m1 is m2, "get_embedder() must return the same instance on every call"
