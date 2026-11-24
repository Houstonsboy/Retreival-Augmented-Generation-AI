"""Graph layer configuration loaded from environment variables.

When GRAPH_ENABLED is false (the default) nothing connects to Neo4j and
importing this module has no side effects beyond reading os.environ.
"""
from __future__ import annotations

import os


def _env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def _bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, str(default)).strip().lower()
    return val in ("1", "true", "yes", "on")


def _int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


GRAPH_ENABLED: bool = _bool("GRAPH_ENABLED", False)
GRAPH_ENV: str = _env("GRAPH_ENV", "dev")

NEO4J_URI: str = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = _env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str | None = _env("NEO4J_PASSWORD")
NEO4J_DATABASE: str = _env("NEO4J_DATABASE", "neo4j")

GRAPH_QUERY_TIMEOUT_S: int = _int("GRAPH_QUERY_TIMEOUT_S", 5)
GRAPH_MAX_POOL: int = _int("GRAPH_MAX_POOL", 10)


def require_password() -> str:
    """Return NEO4J_PASSWORD; raise if not set and graph is enabled."""
    if GRAPH_ENABLED and not NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD must be set when GRAPH_ENABLED=true")
    return NEO4J_PASSWORD or ""
