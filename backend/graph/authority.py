"""Binding-authority helpers (B2).

The BINDS relationship encodes the hierarchy: SC→CoA→{HC,ELC,ELRC}→{SCC,MC}.
is_binding() computes whether a decision from cited_court binds forum_court
by walking the BINDS path in Neo4j.

This module does NOT store a "binding" property on CITES edges — authority is
computed at query time from the graph structure.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# In-process cache for BINDS paths (populated lazily per Neo4j session)
_BINDS_CACHE: dict[tuple[str, str], bool] | None = None


def _load_binds_cache(client) -> dict[tuple[str, str], bool]:
    """Load all reachable BINDS pairs from Neo4j into a local dict."""
    rows = client.read(
        "MATCH (a:Court)-[:BINDS*1..]->(b:Court) "
        "RETURN a.code AS from_court, b.code AS to_court"
    )
    return {(r["from_court"], r["to_court"]) for r in rows}


def is_binding(cited_court: str, forum_court: str, client=None) -> bool:
    """
    Return True if decisions from cited_court bind forum_court.

    Uses BINDS transitive closure. Same-court is also binding (stare decisis).

    When client is None (GRAPH_ENABLED=false), falls back to a hard-coded
    static hierarchy so the function is usable offline.
    """
    if cited_court == forum_court:
        return True

    if client is None:
        return _is_binding_static(cited_court, forum_court)

    global _BINDS_CACHE
    if _BINDS_CACHE is None:
        _BINDS_CACHE = _load_binds_cache(client)

    return (cited_court, forum_court) in _BINDS_CACHE


# Static fallback when Neo4j is unavailable
_STATIC_HIERARCHY: dict[str, int] = {
    "KESC": 1,
    "KECA": 2,
    "KEHC": 3,
    "KEELC": 3,
    "KEELRC": 3,
    "SCC": 4,
    "KEMC": 4,
    "EACA": 9,
    "FOREIGN": 9,
}


def _is_binding_static(cited_court: str, forum_court: str) -> bool:
    cited_rank = _STATIC_HIERARCHY.get(cited_court.upper(), 99)
    forum_rank = _STATIC_HIERARCHY.get(forum_court.upper(), 99)
    if cited_rank == 99 or forum_rank == 99:
        return False
    return cited_rank < forum_rank


def invalidate_cache() -> None:
    """Call after graph structure changes to force a fresh load."""
    global _BINDS_CACHE
    _BINDS_CACHE = None
