"""Neo4j driver singleton.

Usage:
    from graph.client import get_client, health

    result = get_client().read("MATCH (n:Case {case_key: $k}) RETURN n", k=case_key)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from . import config

logger = logging.getLogger(__name__)

_driver = None


def _make_driver():
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable

    password = config.require_password()
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, password),
        max_connection_pool_size=config.GRAPH_MAX_POOL,
        connection_timeout=config.GRAPH_QUERY_TIMEOUT_S,
    )
    try:
        driver.verify_connectivity()
    except ServiceUnavailable as exc:
        driver.close()
        raise RuntimeError(f"Cannot connect to Neo4j at {config.NEO4J_URI}: {exc}") from exc
    return driver


def get_client() -> "_GraphClient":
    """Return the singleton client. Raises if GRAPH_ENABLED is false."""
    if not config.GRAPH_ENABLED:
        raise RuntimeError("Graph layer is disabled (GRAPH_ENABLED=false).")
    global _driver
    if _driver is None:
        _driver = _make_driver()
    return _GraphClient(_driver)


def close_driver():
    """Close the driver on app shutdown."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


_TRANSIENT = None  # populated lazily to avoid import at module load


def _transient_exceptions():
    global _TRANSIENT
    if _TRANSIENT is None:
        from neo4j.exceptions import ServiceUnavailable, SessionExpired
        _TRANSIENT = (ServiceUnavailable, SessionExpired)
    return _TRANSIENT


def _with_retry(fn, max_attempts: int = 3):
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except _transient_exceptions() as exc:
            last_exc = exc
            wait = 0.5 * (2 ** attempt)
            logger.warning("Transient Neo4j error (attempt %d/%d): %s — retrying in %.1fs",
                           attempt + 1, max_attempts, exc, wait)
            time.sleep(wait)
    raise last_exc


class _GraphClient:
    def __init__(self, driver):
        self._driver = driver

    def read(self, query: str, **params) -> list[dict]:
        """Execute a read-only query and return rows as dicts."""
        def _run():
            with self._driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.execute_read(
                    lambda tx: list(tx.run(query, **params))
                )
                return [dict(r) for r in result]

        return _with_retry(_run)

    def write(self, query: str, **params) -> list[dict]:
        """Execute a write query and return rows as dicts."""
        def _run():
            with self._driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.execute_write(
                    lambda tx: list(tx.run(query, **params))
                )
                return [dict(r) for r in result]

        return _with_retry(_run)


def health() -> dict:
    """Return a health dict. Never raises — failures go into the payload."""
    if not config.GRAPH_ENABLED:
        return {"enabled": False}

    t0 = time.monotonic()
    try:
        global _driver
        if _driver is None:
            _driver = _make_driver()
        client = _GraphClient(_driver)
        rows = client.read(
            "CALL dbms.components() YIELD name, versions, edition "
            "RETURN name, versions[0] AS version, edition LIMIT 1"
        )
        server_version = rows[0].get("version", "unknown") if rows else "unknown"
        edition = rows[0].get("edition", "unknown") if rows else "unknown"

        counts = client.read(
            "MATCH (n) WHERE NOT n:Migration RETURN count(n) AS node_count"
        )
        node_count = counts[0].get("node_count", 0) if counts else 0

        from .migrations.runner import current_version
        migration_version = current_version(client)

        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        return {
            "enabled": True,
            "reachable": True,
            "server_version": server_version,
            "edition": edition,
            "node_count": node_count,
            "migration_version": migration_version,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        logger.warning("Graph health check failed: %s", exc)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        return {
            "enabled": True,
            "reachable": False,
            "error": str(exc),
            "latency_ms": latency_ms,
        }
