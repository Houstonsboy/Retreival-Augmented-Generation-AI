"""Shared fixtures for graph integration tests.

Tests in this package require a live Neo4j instance on port 7688 (the test
container).  They are skipped automatically when the container is unreachable.

Start the test instance:
    docker run -d --name legalai-neo4j-test \\
        -p 127.0.0.1:7688:7687 \\
        -e NEO4J_AUTH=neo4j/testpassword \\
        --tmpfs /data neo4j:5.26-community
"""
from __future__ import annotations

import os
import pytest

TEST_URI = os.getenv("NEO4J_TEST_URI", "bolt://localhost:7688")
TEST_PASSWORD = os.getenv("NEO4J_TEST_PASSWORD", "testpassword")
TEST_USER = os.getenv("NEO4J_TEST_USER", "neo4j")


def _make_test_client():
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(TEST_URI, auth=(TEST_USER, TEST_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception:
        driver.close()
        return None

    class _Client:
        def __init__(self, drv):
            self._d = drv

        def read(self, q, **p):
            with self._d.session(database="neo4j") as s:
                return [dict(r) for r in s.execute_read(lambda tx: list(tx.run(q, **p)))]

        def write(self, q, **p):
            with self._d.session(database="neo4j") as s:
                return [dict(r) for r in s.execute_write(lambda tx: list(tx.run(q, **p)))]

        def close(self):
            self._d.close()

    return _Client(driver)


@pytest.fixture(scope="session")
def graph_client():
    """Return a live test-container client, or skip the test."""
    client = _make_test_client()
    if client is None:
        pytest.skip("Neo4j test container not reachable on bolt://localhost:7688")
    yield client
    client.close()


@pytest.fixture(scope="session")
def migrated_client(graph_client):
    """Apply migrations to the test instance and return the client."""
    from graph.migrations.runner import run_pending
    run_pending(graph_client)
    return graph_client
