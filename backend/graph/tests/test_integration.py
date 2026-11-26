"""Graph integration tests — require the test container on port 7688.

Six tests covering: connectivity, migrations, constraints, seed reference data,
repository upsert, and authority binding.

Skip automatically when the container is unreachable (see conftest.py).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# T1: connectivity
# ---------------------------------------------------------------------------

def test_connectivity(graph_client):
    """Neo4j test instance is reachable and returns server info."""
    rows = graph_client.read(
        "CALL dbms.components() YIELD name, versions, edition "
        "RETURN name, versions[0] AS version, edition LIMIT 1"
    )
    assert rows, "dbms.components() returned no rows"
    assert "5" in rows[0].get("version", ""), f"Unexpected version: {rows[0]}"


# ---------------------------------------------------------------------------
# T2: migrations applied
# ---------------------------------------------------------------------------

def test_migrations_applied(migrated_client):
    """Both migrations (0001 and 0002) are recorded in the graph."""
    rows = migrated_client.read(
        "MATCH (m:Migration) RETURN m.version AS v ORDER BY v"
    )
    versions = [r["v"] for r in rows]
    assert 1 in versions, f"Migration 0001 not applied; found: {versions}"
    assert 2 in versions, f"Migration 0002 not applied; found: {versions}"


# ---------------------------------------------------------------------------
# T3: uniqueness constraints exist
# ---------------------------------------------------------------------------

def test_constraints_created(migrated_client):
    """Key uniqueness constraints from migration 0001 exist."""
    rows = migrated_client.read("SHOW CONSTRAINTS YIELD name RETURN name")
    names = {r["name"] for r in rows}
    for expected in ("case_key", "court_code", "principle_key"):
        assert expected in names, f"Constraint '{expected}' missing; found: {names}"


# ---------------------------------------------------------------------------
# T4: seed reference nodes from migration 0002
# ---------------------------------------------------------------------------

def test_seed_courts(migrated_client):
    """Migration 0002 created the court hierarchy with BINDS edges."""
    rows = migrated_client.read(
        "MATCH (a:Court)-[:BINDS]->(b:Court) RETURN a.code AS from_c, b.code AS to_c"
    )
    assert len(rows) >= 6, f"Expected at least 6 BINDS edges, got {len(rows)}"
    scc_binds = {r["from_c"] for r in rows if r["to_c"] == "SCC"}
    assert "KEHC" in scc_binds, "KEHC should BIND SCC (not found)"


# ---------------------------------------------------------------------------
# T5: repository node upsert
# ---------------------------------------------------------------------------

def test_repository_upsert(migrated_client):
    """Neo4jRepository.upsert_node() is idempotent for Case nodes."""
    from graph.repository import Neo4jRepository

    repo = Neo4jRepository(migrated_client)
    repo.upsert_node("Case", "case_key", {
        "case_key": "TEST-2026-99999",
        "citation": "[2026] TEST 99999",
        "name": "Integration Test Case",
        "in_corpus": False,
    })
    # Second upsert (idempotent)
    repo.upsert_node("Case", "case_key", {
        "case_key": "TEST-2026-99999",
        "citation": "[2026] TEST 99999",
        "name": "Integration Test Case (updated)",
        "in_corpus": False,
    })
    rows = migrated_client.read(
        "MATCH (n:Case {case_key: 'TEST-2026-99999'}) RETURN n.name AS name"
    )
    assert len(rows) == 1, "Expected exactly one node after two upserts"
    assert rows[0]["name"] == "Integration Test Case (updated)"
    # Cleanup
    migrated_client.write("MATCH (n:Case {case_key: 'TEST-2026-99999'}) DETACH DELETE n")


# ---------------------------------------------------------------------------
# T6: authority.is_binding() with live graph
# ---------------------------------------------------------------------------

def test_authority_binding(migrated_client):
    """is_binding() correctly identifies KEHC as binding on SCC."""
    from graph.authority import is_binding, invalidate_cache

    invalidate_cache()
    assert is_binding("KEHC", "SCC", client=migrated_client), "KEHC should bind SCC"
    assert is_binding("KECA", "SCC", client=migrated_client), "KECA should bind SCC"
    assert not is_binding("SCC", "KEHC", client=migrated_client), "SCC should NOT bind KEHC"
    assert is_binding("KEHC", "KEHC", client=migrated_client), "Same-court is always binding"
    invalidate_cache()  # clean up for other tests
