"""Tests for graph/repository.py.

Unit tests use a mock client. Integration tests are marked @pytest.mark.integration.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from unittest.mock import MagicMock, call
import pytest


def _mock_client(read_return=None):
    c = MagicMock()
    c.read.return_value = read_return if read_return is not None else []
    c.write.return_value = []
    return c


class TestUpsertNode:
    def test_upsert_calls_merge(self):
        from graph.repository import Neo4jRepository
        c = _mock_client()
        repo = Neo4jRepository(c)
        repo.upsert_node("Case", "case_key", {"case_key": "KEHC-2024-1", "name": "Test Case"})
        c.write.assert_called_once()
        call_args = c.write.call_args
        query = call_args[0][0]
        assert "MERGE" in query
        assert "case_key" in query

    def test_upsert_idempotent(self):
        from graph.repository import Neo4jRepository
        c = _mock_client()
        repo = Neo4jRepository(c)
        node = {"case_key": "KEHC-2024-1", "name": "Test Case"}
        repo.upsert_node("Case", "case_key", node)
        repo.upsert_node("Case", "case_key", node)
        assert c.write.call_count == 2

    def test_flatten_provenance(self):
        from graph.repository import _flatten
        node = {
            "case_key": "X",
            "provenance": {"source_doc_hash": "abc", "extractor": "manual"},
        }
        flat = _flatten(node)
        assert flat["case_key"] == "X"
        assert flat["provenance_source_doc_hash"] == "abc"
        assert flat["provenance_extractor"] == "manual"


class TestGetCase:
    def test_returns_none_when_missing(self):
        from graph.repository import Neo4jRepository
        c = _mock_client(read_return=[])
        repo = Neo4jRepository(c)
        result = repo.get_case("KEHC-9999-999")
        assert result is None

    def test_returns_node_dict(self):
        from graph.repository import Neo4jRepository
        mock_node = MagicMock()
        mock_node.__iter__ = MagicMock(return_value=iter({"case_key": "K", "name": "Test"}.items()))
        dict_node = {"case_key": "K", "name": "Test"}
        c = _mock_client(read_return=[{"n": dict_node}])

        repo = Neo4jRepository(c)
        # Patch the read to return a row where row["n"] is a dict
        c.read.return_value = [{"n": dict_node}]
        result = repo.get_case("K")
        assert result == dict_node


class TestNeighbors:
    def test_depth_limit(self):
        from graph.repository import Neo4jRepository
        c = _mock_client()
        repo = Neo4jRepository(c)
        with pytest.raises(ValueError, match="depth"):
            repo.neighbors("K", depth=3)

    def test_depth_1_query(self):
        from graph.repository import Neo4jRepository
        c = _mock_client(read_return=[])
        repo = Neo4jRepository(c)
        repo.neighbors("KEHC-2024-1", depth=1)
        query = c.read.call_args[0][0]
        assert "1..1" in query


class TestReset:
    def test_refuses_in_prod(self, monkeypatch):
        monkeypatch.setenv("GRAPH_ENV", "prod")
        import importlib
        import graph.config as cfg_mod
        importlib.reload(cfg_mod)
        import graph.repository as repo_mod
        importlib.reload(repo_mod)

        c = _mock_client(read_return=[])
        repo = repo_mod.Neo4jRepository(c)
        with pytest.raises(RuntimeError, match="production"):
            repo.reset(confirm=True)

    def test_requires_confirm_flag(self, monkeypatch):
        monkeypatch.setenv("GRAPH_ENV", "dev")
        import importlib
        import graph.config as cfg_mod
        importlib.reload(cfg_mod)
        import graph.repository as repo_mod
        importlib.reload(repo_mod)

        c = _mock_client(read_return=[])
        repo = repo_mod.Neo4jRepository(c)
        with pytest.raises(ValueError, match="confirm"):
            repo.reset(confirm=False)


@pytest.mark.integration
class TestRepositoryIntegration:
    @pytest.fixture(autouse=True)
    def _require_graph(self):
        if os.environ.get("GRAPH_ENABLED", "false").lower() != "true":
            pytest.skip("requires GRAPH_ENABLED=true")

    def test_upsert_case_idempotent(self):
        from graph.client import get_client
        from graph.repository import Neo4jRepository
        from graph.migrations.runner import run_pending
        c = get_client()
        run_pending(c)
        repo = Neo4jRepository(c)

        node = {"case_key": "KEHC-TEST-1", "citation": "[2024] KEHC 1 (KLR)", "name": "Test v Test"}
        repo.upsert_node("Case", "case_key", node)
        repo.upsert_node("Case", "case_key", node)

        counts = c.read("MATCH (n:Case {case_key: $k}) RETURN count(n) AS c", k="KEHC-TEST-1")
        assert counts[0]["c"] == 1

    def test_reset_refuses_in_prod(self, monkeypatch):
        monkeypatch.setenv("GRAPH_ENV", "prod")
        import importlib
        import graph.config as cfg_mod
        importlib.reload(cfg_mod)
        import graph.repository as repo_mod
        importlib.reload(repo_mod)

        from graph.client import get_client
        c = get_client()
        repo = repo_mod.Neo4jRepository(c)
        with pytest.raises(RuntimeError, match="production"):
            repo.reset(confirm=True)
