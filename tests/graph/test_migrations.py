"""Tests for the migration runner (unit only — no live DB).

Integration tests (apply to real neo4j-test container) are marked
@pytest.mark.integration and skipped unless GRAPH_ENABLED=true.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import hashlib
import pathlib
from unittest.mock import MagicMock, call
import pytest


def _make_client(applied_rows=None, write_rows=None, read_rows=None):
    """Return a mock client for the migration runner."""
    c = MagicMock()
    c.read.return_value = applied_rows if applied_rows is not None else []
    c.write.return_value = write_rows if write_rows is not None else []
    return c


class TestRunPending:
    def test_first_run_applies_0001(self, tmp_path):
        """First run should apply migration 0001."""
        # Create a temporary migrations dir with one file
        (tmp_path / "0001_test.cypher").write_text("CREATE INDEX foo IF NOT EXISTS FOR (n:Foo) ON (n.bar);")

        import graph.migrations.runner as runner
        orig_dir = runner._MIGRATIONS_DIR
        runner._MIGRATIONS_DIR = tmp_path
        try:
            c = _make_client(applied_rows=[])
            new = runner.run_pending(c)
            assert 1 in new
        finally:
            runner._MIGRATIONS_DIR = orig_dir

    def test_second_run_is_noop(self, tmp_path):
        """Second run with the same applied migration should be a no-op."""
        content = "CREATE INDEX foo IF NOT EXISTS FOR (n:Foo) ON (n.bar);"
        f = tmp_path / "0001_test.cypher"
        f.write_text(content)
        checksum = hashlib.sha256(content.encode()).hexdigest()

        import graph.migrations.runner as runner
        orig_dir = runner._MIGRATIONS_DIR
        runner._MIGRATIONS_DIR = tmp_path
        try:
            applied_rows = [{"v": 1, "name": "test", "checksum": checksum, "applied_at": "2026-01-01"}]
            c = _make_client(applied_rows=applied_rows)
            new = runner.run_pending(c)
            assert new == []
        finally:
            runner._MIGRATIONS_DIR = orig_dir

    def test_changed_checksum_refused(self, tmp_path):
        """A modified migration file must be refused."""
        content = "CREATE INDEX foo IF NOT EXISTS FOR (n:Foo) ON (n.bar);"
        f = tmp_path / "0001_test.cypher"
        f.write_text(content)
        wrong_checksum = "deadbeef" + "0" * 56

        import graph.migrations.runner as runner
        orig_dir = runner._MIGRATIONS_DIR
        runner._MIGRATIONS_DIR = tmp_path
        try:
            applied_rows = [{"v": 1, "name": "test", "checksum": wrong_checksum, "applied_at": "2026-01-01"}]
            c = _make_client(applied_rows=applied_rows)
            with pytest.raises(RuntimeError, match="checksum"):
                runner.run_pending(c)
        finally:
            runner._MIGRATIONS_DIR = orig_dir


class TestCurrentVersion:
    def test_no_migrations(self):
        import graph.migrations.runner as runner
        c = _make_client(applied_rows=[{"v": None}])
        result = runner.current_version(c)
        assert result is None

    def test_returns_max(self):
        import graph.migrations.runner as runner
        c = _make_client(applied_rows=[{"v": 3}])
        result = runner.current_version(c)
        assert result == 3


@pytest.mark.integration
class TestMigrationsIntegration:
    """Run against the neo4j-test container (requires GRAPH_ENABLED=true, port 7688)."""

    @pytest.fixture(autouse=True)
    def _require_graph(self):
        if os.environ.get("GRAPH_ENABLED", "false").lower() != "true":
            pytest.skip("requires GRAPH_ENABLED=true")

    def test_0001_constraints_and_indexes_online(self):
        from graph.client import get_client
        from graph.migrations.runner import run_pending, status
        c = get_client()
        run_pending(c)

        # All constraints from 0001 should exist
        constraints = {r["name"] for r in c.read("SHOW CONSTRAINTS YIELD name")}
        for expected in [
            "case_key", "court_code", "judge_key", "migration_ver",
            "paragraph_key", "provision_key", "statute_key",
        ]:
            assert expected in constraints, f"Constraint '{expected}' not found"

        # Vector index should be ONLINE
        indexes = {r["name"]: r.get("state") for r in c.read(
            "SHOW INDEXES YIELD name, state WHERE name = 'paragraph_embedding'"
        )}
        assert "paragraph_embedding" in indexes
        assert indexes["paragraph_embedding"] == "ONLINE"

    def test_idempotent_second_run(self):
        from graph.client import get_client
        from graph.migrations.runner import run_pending
        c = get_client()
        run_pending(c)
        new = run_pending(c)
        assert new == []
