"""Tests for graph/client.py — no real Neo4j connection needed for most tests."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from unittest.mock import MagicMock, patch
import pytest


class TestHealthDisabled:
    """health() with GRAPH_ENABLED=false must return {enabled: false} and never connect."""

    def test_health_disabled_no_connect(self, monkeypatch):
        monkeypatch.setenv("GRAPH_ENABLED", "false")
        # Reload config so the env change takes effect
        import importlib
        import graph.config as cfg_mod
        importlib.reload(cfg_mod)
        import graph.client as client_mod
        importlib.reload(client_mod)

        with patch.object(client_mod, "_make_driver") as mock_make:
            result = client_mod.health()
            mock_make.assert_not_called()
        assert result == {"enabled": False}

    def test_get_client_disabled_raises(self, monkeypatch):
        monkeypatch.setenv("GRAPH_ENABLED", "false")
        import importlib
        import graph.config as cfg_mod
        importlib.reload(cfg_mod)
        import graph.client as client_mod
        importlib.reload(client_mod)

        with pytest.raises(RuntimeError, match="disabled"):
            client_mod.get_client()


class TestRetryLogic:
    """Transient errors are retried; client errors are not."""

    def _make_mock_driver(self):
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return driver, session

    def test_transient_error_retried(self):
        """_with_retry should retry transient errors up to max_attempts."""
        import importlib
        import graph.client as client_mod
        importlib.reload(client_mod)

        from neo4j.exceptions import ServiceUnavailable as SU
        client_mod._TRANSIENT = (SU,)

        attempts = []

        def raise_twice():
            attempts.append(1)
            if len(attempts) < 3:
                raise SU("transient")
            return "ok"

        with patch("time.sleep"):  # skip actual sleeps
            result = client_mod._with_retry(raise_twice, max_attempts=3)

        assert result == "ok"
        assert len(attempts) == 3

    def test_client_error_not_retried(self, monkeypatch):
        import importlib
        import graph.client as client_mod
        importlib.reload(client_mod)

        from neo4j.exceptions import ServiceUnavailable
        client_mod._TRANSIENT = (ServiceUnavailable,)

        from neo4j.exceptions import ClientError
        attempts = []

        def raises_client_error():
            attempts.append(1)
            raise ClientError("client error")

        with pytest.raises(ClientError):
            client_mod._with_retry(raises_client_error, max_attempts=3)
        assert len(attempts) == 1


class TestHealthEnabled:
    """health() with graph enabled and Neo4j reachable (integration — skipped if no DB)."""

    @pytest.mark.skipif(
        os.environ.get("GRAPH_ENABLED", "false").lower() != "true",
        reason="requires GRAPH_ENABLED=true and running Neo4j",
    )
    def test_health_reachable(self):
        import graph.client as client_mod
        result = client_mod.health()
        assert result["enabled"] is True
        assert result["reachable"] is True
        assert "server_version" in result
        assert "community" in result.get("edition", "")
        assert result["node_count"] >= 0

    @pytest.mark.skipif(
        os.environ.get("GRAPH_ENABLED", "false").lower() != "true",
        reason="requires GRAPH_ENABLED=true",
    )
    def test_health_unreachable_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "bolt://localhost:9999")
        import importlib
        import graph.config as cfg_mod
        importlib.reload(cfg_mod)
        import graph.client as client_mod
        importlib.reload(client_mod)
        client_mod._driver = None

        result = client_mod.health()
        assert result["enabled"] is True
        assert result["reachable"] is False
        assert "error" in result
