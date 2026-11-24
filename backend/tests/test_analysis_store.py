"""
Unit tests for SCCVision/analysis_store.py (Phase 1 — FR-04).

Run with:
    cd backend
    PYTHONPATH=. pytest tests/test_analysis_store.py -v
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_BACKEND = os.path.dirname(os.path.dirname(__file__))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis(*, include_evidence=True, verdict="LIKELY VALID"):
    ev = [
        {
            "evidence_id":   "e-chat-1",
            "type":          "chat_screenshot",
            "file_name":     "shot.png",
            "sha256":        "aabbcc",
            "server_sha256": "aabbcc",
            "hash_match":    True,
            "added_at":      "2026-11-17T10:00:00+03:00",
            # OCR payload — must NOT appear in snapshot
            "ocr": {"messages": [{"index": 1, "text": "pay me"}]},
        }
    ] if include_evidence else []

    return {
        "analysis_input": {
            "case_context": {"user_role": "claimant"},
            "evidence":     ev,
        },
        "result": {
            "overall_verdict": verdict,
            "confidence":      0.8,
            "plain_summary":   "Test summary.",
            "missing_information": [],
            "strengthen_checklist": [],
            "disclaimer": "Not legal advice.",
        },
        "verification": {
            "verdict_after":    verdict,
            "quotes_grounded":  True,
            "mpesa_cross_check": {"consistent": True},
        },
        "prompt_version": "deadbeef",
        "model":          "qwen/qwen3.8-27b",
    }


def _make_mock_ref():
    mock_ref = MagicMock()
    mock_ref.set.return_value = None
    return mock_ref


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSaveAnalysisSnapshot:

    def test_returns_analysis_id_on_success(self):
        from SCCVision.analysis_store import save_analysis_snapshot

        mock_ref = _make_mock_ref()
        with patch("SCCVision.analysis_store._analysis_ref", return_value=mock_ref):
            aid = save_analysis_snapshot("uid-abc", _make_analysis())

        assert aid is not None
        assert len(aid) == 32  # uuid4().hex

    def test_snapshot_written_to_correct_path(self):
        from SCCVision.analysis_store import save_analysis_snapshot, _analysis_ref as _orig

        mock_ref = _make_mock_ref()
        captured_calls = []

        def fake_ref(uid, analysis_id):
            captured_calls.append((uid, analysis_id))
            return mock_ref

        with patch("SCCVision.analysis_store._analysis_ref", side_effect=fake_ref):
            aid = save_analysis_snapshot("uid-abc", _make_analysis())

        assert captured_calls[0][0] == "uid-abc"
        assert captured_calls[0][1] == aid  # path uses returned analysis_id

    def test_snapshot_contains_required_fields(self):
        from SCCVision.analysis_store import save_analysis_snapshot

        mock_ref = _make_mock_ref()
        with patch("SCCVision.analysis_store._analysis_ref", return_value=mock_ref):
            save_analysis_snapshot("uid-abc", _make_analysis())

        written_doc = mock_ref.set.call_args[0][0]
        for field in ("analysis_id", "result", "verification", "evidence",
                      "prompt_version", "model", "schema_version"):
            assert field in written_doc, f"missing field: {field}"
        assert written_doc["schema_version"] == 3  # schema version bumped intentionally

    def test_ocr_text_stripped_from_evidence(self):
        from SCCVision.analysis_store import save_analysis_snapshot

        mock_ref = _make_mock_ref()
        with patch("SCCVision.analysis_store._analysis_ref", return_value=mock_ref):
            save_analysis_snapshot("uid-abc", _make_analysis())

        written_doc = mock_ref.set.call_args[0][0]
        ev_items = written_doc["evidence"]
        assert len(ev_items) == 1
        assert "ocr" not in ev_items[0]
        assert ev_items[0]["evidence_id"] == "e-chat-1"
        assert ev_items[0]["hash_match"] is True

    def test_write_failure_returns_none(self):
        from SCCVision.analysis_store import save_analysis_snapshot

        mock_ref = _make_mock_ref()
        mock_ref.set.side_effect = RuntimeError("Firestore unavailable")
        with patch("SCCVision.analysis_store._analysis_ref", return_value=mock_ref):
            aid = save_analysis_snapshot("uid-abc", _make_analysis())

        assert aid is None  # never raises

    def test_size_guard_applied(self):
        from SCCVision.analysis_store import save_analysis_snapshot

        # Create an oversized plain_summary to trigger the guard
        big_analysis = _make_analysis()
        big_analysis["result"]["plain_summary"] = "x" * 1_000_000

        mock_ref = _make_mock_ref()
        with patch("SCCVision.analysis_store._analysis_ref", return_value=mock_ref):
            aid = save_analysis_snapshot("uid-abc", big_analysis)

        assert aid is not None
        written_doc = mock_ref.set.call_args[0][0]
        assert written_doc.get("truncated") is True
        assert len(written_doc["result"]["plain_summary"]) <= 500


class TestGetAnalysisSnapshot:

    def test_returns_dict_when_found(self):
        from SCCVision.analysis_store import get_analysis_snapshot

        snap = MagicMock()
        snap.exists = True
        snap.to_dict.return_value = {"analysis_id": "abc123", "result": {}}

        mock_ref = MagicMock()
        mock_ref.get.return_value = snap

        with patch("SCCVision.analysis_store._analysis_ref", return_value=mock_ref):
            result = get_analysis_snapshot("uid-abc", "abc123")

        assert result["analysis_id"] == "abc123"

    def test_returns_none_when_not_found(self):
        from SCCVision.analysis_store import get_analysis_snapshot

        snap = MagicMock()
        snap.exists = False

        mock_ref = MagicMock()
        mock_ref.get.return_value = snap

        with patch("SCCVision.analysis_store._analysis_ref", return_value=mock_ref):
            result = get_analysis_snapshot("uid-abc", "missing-id")

        assert result is None

    def test_returns_none_on_exception(self):
        from SCCVision.analysis_store import get_analysis_snapshot

        mock_ref = MagicMock()
        mock_ref.get.side_effect = RuntimeError("network error")

        with patch("SCCVision.analysis_store._analysis_ref", return_value=mock_ref):
            result = get_analysis_snapshot("uid-abc", "any-id")

        assert result is None


class TestRouterPhase1:
    """Integration-level tests: router modifications for server_sha256 and analysis_id.

    require_auth is applied at definition time so we cannot re-wrap it.
    Instead we patch auth._extract_bearer and auth._verify_and_load which are
    called on every request inside the already-applied decorator.
    """

    @staticmethod
    def _auth_patches():
        """Return context managers that fake a logged-in litigant."""
        from unittest.mock import patch as _patch
        import flask

        def fake_bearer():
            return ("fake-token", None)

        def fake_verify(token):
            return (("test-uid", {"type": "litigant"}), None)

        return (
            _patch("auth._extract_bearer", side_effect=fake_bearer),
            _patch("auth._verify_and_load", side_effect=fake_verify),
        )

    @staticmethod
    def _get_client():
        import flask
        from SCCVision.router import sccvision_bp
        app = flask.Flask("test_p1")
        app.register_blueprint(sccvision_bp)
        app.config["TESTING"] = True
        return app.test_client()

    def test_evidence_process_stores_server_sha256_and_hash_match(self):
        p1, p2 = self._auth_patches()
        client = self._get_client()
        with p1, p2:
            payload = {
                "ocr_result":    {"full_text": "hello"},
                "file_name":     "test.png",
                "sha256":        "deadbeef",
                "server_sha256": "deadbeef",
            }
            resp = client.post("/sccvision/evidence/process", json=payload)

        assert resp.status_code == 200
        ev = resp.get_json()["evidence"]
        assert ev["sha256"] == "deadbeef"
        assert ev["server_sha256"] == "deadbeef"
        assert ev["hash_match"] is True

    def test_evidence_process_hash_mismatch(self):
        p1, p2 = self._auth_patches()
        client = self._get_client()
        with p1, p2:
            payload = {
                "ocr_result":    {"full_text": "hello"},
                "sha256":        "aabbcc",
                "server_sha256": "ddeeff",
            }
            resp = client.post("/sccvision/evidence/process", json=payload)

        ev = resp.get_json()["evidence"]
        assert ev["hash_match"] is False
