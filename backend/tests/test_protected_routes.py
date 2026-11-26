"""
Auth-gate tests for the six routes that serve PDFs / chunks / locations.

Each route must:
  1. Return HTTP 401 when no Authorization header is present (no token).
  2. Return HTTP 200 (or any non-401 non-403) when a valid advocate token is supplied
     and the underlying handler is stubbed so it does not hit the filesystem.

Routes under test
-----------------
  GET  /api/case-pdf
  GET  /api/chunks/<chunk_id>/location
  GET  /api/documents/<doc_key>/chunks
  GET  /api/cases/<case_identifier>/all-chunks
  GET  /api/statutes/<source_document>/all-chunks
  GET  /api/rule-pdf

Run with:
    cd backend
    PYTHONPATH=. pytest tests/test_protected_routes.py -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_BACKEND = os.path.dirname(os.path.dirname(__file__))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


# ---------------------------------------------------------------------------
# Auth mock helpers — identical pattern to test_s106b_routes.py
# ---------------------------------------------------------------------------

def _bearer_patches(uid: str = "adv-uid", user_type: str = "advocate"):
    """Patch auth internals so Firebase is never called."""
    def fake_bearer():
        return ("fake-token", None)

    def fake_verify(_token):
        return ((uid, {"type": user_type}), None)

    return (
        patch("auth._extract_bearer", side_effect=fake_bearer),
        patch("auth._verify_and_load", side_effect=fake_verify),
    )


# ---------------------------------------------------------------------------
# Shared Flask test client for app.py routes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Import app after stubbing heavy optional dependencies so the module
    loads cleanly in a test environment that may not have all system libs.
    """
    # Stub cv2 and rapidocr so SCCVision does not crash on import
    import types, sys
    _cv2 = types.ModuleType("cv2")
    _cv2.imdecode = lambda *a, **kw: None
    _cv2.IMREAD_COLOR = 1
    sys.modules.setdefault("cv2", _cv2)

    from app import app as flask_app
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


# ---------------------------------------------------------------------------
# 401 without token — no patches needed; auth gate fires before handler
# ---------------------------------------------------------------------------

class TestNoTokenReturns401:
    def test_case_pdf_no_token(self, client):
        resp = client.get("/api/case-pdf?case_id=X")
        assert resp.status_code == 401

    def test_chunk_location_no_token(self, client):
        resp = client.get("/api/chunks/chunk-abc/location")
        assert resp.status_code == 401

    def test_document_chunks_no_token(self, client):
        resp = client.get("/api/documents/some-doc/chunks")
        assert resp.status_code == 401

    def test_case_all_chunks_no_token(self, client):
        resp = client.get("/api/cases/case-001/all-chunks")
        assert resp.status_code == 401

    def test_statute_all_chunks_no_token(self, client):
        resp = client.get("/api/statutes/EvidenceAct.pdf/all-chunks")
        assert resp.status_code == 401

    def test_rule_pdf_no_token(self, client):
        resp = client.get("/api/rule-pdf?doc_type=constitution&source=TheConstitutionOfKenya.pdf")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 403 for wrong role (litigant trying to call advocate-only routes)
# ---------------------------------------------------------------------------

class TestWrongRoleReturns403:
    def test_case_pdf_litigant_forbidden(self, client):
        litigant_patches = (
            patch("auth._extract_bearer", side_effect=lambda: ("tok", None)),
            patch("auth._verify_and_load", side_effect=lambda _t: (("uid", {"type": "litigant"}), None)),
        )
        with litigant_patches[0], litigant_patches[1]:
            resp = client.get("/api/case-pdf?case_id=X",
                              headers={"Authorization": "Bearer tok"})
        assert resp.status_code == 403

    def test_rule_pdf_litigant_forbidden(self, client):
        litigant_patches = (
            patch("auth._extract_bearer", side_effect=lambda: ("tok", None)),
            patch("auth._verify_and_load", side_effect=lambda _t: (("uid", {"type": "litigant"}), None)),
        )
        with litigant_patches[0], litigant_patches[1]:
            resp = client.get(
                "/api/rule-pdf?doc_type=constitution&source=TheConstitutionOfKenya.pdf",
                headers={"Authorization": "Bearer tok"},
            )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Auth passes for advocate — handler is stubbed to avoid filesystem I/O
# ---------------------------------------------------------------------------

class TestAdvocateTokenPassesAuthGate:
    """
    These tests only verify that the auth layer accepts an advocate token
    (status_code != 401 and != 403).  The handlers themselves may return 404
    or 500 when the underlying case/chunk does not exist in the test env.
    """

    def _adv_headers(self):
        return {"Authorization": "Bearer fake-token"}

    def test_case_pdf_auth_accepted(self, client):
        p1, p2 = _bearer_patches()
        with p1, p2:
            resp = client.get("/api/case-pdf?case_id=NONEXISTENT",
                              headers=self._adv_headers())
        assert resp.status_code not in (401, 403), (
            f"Expected auth to pass for advocate, got {resp.status_code}"
        )

    def test_chunk_location_auth_accepted(self, client):
        p1, p2 = _bearer_patches()
        with p1, p2:
            resp = client.get("/api/chunks/nonexistent-chunk/location",
                              headers=self._adv_headers())
        assert resp.status_code not in (401, 403)

    def test_document_chunks_auth_accepted(self, client):
        p1, p2 = _bearer_patches()
        with p1, p2:
            resp = client.get("/api/documents/nonexistent-doc/chunks",
                              headers=self._adv_headers())
        assert resp.status_code not in (401, 403)

    def test_case_all_chunks_auth_accepted(self, client):
        p1, p2 = _bearer_patches()
        with p1, p2:
            resp = client.get("/api/cases/NONEXISTENT/all-chunks",
                              headers=self._adv_headers())
        assert resp.status_code not in (401, 403)

    def test_statute_all_chunks_auth_accepted(self, client):
        p1, p2 = _bearer_patches()
        with p1, p2:
            resp = client.get("/api/statutes/EvidenceAct.pdf/all-chunks",
                              headers=self._adv_headers())
        assert resp.status_code not in (401, 403)

    def test_rule_pdf_auth_accepted(self, client):
        p1, p2 = _bearer_patches()
        with p1, p2:
            resp = client.get(
                "/api/rule-pdf?doc_type=constitution&source=TheConstitutionOfKenya.pdf",
                headers=self._adv_headers(),
            )
        assert resp.status_code not in (401, 403)
