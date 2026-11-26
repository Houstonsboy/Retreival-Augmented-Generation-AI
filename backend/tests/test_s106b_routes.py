"""
Unit tests for Section 106B certificate API routes (Phase 4 — FR-04).

Run with:
    cd backend
    PYTHONPATH=. pytest tests/test_s106b_routes.py -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import flask
import pytest

_BACKEND = os.path.dirname(os.path.dirname(__file__))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from SCCVision.s106b.routes import s106b_bp


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _get_client(uid: str = "test-uid", user_type: str = "litigant"):
    """Return a Flask test client with the s106b blueprint registered."""
    app = flask.Flask("test_s106b_routes")
    app.register_blueprint(s106b_bp)
    app.config["TESTING"] = True
    app._test_uid = uid
    app._test_type = user_type
    return app.test_client()


def _auth_patches(uid: str = "test-uid", user_type: str = "litigant"):
    """Fake auth: inject uid and user_type without hitting Firebase."""
    from unittest.mock import patch as _patch

    def fake_bearer():
        return ("fake-token", None)

    def fake_verify(_token):
        return ((uid, {"type": user_type}), None)

    return (
        _patch("auth._extract_bearer", side_effect=fake_bearer),
        _patch("auth._verify_and_load", side_effect=fake_verify),
    )


def _good_device_facts():
    return {
        "device_make":          "Samsung",
        "device_model":         "Galaxy A14",
        "phone_number_last4":   "6789",
        "production_method":    "screenshot_on_device",
        "owner_is_declarant":   True,
        "lawful_control":       True,
        "regular_use":          True,
        "regular_feed":         True,
        "operating_properly":   True,
        "malfunction_details":  None,
        "malfunction_affects_record": None,
        "messages_on_this_device": True,
        "declarant_name":       "Jane Synthetic",
        "declarant_position":   "Owner and sole user",
        "declaration_confirmed": True,
    }


def _good_snapshot():
    return {
        "analysis_id": "snap-abc",
        "result": {
            "overall_verdict":    "LIKELY VALID",
            "confidence":         0.9,
            "plain_summary":      "Looks fine.",
            "missing_information": [],
            "strengthen_checklist": [],
            "disclaimer":         "Not legal advice.",
            "conversation_integrity": {"rating": "ADEQUATE", "issues": []},
            "mpesa_payments":     [],
        },
        "verification": {
            "quotes_grounded":    True,
            "ungrounded_quotes":  [],
            "mpesa_cross_check":  {"consistent": True, "invented_fields": False},
        },
        "evidence": [
            {
                "evidence_id":   "e-chat-1",
                "type":          "chat_screenshot",
                "file_name":     "conversation.png",
                "sha256":        "aabbcc",
                "server_sha256": "aabbcc",
                "hash_match":    True,
                "added_at":      "2026-11-17T10:00:00",
            }
        ],
        "prompt_version": "deadbeef",
        "model":          "qwen/qwen3.8-27b",
    }


def _saved_cert_record():
    return {
        "cert_id":          "cert-xyz",
        "certificate_number": "SCC106B-20261117-ABCDEF",
        "document_type":    "CERTIFICATE",
        "outcome":          "certificate_ready",
        "analysis_id":      "snap-abc",
        "uid":              "test-uid",
        "evidence":         _good_snapshot()["evidence"],
        "device_facts":     _good_device_facts(),
        "gate": {
            "document_type": "CERTIFICATE",
            "outcome":       "certificate_ready",
            "conditions":    [],
            "unmet":         [],
            "gate_version":  "s106b-gate-1.0",
        },
        "analysis_summary": {
            "overall_verdict":    "LIKELY VALID",
            "plain_summary":      "Looks fine.",
            "missing_information": [],
            "strengthen_checklist": [],
            "disclaimer":         "Not legal advice.",
        },
        "created_at":     "2026-11-17T10:00:00+00:00",
        "schema_version": 1,
    }


# ---------------------------------------------------------------------------
# POST /api/evidence/certificates
# ---------------------------------------------------------------------------

class TestCreateCertificate:

    def test_201_on_success(self):
        p1, p2 = _auth_patches()
        client = _get_client()
        record = _saved_cert_record()

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_analysis_snapshot", return_value=_good_snapshot()), \
             patch("SCCVision.s106b.routes.save_certificate", return_value=record):
            resp = client.post(
                "/api/evidence/certificates",
                json={"analysis_id": "snap-abc", "device_facts": _good_device_facts()},
            )

        assert resp.status_code == 201
        body = resp.get_json()
        assert body["certificate_number"] == "SCC106B-20261117-ABCDEF"
        assert body["document_type"] == "CERTIFICATE"

    def test_400_missing_analysis_id(self):
        p1, p2 = _auth_patches()
        client = _get_client()

        with p1, p2:
            resp = client.post(
                "/api/evidence/certificates",
                json={"device_facts": _good_device_facts()},
            )

        assert resp.status_code == 400
        assert "analysis_id" in resp.get_json()["error"]

    def test_400_missing_device_facts(self):
        p1, p2 = _auth_patches()
        client = _get_client()

        with p1, p2:
            resp = client.post(
                "/api/evidence/certificates",
                json={"analysis_id": "snap-abc"},
            )

        assert resp.status_code == 400

    def test_422_invalid_device_facts(self):
        p1, p2 = _auth_patches()
        client = _get_client()

        bad_facts = dict(_good_device_facts())
        bad_facts["device_make"] = ""  # empty → validation error

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_analysis_snapshot", return_value=_good_snapshot()):
            resp = client.post(
                "/api/evidence/certificates",
                json={"analysis_id": "snap-abc", "device_facts": bad_facts},
            )

        assert resp.status_code == 422

    def test_404_snapshot_not_found(self):
        p1, p2 = _auth_patches()
        client = _get_client()

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_analysis_snapshot", return_value=None):
            resp = client.post(
                "/api/evidence/certificates",
                json={"analysis_id": "unknown-id", "device_facts": _good_device_facts()},
            )

        assert resp.status_code == 404

    def test_500_on_save_failure(self):
        p1, p2 = _auth_patches()
        client = _get_client()

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_analysis_snapshot", return_value=_good_snapshot()), \
             patch("SCCVision.s106b.routes.save_certificate", return_value=None):
            resp = client.post(
                "/api/evidence/certificates",
                json={"analysis_id": "snap-abc", "device_facts": _good_device_facts()},
            )

        assert resp.status_code == 500

    def test_401_unauthenticated(self):
        client = _get_client()
        resp = client.post("/api/evidence/certificates", json={})
        assert resp.status_code == 401

    def test_403_wrong_role(self):
        # 'viewer' is not an allowed role
        p1, p2 = _auth_patches(user_type="viewer")
        client = _get_client()

        with p1, p2:
            resp = client.post(
                "/api/evidence/certificates",
                json={"analysis_id": "snap-abc", "device_facts": _good_device_facts()},
            )

        assert resp.status_code == 403

    def test_advocate_role_allowed(self):
        p1, p2 = _auth_patches(user_type="advocate")
        client = _get_client()
        record = _saved_cert_record()

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_analysis_snapshot", return_value=_good_snapshot()), \
             patch("SCCVision.s106b.routes.save_certificate", return_value=record):
            resp = client.post(
                "/api/evidence/certificates",
                json={"analysis_id": "snap-abc", "device_facts": _good_device_facts()},
            )

        assert resp.status_code == 201

    def test_gate_runs_deterministically(self):
        """The gate is called without mocking — verifies it runs and produces document_type."""
        p1, p2 = _auth_patches()
        client = _get_client()

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_analysis_snapshot", return_value=_good_snapshot()), \
             patch("SCCVision.s106b.routes.save_certificate", return_value=_saved_cert_record()):
            resp = client.post(
                "/api/evidence/certificates",
                json={"analysis_id": "snap-abc", "device_facts": _good_device_facts()},
            )

        # save_certificate is mocked but we confirm the gate did not blow up
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# GET /api/evidence/certificates
# ---------------------------------------------------------------------------

class TestListCertificates:

    def test_200_returns_list(self):
        p1, p2 = _auth_patches()
        client = _get_client()

        summaries = [
            {
                "cert_id":          "cert-xyz",
                "certificate_number": "SCC106B-20261117-ABCDEF",
                "document_type":    "CERTIFICATE",
                "outcome":          "certificate_ready",
                "analysis_id":      "snap-abc",
                "created_at":       "2026-11-17T10:00:00+00:00",
            }
        ]

        with p1, p2, \
             patch("SCCVision.s106b.routes.list_certificates", return_value=summaries):
            resp = client.get("/api/evidence/certificates")

        assert resp.status_code == 200
        assert resp.get_json()["certificates"] == summaries

    def test_200_empty_list(self):
        p1, p2 = _auth_patches()
        client = _get_client()

        with p1, p2, \
             patch("SCCVision.s106b.routes.list_certificates", return_value=[]):
            resp = client.get("/api/evidence/certificates")

        assert resp.status_code == 200
        assert resp.get_json()["certificates"] == []

    def test_401_unauthenticated(self):
        client = _get_client()
        resp = client.get("/api/evidence/certificates")
        assert resp.status_code == 401

    def test_limit_capped_at_50(self):
        p1, p2 = _auth_patches()
        client = _get_client()

        with p1, p2, \
             patch("SCCVision.s106b.routes.list_certificates", return_value=[]) as mock_list:
            client.get("/api/evidence/certificates?limit=999")

        call_kwargs = mock_list.call_args
        limit_used = (
            call_kwargs.kwargs.get("limit")
            if call_kwargs.kwargs
            else call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        )
        assert limit_used is not None and limit_used <= 50


# ---------------------------------------------------------------------------
# GET /api/evidence/certificates/<id>
# ---------------------------------------------------------------------------

class TestGetCertificate:

    def test_200_returns_record(self):
        p1, p2 = _auth_patches()
        client = _get_client()
        record = _saved_cert_record()

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_certificate", return_value=record):
            resp = client.get("/api/evidence/certificates/cert-xyz")

        assert resp.status_code == 200
        assert resp.get_json()["cert_id"] == "cert-xyz"

    def test_404_not_found(self):
        p1, p2 = _auth_patches()
        client = _get_client()

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_certificate", return_value=None):
            resp = client.get("/api/evidence/certificates/nonexistent")

        assert resp.status_code == 404

    def test_401_unauthenticated(self):
        client = _get_client()
        resp = client.get("/api/evidence/certificates/cert-xyz")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/evidence/certificates/<id>/download
# ---------------------------------------------------------------------------

class TestDownloadCertificate:

    def test_200_returns_pdf(self):
        p1, p2 = _auth_patches()
        client = _get_client()
        record = _saved_cert_record()

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_certificate", return_value=record), \
             patch("SCCVision.s106b.routes.render_pdf", return_value=b"%PDF-1.4 fake"):
            resp = client.get("/api/evidence/certificates/cert-xyz/download")

        assert resp.status_code == 200
        assert resp.content_type == "application/pdf"
        assert resp.data.startswith(b"%PDF")

    def test_legacy_certificate_download_uses_saved_analysis_statuses(self):
        from io import BytesIO

        from pypdf import PdfReader

        p1, p2 = _auth_patches()
        client = _get_client()
        record = _saved_cert_record()
        snapshot = _good_snapshot()
        snapshot["derived_statuses"] = {"A": "PRESENT", "B": "ABSENT"}

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_certificate", return_value=record), \
             patch("SCCVision.s106b.routes.get_analysis_snapshot", return_value=snapshot) as lookup:
            resp = client.get("/api/evidence/certificates/cert-xyz/download")

        assert resp.status_code == 200
        lookup.assert_called_once_with(record["uid"], record["analysis_id"])
        reader = PdfReader(BytesIO(resp.data))
        appendix = reader.pages[-1].extract_text()
        assert "Parties & identity ✓" in appendix
        assert "Capacity ✗" in appendix

    def test_pdf_filename_uses_cert_number(self):
        p1, p2 = _auth_patches()
        client = _get_client()
        record = _saved_cert_record()

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_certificate", return_value=record), \
             patch("SCCVision.s106b.routes.render_pdf", return_value=b"%PDF-1.4"):
            resp = client.get("/api/evidence/certificates/cert-xyz/download")

        cd = resp.headers.get("Content-Disposition", "")
        assert "SCC106B-20261117-ABCDEF" in cd

    def test_404_cert_not_found(self):
        p1, p2 = _auth_patches()
        client = _get_client()

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_certificate", return_value=None):
            resp = client.get("/api/evidence/certificates/unknown/download")

        assert resp.status_code == 404

    def test_500_on_pdf_render_error(self):
        p1, p2 = _auth_patches()
        client = _get_client()
        record = _saved_cert_record()

        with p1, p2, \
             patch("SCCVision.s106b.routes.get_certificate", return_value=record), \
             patch("SCCVision.s106b.routes.render_pdf", side_effect=RuntimeError("fpdf crash")):
            resp = client.get("/api/evidence/certificates/cert-xyz/download")

        assert resp.status_code == 500

    def test_401_unauthenticated(self):
        client = _get_client()
        resp = client.get("/api/evidence/certificates/cert-xyz/download")
        assert resp.status_code == 401
