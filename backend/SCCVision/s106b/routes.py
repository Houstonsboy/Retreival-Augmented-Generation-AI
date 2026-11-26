"""Flask Blueprint: Section 106B certificate endpoints.

Routes:
    POST /api/evidence/certificates          — run gate, persist certificate
    GET  /api/evidence/certificates          — list user's certificates
    GET  /api/evidence/certificates/<id>     — get one certificate (full record)
    GET  /api/evidence/certificates/<id>/download  — download PDF
"""

from __future__ import annotations

import logging

from flask import Blueprint, Response, g, jsonify, request
from pydantic import ValidationError

from auth import require_role
from SCCVision.analysis_store import get_analysis_snapshot
from .document import render_pdf
from .gate import evaluate
from .models import DeviceFacts
from .store import contract_analysis_fields, get_certificate, list_certificates, save_certificate

logger = logging.getLogger(__name__)

s106b_bp = Blueprint("s106b", __name__, url_prefix="/api/evidence/certificates")

_ROLES = ("litigant", "advocate", "admin")


@s106b_bp.route("", methods=["POST"])
@require_role(*_ROLES)
def create_certificate():
    """POST /api/evidence/certificates

    Body (JSON):
        analysis_id   str   — ID of a saved analysis snapshot
        device_facts  dict  — DeviceFacts fields
    """
    body = request.get_json(silent=True) or {}
    analysis_id = body.get("analysis_id", "").strip()
    device_facts_raw = body.get("device_facts")

    if not analysis_id:
        return jsonify({"error": "analysis_id is required"}), 400
    if not device_facts_raw or not isinstance(device_facts_raw, dict):
        return jsonify({"error": "device_facts is required"}), 400

    # Validate DeviceFacts
    try:
        facts = DeviceFacts(**device_facts_raw)
    except ValidationError as exc:
        # Strip Pydantic ctx/url fields which may contain non-JSON-serializable values
        errors = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return jsonify({"error": "Invalid device_facts", "details": errors}), 422

    # Load analysis snapshot — owner check is implicit (uid-scoped query)
    snapshot = get_analysis_snapshot(g.uid, analysis_id)
    if snapshot is None:
        return jsonify({"error": "Analysis not found or access denied"}), 404

    # Run the deterministic gate
    gate = evaluate(snapshot, facts)

    # Persist and return
    record = save_certificate(g.uid, analysis_id, snapshot, facts, gate)
    if record is None:
        return jsonify({"error": "Failed to save certificate; please try again"}), 500

    return jsonify(record), 201


@s106b_bp.route("", methods=["GET"])
@require_role(*_ROLES)
def list_certs():
    """GET /api/evidence/certificates

    Query params:
        limit  int  default 20, max 50
    """
    try:
        limit = min(int(request.args.get("limit", 20)), 50)
    except (TypeError, ValueError):
        limit = 20

    certs = list_certificates(g.uid, limit=limit)
    return jsonify({"certificates": certs}), 200


@s106b_bp.route("/<string:cert_id>", methods=["GET"])
@require_role(*_ROLES)
def get_cert(cert_id: str):
    """GET /api/evidence/certificates/<cert_id>"""
    record = get_certificate(g.uid, cert_id)
    if record is None:
        return jsonify({"error": "Certificate not found"}), 404
    return jsonify(record), 200


@s106b_bp.route("/<string:cert_id>/download", methods=["GET"])
@require_role(*_ROLES)
def download_cert(cert_id: str):
    """GET /api/evidence/certificates/<cert_id>/download

    Regenerates the PDF deterministically from the stored record.
    """
    record = get_certificate(g.uid, cert_id)
    if record is None:
        return jsonify({"error": "Certificate not found"}), 404

    # Older certificates did not store element statuses. Recover them from the
    # original owner-scoped analysis when it is still available.
    if "derived_statuses" not in record and record.get("analysis_id"):
        snapshot = get_analysis_snapshot(g.uid, record["analysis_id"])
        if snapshot is not None:
            record = {**record, **contract_analysis_fields(snapshot)}

    try:
        pdf_bytes = render_pdf(record)
    except Exception as exc:
        logger.error("certificate PDF render failed cert_id=%s: %s", cert_id, exc)
        return jsonify({"error": "Failed to generate PDF"}), 500

    cert_number = record.get("certificate_number", cert_id)
    filename = f"{cert_number}.pdf"

    return Response(
        pdf_bytes,
        status=200,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )
