"""Persist and retrieve Section 106B certificate records.

Certificates are stored at users/{uid}/certificates/{cert_id} in Firestore.
The PDF is never stored — it is regenerated deterministically from the record
on every download request.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from firebase_admin import firestore as fb_firestore

from . import GATE_VERSION
from .models import DeviceFacts, GateResult

logger = logging.getLogger(__name__)

_MAX_DOC_BYTES = 900_000
_SCHEMA_VERSION = 1


def _db():
    return fb_firestore.client()


def _cert_ref(uid: str, cert_id: str):
    return _db().collection("users").document(uid).collection("certificates").document(cert_id)


def _certs_col(uid: str):
    return _db().collection("users").document(uid).collection("certificates")


def _make_cert_number() -> str:
    """Generate SCC106B-YYYYMMDD-XXXXXX certificate number."""
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:6].upper()
    return f"SCC106B-{today}-{suffix}"


def contract_analysis_fields(snapshot: dict) -> dict:
    """Keep the status and summary fields needed to regenerate a PDF."""
    result = snapshot.get("result") or {}
    statuses = {
        letter: element.get("status")
        for letter, element in (result.get("elements") or {}).items()
        if isinstance(element, dict)
    }
    statuses.update(snapshot.get("derived_statuses") or {})
    return {
        "derived_statuses": statuses,
        "confidence": snapshot.get("confidence"),
        "verdict": snapshot.get("verdict") or result.get("overall_verdict"),
    }


def save_certificate(
    uid: str,
    analysis_id: str,
    snapshot: dict,
    facts: DeviceFacts,
    gate: GateResult,
) -> dict | None:
    """Build and persist a certificate record; return the saved dict or None on error.

    The returned dict is exactly what will be used by render_pdf() and the GET endpoint.
    It does NOT include Firestore SERVER_TIMESTAMP (replaced by datetime string after write).
    """
    cert_id = uuid.uuid4().hex
    cert_number = _make_cert_number()

    # Evidence schedule (already stripped by analysis_store)
    evidence = snapshot.get("evidence") or []

    # Analysis summary fields for the PDF
    result = snapshot.get("result") or {}
    analysis_summary = {
        "overall_verdict":    result.get("overall_verdict"),
        "plain_summary":      (result.get("plain_summary") or "")[:500],
        "missing_information": result.get("missing_information") or [],
        "strengthen_checklist": result.get("strengthen_checklist") or [],
        "disclaimer":         result.get("disclaimer"),
        "prompt_version":     snapshot.get("prompt_version"),
        "model":              snapshot.get("model"),
    }

    record: dict = {
        "cert_id":          cert_id,
        "certificate_number": cert_number,
        "document_type":    gate.document_type,
        "outcome":          gate.outcome,
        "analysis_id":      analysis_id,
        "uid":              uid,
        "evidence":         evidence,
        "device_facts":     facts.model_dump(),
        "gate":             gate.model_dump(),
        "analysis_summary": analysis_summary,
        **contract_analysis_fields(snapshot),
        "created_at":       fb_firestore.SERVER_TIMESTAMP,
        "schema_version":   _SCHEMA_VERSION,
    }

    try:
        _cert_ref(uid, cert_id).set(record)
    except Exception as exc:
        logger.warning("certificate_store: write failed uid=%s: %s", uid, exc)
        return None

    # Replace the SERVER_TIMESTAMP sentinel with an ISO string for immediate use
    record["created_at"] = datetime.now(tz=timezone.utc).isoformat()
    return record


def get_certificate(uid: str, cert_id: str) -> dict | None:
    """Return certificate record or None if not found / wrong owner."""
    try:
        snap = _cert_ref(uid, cert_id).get()
        if not snap.exists:
            return None
        d = snap.to_dict()
        # Firestore timestamps → ISO strings
        d = _coerce_timestamps(d)
        if d.get("uid") != uid:
            return None
        return d
    except Exception as exc:
        logger.warning("certificate_store: get failed uid=%s id=%s: %s", uid, cert_id, exc)
        return None


def list_certificates(uid: str, limit: int = 20) -> list[dict]:
    """Return summary list of certificates for uid, most-recent first."""
    try:
        query = (
            _certs_col(uid)
            .order_by("created_at", direction=fb_firestore.Query.DESCENDING)
            .limit(limit)
        )
        return [
            _summary(_coerce_timestamps(doc.to_dict()))
            for doc in query.stream()
        ]
    except Exception as exc:
        logger.warning("certificate_store: list failed uid=%s: %s", uid, exc)
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary(record: dict) -> dict:
    """Return a lightweight summary dict for list responses."""
    return {
        "cert_id":          record.get("cert_id"),
        "certificate_number": record.get("certificate_number"),
        "document_type":    record.get("document_type"),
        "outcome":          record.get("outcome"),
        "analysis_id":      record.get("analysis_id"),
        "created_at":       record.get("created_at"),
    }


def _coerce_timestamps(d: dict) -> dict:
    """Convert Firestore DatetimeWithNanoseconds to ISO strings."""
    if not d:
        return d
    result = dict(d)
    for key, val in result.items():
        if hasattr(val, "isoformat"):
            result[key] = val.isoformat()
    return result
