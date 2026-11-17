"""Persist analysis snapshots for certificate generation.

Stores a stripped-down version of each /sccvision/case/analyze response to
Firestore at users/{uid}/analyses/{analysis_id}.  The snapshot is the source
of truth for certificate generation so the certificate is never built from
browser-supplied data.

Modelled on history_store.py; reuses its _size_guard pattern.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from firebase_admin import firestore as fb_firestore

logger = logging.getLogger(__name__)

_MAX_DOC_BYTES = 900_000  # Firestore 1 MB limit with safety margin


def _db():
    return fb_firestore.client()


def _analysis_ref(uid: str, analysis_id: str):
    return _db().collection("users").document(uid).collection("analyses").document(analysis_id)


def _size_guard(doc: dict) -> dict:
    """Truncate oversized snapshots before writing, preserving required fields."""
    try:
        raw = json.dumps(doc, default=str).encode("utf-8")
    except Exception:
        return doc
    if len(raw) <= _MAX_DOC_BYTES:
        return doc
    doc = dict(doc)
    doc["truncated"] = True
    result = doc.get("result") or {}
    if "plain_summary" in result and result["plain_summary"]:
        result = dict(result)
        result["plain_summary"] = result["plain_summary"][:500]
        doc["result"] = result
    return doc


def save_analysis_snapshot(
    uid: str,
    analysis: dict,
) -> str | None:
    """Write a stripped analysis snapshot; return analysis_id or None on error.

    ``analysis`` is the dict returned by pipeline.analyze_case().  We store:
    - result fields (all AnalystResult fields)
    - verification block
    - prompt_version, model
    - evidence list stripped to: evidence_id, type, file_name, sha256,
      server_sha256, hash_match, added_at  (no OCR text, no images)

    Returns the generated analysis_id string, or None if the write fails.
    """
    analysis_id = uuid.uuid4().hex

    # Build evidence schedule (no OCR payloads)
    _KEEP = {"evidence_id", "type", "file_name", "sha256", "server_sha256", "hash_match", "added_at"}
    raw_evidence = analysis.get("analysis_input", {}).get("evidence") or []
    evidence_schedule = [
        {k: v for k, v in item.items() if k in _KEEP}
        for item in raw_evidence
    ]

    doc = {
        "analysis_id":   analysis_id,
        "prompt_version": analysis.get("prompt_version"),
        "model":          analysis.get("model"),
        "result":         analysis.get("result") or {},
        "verification":   analysis.get("verification") or {},
        "evidence":       evidence_schedule,
        "created_at":     fb_firestore.SERVER_TIMESTAMP,
        "schema_version": 1,
        "truncated":      False,
    }
    doc = _size_guard(doc)

    try:
        _analysis_ref(uid, analysis_id).set(doc)
        return analysis_id
    except Exception as exc:
        logger.warning("analysis_store: failed to save snapshot for uid=%s: %s", uid, exc)
        return None


def get_analysis_snapshot(uid: str, analysis_id: str) -> dict | None:
    """Return the analysis snapshot dict or None if not found / wrong owner."""
    try:
        snap = _analysis_ref(uid, analysis_id).get()
        if not snap.exists:
            return None
        return snap.to_dict()
    except Exception as exc:
        logger.warning("analysis_store: get failed uid=%s id=%s: %s", uid, analysis_id, exc)
        return None
