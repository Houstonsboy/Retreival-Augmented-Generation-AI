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
    - result fields (AnalystResult-compat for s106B gate)
    - verification block (quotes_grounded, refs_grounded, mpesa_cross_check)
    - v2 fields: input_hash, derived_statuses, cert_eligibility, confidence
    - prompt_version, rules_version, model, calls_used
    - guided_follow_ups (capped list of FollowUpRequestV2 dicts)
    - case tracking: case_id, round, parent_analysis_id
    - evidence list stripped to: evidence_id, type, file_name, sha256,
      server_sha256, hash_match, added_at  (no OCR text, no images)

    Returns the generated analysis_id string, or None if the write fails.
    """
    analysis_id = uuid.uuid4().hex

    _KEEP = {"evidence_id", "type", "file_name", "sha256", "server_sha256", "hash_match", "added_at"}
    raw_evidence = analysis.get("analysis_input", {}).get("evidence") or []
    evidence_schedule = [
        {k: v for k, v in item.items() if k in _KEEP}
        for item in raw_evidence
    ]

    # Strip very large fields from follow-ups for Firestore (keep first 6, drop `how` if huge)
    guided_raw = analysis.get("guided_follow_ups") or []
    guided_stored = []
    for req in guided_raw[:6]:
        r = dict(req)
        if len(r.get("how", "")) > 400:
            r["how"] = r["how"][:400] + "…"
        guided_stored.append(r)

    doc = {
        "analysis_id":        analysis_id,
        "input_hash":         analysis.get("input_hash"),
        "prompt_version":     analysis.get("prompt_version"),
        "verify_prompt_version": analysis.get("verify_prompt_version"),
        "rules_version":      analysis.get("rules_version"),
        "model":              analysis.get("model"),
        "calls_used":         analysis.get("calls_used"),
        "trigger_fired":      analysis.get("trigger_fired"),
        # v2 deterministic fields
        "derived_statuses":   analysis.get("derived_statuses") or {},
        "cert_eligibility":   analysis.get("cert_eligibility") or {},
        "confidence":         analysis.get("confidence"),
        "verdict":            analysis.get("verdict"),
        "integrity_rating":   analysis.get("integrity_rating"),
        "unstable_fields":    analysis.get("unstable_fields") or [],
        "model_disagreements": analysis.get("model_disagreements") or {},
        "timestamp_info":     analysis.get("timestamp_info") or {},
        "confidence_breakdown": analysis.get("confidence_breakdown") or {},
        # Guided follow-ups
        "guided_follow_ups":  guided_stored,
        # Case tracking
        "case_id":            analysis.get("case_id") or "",
        "round":              analysis.get("round") or 1,
        "parent_analysis_id": analysis.get("parent_analysis_id"),
        # Legacy fields (kept for s106B gate)
        "result":             analysis.get("result") or {},
        "verification":       analysis.get("verification") or {},
        "evidence":           evidence_schedule,
        "created_at":         fb_firestore.SERVER_TIMESTAMP,
        "schema_version":     3,
        "truncated":          False,
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


def get_analysis_by_hash(uid: str, input_hash: str) -> dict | None:
    """Return the most-recent analysis snapshot matching input_hash, or None.

    Used by the pipeline to serve cached results (0 Groq calls) when the
    same canonical input has been analyzed before.
    """
    try:
        docs = (
            _db()
            .collection("users")
            .document(uid)
            .collection("analyses")
            .where("input_hash", "==", input_hash)
            .order_by("created_at", direction="DESCENDING")
            .limit(1)
            .get()
        )
        for doc in docs:
            if doc.exists:
                data = doc.to_dict()
                data["analysis_id"] = doc.id
                return data
        return None
    except Exception as exc:
        logger.warning("analysis_store: hash lookup failed uid=%s hash=%s: %s", uid, input_hash, exc)
        return None
