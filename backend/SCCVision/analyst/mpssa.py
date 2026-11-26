"""M-Pesa SMS screenshot parsing and evidence classification.

Splits the OCR ``full_text`` of an M-Pesa SMS screenshot into one entry per
transaction code and extracts the fields deterministically. Also classifies an
OCR result as ``chat_screenshot``, ``mpesa_screenshot`` or ``other``.

Reuses the tolerant single-message field extraction in ``enrich`` where the
formats overlap, but never guesses: missing fields stay ``null``.
"""

from __future__ import annotations

import re
from typing import Any

from .enrich import parse_mpesa_message

_CODE_RE = re.compile(r"\b[A-Z0-9]{10}\b")


def _has_letter_digit(code: str) -> bool:
    return bool(re.search(r"[A-Z]", code)) and bool(re.search(r"\d", code))


def find_codes(text: str | None) -> list[str]:
    """Return the distinct 10-character transaction codes in ``text``."""
    seen: list[str] = []
    for m in _CODE_RE.finditer((text or "").upper()):
        tok = m.group(0)
        if _has_letter_digit(tok) and tok not in seen:
            seen.append(tok)
    return seen


_BANK_DATE_HEADER_RE = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*"
    r"[\s,]*\d{1,2}\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*",
    re.IGNORECASE,
)


def _split_segments(full: str, codes: list[str]) -> list[str]:
    """Split OCR text into one segment per M-Pesa transaction.

    Standard Safaricom confirmations start with their transaction code, so they
    are split at code positions. Bank-originated notifications (e.g. Absa) place
    the code mid-message after "MPESA ref:", so those are split at the date/time
    header that precedes each message instead.
    """
    normalized = " ".join((full or "").split())
    is_bank = "confirmed" not in normalized.lower() and bool(
        re.search(r"\b(?:deposited|transferred)\b", normalized, re.IGNORECASE)
    )
    if is_bank:
        parts = re.split(_BANK_DATE_HEADER_RE, normalized)
        segments = [p.strip() for p in parts if find_codes(p)]
        if segments:
            return segments

    positions = sorted({normalized.upper().find(code) for code in codes})
    segments: list[str] = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(normalized)
        segments.append(normalized[pos:end])
    return segments


def classify_evidence(ocr_result: dict[str, Any]) -> str:
    """Classify an OCR result as ``chat_screenshot`` / ``mpesa_screenshot`` / ``other``."""
    full = (ocr_result.get("full_text") or "")
    messages = ocr_result.get("messages") or []
    contact = ocr_result.get("contact") or {}

    has_confirmed = "confirmed" in full.lower()
    has_ksh = bool(re.search(r"\b(?:ksh|kes)\b", full, re.IGNORECASE))
    codes = find_codes(full)
    mpesa_header = bool(re.search(r"\bm[- ]?pesa\b", full, re.IGNORECASE))

    if has_confirmed and codes and has_ksh:
        return "mpesa_screenshot"
    if mpesa_header and (codes or has_confirmed):
        return "mpesa_screenshot"

    has_chat_contact = bool(contact.get("name") or contact.get("presence") is not None)
    has_timestamped_message = any(
        (m.get("timestamp") or m.get("time")) for m in messages
    )
    has_sender_message = any(
        (m.get("sender") in ("me", "them") or (m.get("sender") or "").strip()) for m in messages
    )
    if messages and (has_chat_contact or has_timestamped_message or has_sender_message):
        return "chat_screenshot"

    return "other"


def parse_mpesa_screenshot(
    ocr_result: dict[str, Any],
    evidence_id: str | None = None,
) -> dict[str, Any]:
    """Parse an M-Pesa SMS screenshot into one entry per transaction code.

    Returns ``{"layout": "supported", "entries": [...]}`` or
    ``{"layout": "unsupported_layout", "raw_text": ..., "entries": []}``.
    """
    full = ocr_result.get("full_text") or ""
    codes = find_codes(full)

    if not codes:
        return {
            "layout": "unsupported_layout",
            "reason": "no_transaction_codes",
            "raw_text": full,
            "entries": [],
        }

    # Split the text into one segment per transaction.
    segments = _split_segments(full, codes)

    entries: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        parsed = parse_mpesa_message(seg, i + 1)
        if parsed is None:
            continue
        till_or_paybill = parsed.get("till_number") or parsed.get("paybill")
        entries.append({
            "source": "mpesa_screenshot",
            "evidence_id": evidence_id,
            "code": parsed.get("code"),
            "amount_kes": parsed.get("amount_kes"),
            "recipient": parsed.get("recipient") or parsed.get("recipient_number"),
            "till_or_paybill": till_or_paybill,
            "datetime": parsed.get("datetime"),
            "date": parsed.get("date"),
            "time": parsed.get("time"),
            "balance": parsed.get("balance"),
            "cost": parsed.get("cost"),
            "parse_confidence": parsed.get("parse_confidence"),
        })

    if not entries:
        return {
            "layout": "unsupported_layout",
            "reason": "no_confirmations_parsed",
            "raw_text": full,
            "entries": [],
        }

    return {"layout": "supported", "entries": entries}
