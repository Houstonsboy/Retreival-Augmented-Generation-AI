"""Deterministic enrichment of the cleaned OCR result into ``analysis_input``.

Everything here is code-only: sender normalisation, PII masking, the tolerant
M-Pesa parser, payment-state hints, mentioned amounts and integrity flags. No
LLM is involved, and no message text is ever logged.

The output of :func:`build_analysis_input` is exactly what is serialised and
sent to the analyst as the (untrusted) data block.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Sender normalisation
# ---------------------------------------------------------------------------

def normalize_sender(message: dict, contact: dict | None) -> tuple[str, str | None]:
    """Map the OCR sender label to ``("me" | "them", sender_name)``.

    The OCR layer already emits ``"me"`` for outgoing messages, but for
    incoming messages it emits the contact's *name* (e.g. ``"Neptune Shelters"``)
    rather than the literal ``"them"``. This normalises that so every consumer
    (analysis, verification, UI) references the same two-value sender axis while
    preserving the human-readable name separately.
    """
    raw = (message.get("sender") or "").strip()
    if raw == "me":
        return "me", None
    if raw == "them":
        return "them", (contact or {}).get("name")
    if raw == "unknown":
        return "them", None
    # Any other label is a detected contact/participant name.
    return "them", raw


# ---------------------------------------------------------------------------
# PII masking (phones and national ID numbers)
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(r"(?<![\w])(?:\+?254\d{9}|0\d{9})(?![\w])")
_ID_RE = re.compile(r"(?<![\w])(\d{7,8})(?![\w])")


def mask_phone(text: str) -> str:
    """Mask the middle digits of Kenyan phone numbers, keeping first 4 + last 3."""
    def _repl(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) <= 7:
            return m.group(0)
        return digits[:4] + "***" + digits[-3:]
    return _PHONE_RE.sub(_repl, text or "")


def mask_id_number(text: str) -> str:
    """Mask the middle digits of standalone 7-8 digit ID-like numbers.

    Runs after :func:`mask_phone`, so already-masked phone numbers (which now
    contain ``***``) are not re-matched. Till numbers (6 digits) and amounts
    (usually < 7 digits, or containing a decimal/comma) are left untouched.
    """
    def _repl(m: re.Match) -> str:
        digits = m.group(1)
        if len(digits) < 7:
            return digits
        return digits[:3] + "***" + digits[-3:]
    return _ID_RE.sub(_repl, text or "")


def mask_pii(text: str | None) -> str:
    """Mask phone and ID numbers in free text (phones first, then IDs)."""
    if text is None:
        return ""
    return mask_id_number(mask_phone(str(text)))


# ---------------------------------------------------------------------------
# Payload-size guards
# ---------------------------------------------------------------------------

_MAX_MSG_TEXT    = 400     # chars per message before truncation
_MAX_MESSAGES    = 60      # total messages per analysis call
_MAX_TOTAL_CHARS = 24_000  # hard ceiling on combined message text (~6k tokens)


def _trim_text(text: str) -> str:
    """Truncate a single message to _MAX_MSG_TEXT chars, appending '…' if cut."""
    if len(text) <= _MAX_MSG_TEXT:
        return text
    return text[:_MAX_MSG_TEXT] + "…"


def _cap_messages(messages: list[dict]) -> list[dict]:
    """Keep at most _MAX_MESSAGES messages, then apply a total-char budget.

    Two-stage approach so stitched images with many long messages stay within
    Groq's HTTP payload limit regardless of message count or text length:
      1. Count cap: keep first + last halves with a gap marker.
      2. Char budget: if the capped list still exceeds _MAX_TOTAL_CHARS of
         combined text, trim from the middle outward until it fits.
    """
    # Stage 1: count cap
    if len(messages) > _MAX_MESSAGES:
        half = _MAX_MESSAGES // 2
        omitted = len(messages) - _MAX_MESSAGES
        gap: dict = {
            "index": messages[half]["index"],
            "sender": "system",
            "sender_name": None,
            "text": f"[{omitted} messages omitted to fit analysis limit]",
            "date": None,
            "time": None,
            "edited": False,
            "ocr_confidence": 1.0,
            "sender_confidence": "high",
        }
        messages = messages[:half] + [gap] + messages[-half:]

    # Stage 2: total character budget — trim middle until under the ceiling
    total = sum(len(m.get("text") or "") for m in messages)
    while total > _MAX_TOTAL_CHARS and len(messages) > 2:
        # Remove the middle message (index len//2) to preserve first + last
        mid = len(messages) // 2
        removed = messages.pop(mid)
        total -= len(removed.get("text") or "")

    return messages


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _has_digits(text: str) -> bool:
    return any(ch.isdigit() for ch in (text or ""))


def _ocr_conf(m: dict) -> float:
    val = m.get("avg_confidence") if "avg_confidence" in m else m.get("ocr_confidence")
    try:
        return float(val)
    except (TypeError, ValueError):
        return 1.0


def _norm(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


# ---------------------------------------------------------------------------
# M-Pesa confirmation parser (tolerant, never guesses)
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(r"\b[A-Z0-9]{10}\b")
_AMOUNT_RE = re.compile(r"(?:Ksh|KSh|KSHS|KES|Kes)\s*([\d,]+(?:\.\d{1,2})?)")
_DATETIME_RE = re.compile(
    r"\bon\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+at\s+(\d{1,2}:\d{2}\s?(?:AM|PM)?)",
    re.IGNORECASE,
)
_BALANCE_RE = re.compile(
    r"balance\s+is\s+(?:Ksh|KES)?\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE
)
_COST_RE = re.compile(
    r"transaction\s+cost,?\s+(?:Ksh|KES)?\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE
)
_TILL_RE = re.compile(
    r"(?:till|buy\s*goods)\s*(?:no\.?|number)?\s*[:#]?\s*(\d{4,})", re.IGNORECASE
)
_PAYBILL_RE = re.compile(
    r"paybill\s*(?:no\.?|number)?\s*[:#]?\s*(\d{4,})", re.IGNORECASE
)
_ACCOUNT_RE = re.compile(
    r"(?:account|acc\.?)\s*(?:no\.?|number)?\s*[:#]?\s*([A-Z0-9]+)", re.IGNORECASE
)
_RECIPIENT_RE = re.compile(
    r"(?:sent|paid)\s+to\s+(.+?)(?:\s+on\s+\d{1,2}/\d{1,2}/\d{2,4}|$)",
    re.IGNORECASE,
)
_RECIPIENT_NUM_RE = re.compile(r"(\+?254\d{9}|0\d{9})")

# Bank-originated M-Pesa notifications (e.g. Absa / Equity) do not carry the
# literal word "Confirmed". They use "deposited KES... MPESA ref: CODE ... for
# RECIPIENT" or "NAME has transferred KES... to your MPESA ref: CODE".
_BANK_DEPOSIT_RE = re.compile(
    r"\bdeposited\s+(?:Ksh|KSh|KSHS|KES|Kes)\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_BANK_TRANSFER_RE = re.compile(
    r"\btransferred\s+(?:Ksh|KSh|KSHS|KES|Kes)\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_BANK_FOR_RECIPIENT_RE = re.compile(
    r"\bfor\s+(.+?)(?:\s+through\b|\s+via\b|$)", re.IGNORECASE
)
_BANK_SENDER_RE = re.compile(r"(.+?)\s+has\s+transferred\b", re.IGNORECASE)


def _find_code(text: str) -> str | None:
    for m in _CODE_RE.finditer((text or "").upper()):
        tok = m.group(0)
        if re.search(r"[A-Z]", tok) and re.search(r"\d", tok):
            return tok
    return None


def _extract_recipient(text: str) -> tuple[str | None, str | None]:
    m = _RECIPIENT_RE.search(text or "")
    if not m:
        m = _BANK_FOR_RECIPIENT_RE.search(text or "")
    if not m:
        return None, None
    seg = m.group(1).strip().rstrip(".,")
    number: str | None = None
    num_m = _RECIPIENT_NUM_RE.search(seg)
    if num_m:
        number = num_m.group(0)
        seg = seg.replace(number, "")
    name = seg.strip().strip(".").strip() or None
    return name, number


def parse_mpesa_message(text: str, msg_index: int) -> dict | None:
    """Parse a single message as an M-Pesa confirmation, or return ``None``.

    A confirmation is recognised when the literal word ``Confirmed`` is present
    (the marker the standard Safaricom message carries) or when the message is a
    bank-originated M-Pesa notification (e.g. ``deposited KES... MPESA ref:
    CODE... for RECIPIENT`` or ``NAME has transferred KES...``). Missing fields
    stay ``null`` and a ``parse_confidence`` in [0, 1] is reported.
    """
    if not text:
        return None
    lower = (text or "").lower()
    is_confirmed = "confirmed" in lower
    is_bank = bool(_BANK_DEPOSIT_RE.search(text) or _BANK_TRANSFER_RE.search(text))
    if not is_confirmed and not is_bank:
        return None

    code = _find_code(text)
    amount_raw = None
    am = _AMOUNT_RE.search(text or "")
    if am:
        amount_raw = am.group(0)
    amount = _to_float(am.group(1)) if am else None

    date = time = None
    dt = _DATETIME_RE.search(text or "")
    if dt:
        date = dt.group(1)
        time = dt.group(2)

    balance = cost = None
    bm = _BALANCE_RE.search(text or "")
    if bm:
        balance = _to_float(bm.group(1))
    cm = _COST_RE.search(text or "")
    if cm:
        cost = _to_float(cm.group(1))

    recipient_name, recipient_number = _extract_recipient(text)

    sender = None
    sm = _BANK_SENDER_RE.search(text or "")
    if sm:
        sender = sm.group(1).strip().rstrip(".,") or None

    till = paybill = account = None
    tm = _TILL_RE.search(text or "")
    if tm:
        till = tm.group(1)
    pm = _PAYBILL_RE.search(text or "")
    if pm:
        paybill = pm.group(1)
    ac = _ACCOUNT_RE.search(text or "")
    if ac:
        account = ac.group(1)

    # Core fields used for confidence: code, amount, a recipient identifier,
    # and a date/time. Partial confirmations still parse with lower confidence.
    core = [
        code,
        amount is not None,
        bool(recipient_name or recipient_number or till or paybill or account),
        bool(date or time),
    ]
    parse_confidence = round(sum(1 for c in core if c) / len(core), 2)

    return {
        "code": code,
        "amount_kes": amount,
        "amount_raw": amount_raw,
        "sender": sender,  # determinable for bank "has transferred" messages
        "recipient": recipient_name,
        "recipient_number": recipient_number,
        "till_number": till,
        "paybill": paybill,
        "account": account,
        "datetime": f"{date} {time}".strip() if (date or time) else None,
        "date": date,
        "time": time,
        "balance": balance,
        "cost": cost,
        "msg_index": msg_index,
        "parse_confidence": parse_confidence,
    }


# ---------------------------------------------------------------------------
# Payment-state hints and mentioned amounts
# ---------------------------------------------------------------------------

_CLAIM_RE = re.compile(
    r"\b(done|nimetuma|nimelipa|sent|paid|deposited|sending|nimepay|nimelipa)\b",
    re.IGNORECASE,
)
_DETAILS_RE = re.compile(r"payment\s+details", re.IGNORECASE)


def detect_payment_hints(text: str) -> dict:
    """Per-message payment-state hints (leads for the Analyst, not verdicts)."""
    t = text or ""
    return {
        "details_given": bool(
            _DETAILS_RE.search(t)
            or _TILL_RE.search(t)
            or _PAYBILL_RE.search(t)
            or _ACCOUNT_RE.search(t)
        ),
        "claimed": bool(_CLAIM_RE.search(t)),
        "confirmed": False,
    }


def extract_amounts(text: str) -> list[dict]:
    """Extract amounts mentioned in chat text, for cross-checks."""
    amounts: list[dict] = []
    seen: set[tuple[float, str]] = set()

    for m in re.finditer(
        r"(?:Ksh|KSh|KSHS|KES|Kes)\s*([\d,]+(?:\.\d{1,2})?)", text or "", re.IGNORECASE
    ):
        value = _to_float(m.group(1))
        if value is not None:
            key = (round(value, 2), m.group(0).strip())
            if key not in seen:
                seen.add(key)
                amounts.append({"value": value, "raw": m.group(0).strip()})

    for m in re.finditer(
        r"\b(?:amount|charge|price|rate|fee|cost|deposit)\b"
        r"\s*(?:is|of|:|=|was)?\s*(?:Ksh|KES)?\s*([\d,]+(?:\.\d{1,2})?)",
        text or "",
        re.IGNORECASE,
    ):
        value = _to_float(m.group(1))
        if value is not None:
            key = (round(value, 2), m.group(0).strip())
            if key not in seen:
                seen.add(key)
                amounts.append({"value": value, "raw": m.group(0).strip()})

    return amounts


# ---------------------------------------------------------------------------
# Integrity flags
# ---------------------------------------------------------------------------

_DATE_CHIP_RE = re.compile(r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.IGNORECASE)
_DELETED_RE = re.compile(r"deleted", re.IGNORECASE)
_MEDIA_RE = re.compile(
    r"(photo|voice\s+message|video|gif|image|document|<media\s+omitted>|audio|sticker|location)",
    re.IGNORECASE,
)


def _to_minutes(hhmm: str) -> int | None:
    try:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def build_integrity_flags(
    raw_messages: list[dict],
    system_events: list[dict] | None,
    image_count: int = 1,
) -> list[str]:
    """Compute code-level integrity flags from the *unmasked* OCR messages."""
    flags: list[str] = []
    events = system_events or []

    if not any(_DATE_CHIP_RE.search(ev.get("text") or "") for ev in events):
        flags.append("missing_dates")

    times = [
        _to_minutes(m.get("timestamp") or m.get("time"))
        for m in raw_messages
        if (m.get("timestamp") or m.get("time"))
    ]
    if any(
        b is not None and a is not None and b < a
        for a, b in zip(times, times[1:])
    ):
        flags.append("time_order_backwards")

    if any(m.get("edited") for m in raw_messages):
        flags.append("edited_messages")
    if any(m.get("incomplete") or not (m.get("timestamp") or m.get("time")) for m in raw_messages):
        flags.append("incomplete_messages")
    if any((m.get("sender_confidence") or "high").lower() != "high" for m in raw_messages):
        flags.append("low_sender_confidence")
    if any(_has_digits(m.get("text") or "") and _ocr_conf(m) < 0.85 for m in raw_messages):
        flags.append("low_ocr_confidence")
    if any(_DELETED_RE.search(ev.get("text") or "") for ev in events):
        flags.append("deleted_messages")
    if any(_MEDIA_RE.search(m.get("text") or "") for m in raw_messages):
        flags.append("media_placeholder")
    if image_count > 1:
        flags.append("screenshot_boundary")

    return flags


# ---------------------------------------------------------------------------
# Multi-image merge
# ---------------------------------------------------------------------------

def _same_message(a: dict, b: dict) -> bool:
    return (
        _norm(a.get("text")) == _norm(b.get("text"))
        and (a.get("timestamp") or a.get("time")) == (b.get("timestamp") or b.get("time"))
        and a.get("sender") == b.get("sender")
    )


def merge_ocr_results(results: list[dict]) -> dict:
    """Concatenate messages from multiple screenshots, de-duplicating boundaries.

    Exact duplicates (same text, time and sender) at the join between two
    screenshots are dropped once. Message indexes are re-numbered 1-based and
    ``system_events`` keep a (approximate) remapped ``after_message_index``.
    """
    merged: list[dict] = []
    events: list[dict] = []
    contact: dict | None = None

    for r in results:
        if contact is None and (r.get("contact") or {}).get("name"):
            contact = r["contact"]
        offset = len(merged)
        for m in r.get("messages") or []:
            if merged and _same_message(merged[-1], m):
                continue
            merged.append(dict(m))
        for ev in r.get("system_events") or []:
            ev = dict(ev)
            ev["after_message_index"] = offset + int(ev.get("after_message_index") or 0)
            events.append(ev)

    for i, m in enumerate(merged, 1):
        m["index"] = i

    return {"contact": contact, "messages": merged, "system_events": events}


# ---------------------------------------------------------------------------
# analysis_input
# ---------------------------------------------------------------------------

def build_analysis_input(
    cleaned: dict,
    case_context: dict | None,
    image_count: int = 1,
) -> dict:
    """Build the masked ``analysis_input`` object sent to the analyst."""
    contact = cleaned.get("contact") or {}
    raw_messages = cleaned.get("messages") or []
    raw_events = cleaned.get("system_events") or []
    context = case_context or {}

    user_role = context.get("user_role") or "unknown"
    if user_role not in ("claimant", "respondent", "unknown"):
        user_role = "unknown"

    messages: list[dict] = []
    mpesa_extracted: list[dict] = []
    payment_hints: list[dict] = []
    mentioned_amounts: list[dict] = []

    for m in raw_messages:
        text_raw = m.get("text") or ""
        sender, sender_name = normalize_sender(m, contact)
        idx = m.get("index")

        messages.append({
            "index": idx,
            "sender": sender,
            "sender_name": mask_pii(sender_name),
            "text": _trim_text(mask_pii(text_raw)),
            "date": None,
            "time": m.get("timestamp") or m.get("time"),
            "edited": bool(m.get("edited")),
            "ocr_confidence": (
                m.get("avg_confidence")
                if "avg_confidence" in m
                else m.get("ocr_confidence")
            ),
            "sender_confidence": m.get("sender_confidence"),
        })

        parsed = parse_mpesa_message(text_raw, idx)
        hints = detect_payment_hints(text_raw)
        if parsed is not None:
            if parsed.get("recipient_number"):
                parsed["recipient_number"] = mask_pii(parsed["recipient_number"])
            mpesa_extracted.append(parsed)
            hints["confirmed"] = True
        hints["msg_index"] = idx
        payment_hints.append(hints)

        for amt in extract_amounts(text_raw):
            mentioned_amounts.append({
                "amount": amt["value"],
                "raw": amt["raw"],
                "msg_index": idx,
            })

    system_events = [
        {
            "after_message_index": ev.get("after_message_index"),
            "text": mask_pii(ev.get("text") or ""),
        }
        for ev in raw_events
    ]

    return {
        "case_context": {
            "user_role": user_role,
            "user_account": mask_pii(context.get("user_account") or ""),
        },
        "contact": {
            "name": mask_pii(contact.get("name")) if contact.get("name") else None,
            "presence": contact.get("presence"),
        },
        "messages": _cap_messages(messages),
        "system_events": system_events,
        "mpesa_extracted": mpesa_extracted,
        "integrity_flags": build_integrity_flags(raw_messages, raw_events, image_count),
        "mentioned_amounts": mentioned_amounts,
    }


# ---------------------------------------------------------------------------
# Case-file based analysis_input (new INPUT section)
# ---------------------------------------------------------------------------

def _effective_chat_messages(evidence_item: dict) -> list[dict]:
    """Return the (possibly user-edited) OCR messages for a chat evidence item."""
    ocr = evidence_item.get("ocr") or {}
    messages = [dict(m) for m in (ocr.get("messages") or [])]
    edits = evidence_item.get("user_edits") or {}
    for m in messages:
        idx = m.get("index")
        edited = None
        if isinstance(edits, dict):
            edited = edits.get(str(idx)) if str(idx) in edits else edits.get(idx)
        if isinstance(edited, dict) and edited.get("text") is not None:
            m["text"] = edited["text"]
        elif isinstance(edited, str):
            m["text"] = edited
    return messages


def _chat_mpesa_entry(parsed: dict, evidence_id: str | None) -> dict:
    till_or_paybill = parsed.get("till_number") or parsed.get("paybill")
    return {
        "source": "chat_message",
        "evidence_id": evidence_id,
        "code": parsed.get("code"),
        "amount_kes": parsed.get("amount_kes"),
        "recipient": parsed.get("recipient") or parsed.get("recipient_number"),
        "till_or_paybill": till_or_paybill,
        "datetime": parsed.get("datetime"),
        "parse_confidence": parsed.get("parse_confidence"),
    }


def _previous_analysis(case_file: dict) -> dict | None:
    analyses = case_file.get("analyses") or []
    if not analyses:
        return None
    last = analyses[-1]
    return {
        "verdict": last.get("verdict"),
        "element_statuses": last.get("element_statuses"),
        "open_request_ids": last.get("open_request_ids"),
    }


def build_analysis_input_from_case(
    case_file: dict,
    date_checks: dict,
) -> dict:
    """Build the masked ``analysis_input`` from the whole (frontend-held) case file."""
    context = case_file.get("case_context") or {}

    user_role = context.get("user_role") or "unknown"
    if user_role not in ("claimant", "respondent", "unknown"):
        user_role = "unknown"

    evidence_list: list[dict] = []
    messages: list[dict] = []
    system_events: list[dict] = []
    contact: dict | None = None
    raw_messages: list[dict] = []
    raw_events: list[dict] = []
    chat_count = 0

    evidence_items = case_file.get("evidence") or []

    for ev in evidence_items:
        evidence_list.append({
            "evidence_id": ev.get("evidence_id"),
            "type": ev.get("type"),
            "label": ev.get("label") or ev.get("file_name"),
        })
        if ev.get("type") != "chat_screenshot":
            continue
        chat_count += 1
        ocr = ev.get("ocr") or {}
        if contact is None and (ocr.get("contact") or {}).get("name"):
            contact = ocr["contact"]
        eid = ev.get("evidence_id")
        for m in _effective_chat_messages(ev):
            raw_messages.append(m)
            idx = len(messages) + 1
            sender, sender_name = normalize_sender(m, contact)
            text_raw = m.get("text") or ""
            messages.append({
                "index": idx,
                "evidence_id": eid,
                "sender": sender,
                "sender_name": mask_pii(sender_name),
                "text": _trim_text(mask_pii(text_raw)),
                "date": None,
                "time": m.get("timestamp") or m.get("time"),
                "edited": bool(m.get("edited")),
                "ocr_confidence": (
                    m.get("avg_confidence")
                    if "avg_confidence" in m
                    else m.get("ocr_confidence")
                ),
                "sender_confidence": m.get("sender_confidence"),
            })
        for se in ocr.get("system_events") or []:
            raw_events.append(se)
            system_events.append({
                "evidence_id": eid,
                "after_message_index": se.get("after_message_index"),
                "text": mask_pii(se.get("text") or ""),
            })

    # M-Pesa extracted from every source: chat text, M-Pesa screenshots, manual.
    mpesa_extracted: list[dict] = []
    for ev in evidence_items:
        eid = ev.get("evidence_id")
        if ev.get("type") in ("mpesa_screenshot", "other"):
            for p in ev.get("parsed") or []:
                entry = {
                    "source": p.get("source", "mpesa_screenshot"),
                    "evidence_id": eid,
                    "code": p.get("code"),
                    "amount_kes": p.get("amount_kes"),
                    "recipient": p.get("recipient"),
                    "till_or_paybill": p.get("till_or_paybill"),
                    "datetime": p.get("datetime"),
                    "parse_confidence": p.get("parse_confidence"),
                }
                mpesa_extracted.append(entry)

    payment_hints: list[dict] = []
    mentioned_amounts: list[dict] = []
    for m in messages:
        text_raw = m["text"]
        idx = m["index"]
        parsed = parse_mpesa_message(text_raw, idx)
        hints = detect_payment_hints(text_raw)
        if parsed is not None:
            if parsed.get("recipient_number"):
                parsed["recipient_number"] = mask_pii(parsed["recipient_number"])
            mpesa_extracted.append(_chat_mpesa_entry(parsed, m["evidence_id"]))
            hints["confirmed"] = True
        hints["msg_index"] = idx
        payment_hints.append(hints)
        for amt in extract_amounts(text_raw):
            mentioned_amounts.append({
                "amount": amt["value"],
                "raw": amt["raw"],
                "msg_index": idx,
            })

    integrity_flags = build_integrity_flags(raw_messages, raw_events, chat_count or 1)
    if date_checks.get("no_dates") and "missing_dates" not in integrity_flags:
        integrity_flags.append("missing_dates")
    if date_checks.get("future_dated_events"):
        integrity_flags.append("future_dated_event")
    if date_checks.get("order_violations"):
        integrity_flags.append("date_order_violation")

    user_provided_facts = []
    for f in case_file.get("user_provided_facts") or []:
        fact = f.get("fact")
        if f.get("kind") in ("text", "choice") and isinstance(fact, str):
            fact = mask_pii(fact)
        user_provided_facts.append({
            "id": f.get("id"),
            "kind": f.get("kind"),
            "fact": fact,
            "applies_to": f.get("applies_to"),
            "answers_request_id": f.get("answers_request_id"),
            "status": f.get("status", "user-stated"),
        })

    return {
        "case_context": {
            "user_role": user_role,
            "user_account": mask_pii(context.get("user_account") or ""),
            "analysis_date": date_checks.get("analysis_date"),
        },
        "evidence": evidence_list,
        "contact": {
            "name": mask_pii(contact.get("name")) if contact and contact.get("name") else None,
            "presence": contact.get("presence") if contact else None,
        },
        "messages": _cap_messages(messages),
        "system_events": system_events,
        "mpesa_extracted": mpesa_extracted,
        "mentioned_amounts": mentioned_amounts,
        "date_checks": date_checks,
        "integrity_flags": integrity_flags,
        "user_provided_facts": user_provided_facts,
        "previous_analysis": _previous_analysis(case_file),
    }
