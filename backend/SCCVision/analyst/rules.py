"""Deterministic verification rules for the analyst result.

These rules double-check the LLM against the (masked) data it was shown and the
case file. They are code-only and never call the LLM:

- quote grounding (every evidence quote must trace to a message)
- M-Pesa cross-matching, reclassification and cross-evidence checks
- element status enforcement and no-evidence downgrade
- timeline referencing
- conservative verdict rule (with the ``no_dates`` cap)
- premature-breach detection
- deterministic follow-up request generation and merging

All rules operate on the *masked* ``analysis_input`` so masking never causes
false quote mismatches.
"""

from __future__ import annotations

import re
from typing import Any

from .models import (
    AnalystResult,
    ElementStatus,
    FollowUpRequest,
    OverallVerdict,
    PaymentClassification,
)

_AMOUNT_TOL = 1.0
_NO_ISSUE_ELEMENTS = {"B", "H", "I"}
_NOT_REQUIRED_ELEMENTS = {"J"}
_REQUIRED_PRESENT = {"C", "D", "E", "F", "G"}
_A_MIN = {"A"}

_CONFIRMATION_CLASSES = {
    PaymentClassification.VERIFIED_LOOKING,
    PaymentClassification.PRESENT_BUT_UNVERIFIED,
    PaymentClassification.SUSPICIOUS,
}


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


# ---------------------------------------------------------------------------
# Quote grounding
# ---------------------------------------------------------------------------

def check_quotes_grounded(
    result: AnalystResult,
    analysis_input: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return a list of evidence quotes that cannot be traced to a message."""
    messages = analysis_input.get("messages") or []
    normalized = [_norm(m.get("text") or "") for m in messages]
    max_index = max((int(m.get("index") or 0) for m in messages), default=0)

    ungrounded: list[dict[str, Any]] = []
    for key, element in (result.elements or {}).items():
        for ev in (element.evidence or []):
            quote_norm = _norm(ev.quote)
            if not quote_norm:
                continue
            idx = ev.msg_index
            out_of_range = idx is None or not isinstance(idx, int) or idx < 1 or idx > max_index
            found = any(quote_norm in n for n in normalized)
            if out_of_range or not found:
                ungrounded.append({
                    "element": key,
                    "msg_index": ev.msg_index,
                    "quote": ev.quote,
                    "reason": "message_index_out_of_range" if out_of_range else "quote_not_found",
                })
    return ungrounded


# ---------------------------------------------------------------------------
# M-Pesa cross-match + reclassification + cross-evidence
# ---------------------------------------------------------------------------

def _parsed_by_code(analysis_input: dict[str, Any]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in analysis_input.get("mpesa_extracted") or []:
        if p.get("code"):
            out.setdefault(str(p["code"]).upper(), p)
    return out


def _match_parsed(entry: Any, parsed_list: list[dict]) -> dict | None:
    if entry.code:
        code = str(entry.code).upper()
        for p in parsed_list:
            if p.get("code") and str(p["code"]).upper() == code:
                return p
    if entry.amount_kes is not None:
        for p in parsed_list:
            if p.get("amount_kes") is None:
                continue
            if abs(float(p["amount_kes"]) - float(entry.amount_kes)) > _AMOUNT_TOL:
                continue
            if entry.recipient:
                if p.get("recipient") and _norm(entry.recipient).lower() in _norm(p["recipient"]).lower():
                    return p
                if p.get("till_or_paybill") and entry.recipient == p["till_or_paybill"]:
                    return p
    return None


def check_mpesa_cross_match(
    result: AnalystResult,
    analysis_input: dict[str, Any],
) -> dict[str, Any]:
    """Reclassify payments and run cross-evidence checks. Mutates payments."""
    parsed = analysis_input.get("mpesa_extracted") or []
    parsed_by_code = _parsed_by_code(analysis_input)
    mentioned = [a.get("amount") for a in (analysis_input.get("mentioned_amounts") or [])]
    chat_messages = analysis_input.get("messages") or []

    llm_codes_before = {str(p.code).upper() for p in result.mpesa_payments if p.code}

    reclassified: list[dict[str, Any]] = []
    invented_fields = False
    consistent = True

    for entry in result.mpesa_payments:
        presents_confirmation = (
            "confirmation" in (entry.basis or [])
            or entry.classification in _CONFIRMATION_CLASSES
        )
        match = _match_parsed(entry, parsed) if entry.code or entry.amount_kes is not None else None

        if presents_confirmation and match is None:
            consistent = False
            reclassified.append({
                "code": entry.code,
                "amount_kes": entry.amount_kes,
                "before": entry.classification.value,
                "after": PaymentClassification.CLAIMED_ONLY.value,
            })
            entry.classification = PaymentClassification.CLAIMED_ONLY
            if entry.code is not None or entry.datetime is not None:
                invented_fields = True
            entry.code = None
            entry.datetime = None
            entry.basis = [b for b in (entry.basis or []) if b != "confirmation"]

    # Cross-evidence checks.
    chat_till_numbers = _chat_till_numbers(chat_messages)
    till_mismatches: list[dict[str, Any]] = []
    amount_mismatches: list[dict[str, Any]] = []
    duplicate_codes: list[str] = []

    codes_seen: dict[str, int] = {}
    for p in parsed:
        code = (p.get("code") or "").upper()
        if code:
            codes_seen[code] = codes_seen.get(code, 0) + 1

    for p in parsed:
        till = p.get("till_or_paybill")
        if till and chat_till_numbers:
            for chat_till in chat_till_numbers:
                if till != chat_till:
                    till_mismatches.append({
                        "code": p.get("code"),
                        "confirmation_till": till,
                        "chat_till": chat_till,
                    })
        amt = p.get("amount_kes")
        if amt is not None and mentioned:
            if not any(abs(float(a) - float(amt)) <= _AMOUNT_TOL for a in mentioned if a is not None):
                amount_mismatches.append({
                    "code": p.get("code"),
                    "amount_kes": amt,
                    "chat_amounts": mentioned,
                })
    duplicate_codes = [c for c, n in codes_seen.items() if n > 1]

    return {
        "llm_payment_count": len(result.mpesa_payments),
        "parsed_payment_count": len(parsed),
        "codes_llm_only": sorted(llm_codes_before - set(parsed_by_code)),
        "codes_parsed_only": sorted(
            set(parsed_by_code) - llm_codes_before
        ),
        "reclassified": reclassified,
        "invented_fields": invented_fields,
        "till_mismatches": till_mismatches,
        "amount_mismatches": amount_mismatches,
        "duplicate_codes": duplicate_codes,
        "consistent": consistent and not till_mismatches and not amount_mismatches,
    }


def _chat_till_numbers(messages: list[dict]) -> list[str]:
    tills: list[str] = []
    till_re = re.compile(r"(?:till|buy\s*goods)\s*(?:no\.?|number)?\s*[:#]?\s*(\d{4,})", re.IGNORECASE)
    paybill_re = re.compile(r"paybill\s*(?:no\.?|number)?\s*[:#]?\s*(\d{4,})", re.IGNORECASE)
    for m in messages:
        text = m.get("text") or ""
        for mm in till_re.finditer(text):
            if mm.group(1) not in tills:
                tills.append(mm.group(1))
        for mm in paybill_re.finditer(text):
            if mm.group(1) not in tills:
                tills.append(mm.group(1))
    return tills


# ---------------------------------------------------------------------------
# Element status enforcement
# ---------------------------------------------------------------------------

def check_element_statuses(result: AnalystResult) -> list[dict[str, Any]]:
    """Enforce per-element status rules and the no-evidence downgrade."""
    issues: list[dict[str, Any]] = []
    for key, element in (result.elements or {}).items():
        status = element.status
        if status is ElementStatus.NO_ISSUE_INDICATED and key not in _NO_ISSUE_ELEMENTS:
            element.status = ElementStatus.UNCLEAR
            issues.append({"element": key, "issue": "invalid_no_issue_indicated"})
        if status is ElementStatus.NOT_REQUIRED and key not in _NOT_REQUIRED_ELEMENTS:
            element.status = ElementStatus.UNCLEAR
            issues.append({"element": key, "issue": "invalid_not_required"})
        if status in (ElementStatus.PRESENT, ElementStatus.PARTIAL) and not element.evidence:
            element.status = ElementStatus.UNCLEAR
            issues.append({"element": key, "issue": "no_evidence"})
    return issues


# ---------------------------------------------------------------------------
# Timeline referencing
# ---------------------------------------------------------------------------

def check_timeline(result: AnalystResult) -> list[dict[str, Any]]:
    """Flag timeline entries with neither msg_index nor evidence_id."""
    unreferenced: list[dict[str, Any]] = []
    for i, entry in enumerate(result.timeline or []):
        if entry.msg_index is None and entry.evidence_id is None:
            unreferenced.append({"index": i, "event": entry.event})
    return unreferenced


# ---------------------------------------------------------------------------
# Verdict rule
# ---------------------------------------------------------------------------

def compute_verdict(
    result: AnalystResult,
    integrity_flags: list[str],
    date_checks: dict[str, Any],
) -> tuple[OverallVerdict, bool, list[str]]:
    """Return (effective_verdict, downgraded, reasons)."""
    original = result.overall_verdict
    verdict = original
    reasons: list[str] = []

    integrity = result.conversation_integrity
    if integrity and integrity.rating.value == "WEAK":
        reasons.append("record_integrity_weak")
    if date_checks.get("no_dates"):
        reasons.append("missing_dates")
    if "edited_messages" in integrity_flags:
        reasons.append("edited_messages")

    # LIKELY VALID gate: C,D,E,F,G must be PRESENT, A at least PARTIAL, no ABSENT.
    if original is OverallVerdict.LIKELY_VALID:
        elements = result.elements or {}
        statuses = {k: (elements[k].status if k in elements else None) for k in "ABCDEFGHIJK"}

        missing_present = [k for k in _REQUIRED_PRESENT if statuses.get(k) is not ElementStatus.PRESENT]
        if missing_present:
            reasons.append(f"elements_not_present:{','.join(missing_present)}")
        if statuses.get("A") not in (ElementStatus.PRESENT, ElementStatus.PARTIAL):
            reasons.append("element_a_not_partial")
        absent = [k for k, s in statuses.items() if s is ElementStatus.ABSENT]
        if absent:
            reasons.append(f"elements_absent:{','.join(absent)}")
        if date_checks.get("no_dates"):
            reasons.append("no_dates")

    # A missing date forbids a definitive LIKELY VALID.
    if verdict is OverallVerdict.LIKELY_VALID and reasons:
        verdict = OverallVerdict.POSSIBLY_VALID_WITH_GAPS

    return verdict, verdict is not original, reasons


# ---------------------------------------------------------------------------
# Premature breach
# ---------------------------------------------------------------------------

_BREACH_HINTS = re.compile(
    r"\b(breach|non[- ]?delivery|failed to deliver|did not deliver|not performed|refused)\b",
    re.IGNORECASE,
)


def check_premature_breach(
    result: AnalystResult,
    date_checks: dict[str, Any],
) -> dict[str, Any] | None:
    """If performance is not yet due but K claims breach, reset K to UNCLEAR."""
    perf = date_checks.get("performance_due") or {}
    if perf.get("status") != "not_yet_due":
        return None
    k = (result.elements or {}).get("K")
    summary_asserts = bool(_BREACH_HINTS.search(result.plain_summary or ""))
    if k is None and not summary_asserts:
        return None
    if k is not None and k.status not in (ElementStatus.PRESENT, ElementStatus.ABSENT) and not summary_asserts:
        return None
    if k is not None:
        k.status = ElementStatus.UNCLEAR
    notice = (
        f"Performance is not due until {perf.get('date')}"
        if perf.get("date")
        else "Performance is not yet due"
    )
    return {"premature_breach_claim": True, "notice": notice}


# ---------------------------------------------------------------------------
# Deterministic follow-ups
# ---------------------------------------------------------------------------

def _answered_ids(case_file: dict[str, Any]) -> set[str]:
    answered: set[str] = set()
    for f in case_file.get("user_provided_facts") or []:
        if f.get("answers_request_id"):
            answered.add(str(f["answers_request_id"]))
    for r in case_file.get("requests") or []:
        if r.get("status") in ("answered", "dismissed"):
            answered.add(str(r.get("id")))
    return answered


def build_deterministic_followups(
    analysis_input: dict[str, Any],
    case_file: dict[str, Any],
) -> list[FollowUpRequest]:
    requests: list[FollowUpRequest] = []
    answered = _answered_ids(case_file)
    date_checks = analysis_input.get("date_checks") or {}
    integrity_flags = analysis_input.get("integrity_flags") or []
    evidence = case_file.get("evidence") or []
    messages = analysis_input.get("messages") or []
    payment_hints = analysis_input.get("payment_hints") or []
    mpesa_extracted = analysis_input.get("mpesa_extracted") or []

    def add(req: FollowUpRequest):
        if req.id not in answered and req.id not in {r.id for r in requests}:
            requests.append(req)

    # Missing conversation date.
    if date_checks.get("no_dates"):
        for ev in evidence:
            if ev.get("type") == "chat_screenshot":
                add(FollowUpRequest(
                    id=f"provide_date:E:{ev.get('evidence_id')}",
                    element="E",
                    type="provide_date",
                    prompt="What date did this conversation happen?",
                    why_it_matters="Without a date the timeline is relative only and the record cannot be rated strong.",
                    blocking=True,
                    target=ev.get("evidence_id"),
                    date_role="conversation",
                ))

    # Payment claimed or details given but no M-Pesa confirmation evidence.
    claimed_or_details = any(
        (h.get("claimed") or h.get("details_given")) for h in payment_hints
    )
    has_confirmation = bool(mpesa_extracted)
    if claimed_or_details and not has_confirmation:
        add(FollowUpRequest(
            id="upload_mpesa_screenshot:K:payment",
            element="K",
            type="upload_mpesa_screenshot",
            prompt="Upload the M-Pesa confirmation for the payment.",
            why_it_matters="A chat claim of payment is not evidence; a confirmation could change the verdict.",
            blocking=True,
            target="payment",
        ))

    # Edited messages.
    for m in messages:
        if m.get("edited"):
            add(FollowUpRequest(
                id=f"provide_original_message:C:{m.get('index')}",
                element="C",
                type="provide_original_message",
                prompt=f"Provide the original wording of message {m.get('index')}.",
                why_it_matters="The original wording of an edited message is unknown and may change the terms.",
                blocking=False,
                target=str(m.get("index")),
            ))

    # Media placeholders.
    if "media_placeholder" in integrity_flags:
        add(FollowUpRequest(
            id="upload_other_evidence:K:media",
            element="K",
            type="upload_other_evidence",
            prompt="Upload any media (voice note, image, document) that was referenced.",
            why_it_matters="Referenced media may contain terms or performance evidence.",
            blocking=False,
            target="media",
        ))

    # Screenshot boundary / conversation starts mid-way.
    if "screenshot_boundary" in integrity_flags:
        add(FollowUpRequest(
            id="upload_chat_screenshot:G:earlier_or_later",
            element="G",
            type="upload_chat_screenshot",
            prompt="Upload earlier or later screenshots of this conversation if you have them.",
            why_it_matters="Missing parts of the conversation may hide agreed terms.",
            blocking=False,
            target="earlier_or_later",
        ))

    return requests


_UPLOAD_TYPE_TO_EVIDENCE_TYPE: dict[str, str] = {
    "upload_mpesa_screenshot": "mpesa_screenshot",
    "upload_chat_screenshot":  "chat_screenshot",
}


def merge_followups(
    result: AnalystResult,
    analysis_input: dict[str, Any],
    case_file: dict[str, Any],
) -> list[FollowUpRequest]:
    """Merge deterministic requests with the LLM's, deduplicated and capped."""
    deterministic = build_deterministic_followups(analysis_input, case_file)
    answered = _answered_ids(case_file)

    # Build set of evidence types already in the case file so we can suppress
    # upload requests for evidence the user has already provided.
    existing_evidence_types = {
        (ev.get("type") or "") for ev in (case_file.get("evidence") or [])
    }

    merged: list[FollowUpRequest] = list(deterministic)
    seen = {r.id for r in merged}
    for req in result.follow_up_requests or []:
        if req.id in answered or req.id in seen:
            continue
        # Suppress upload requests for evidence types already uploaded.
        mapped = _UPLOAD_TYPE_TO_EVIDENCE_TYPE.get(req.type)
        if mapped and mapped in existing_evidence_types:
            continue
        merged.append(req)
        seen.add(req.id)

    # Blocking first, then stable by id; cap at 6.
    merged.sort(key=lambda r: (not r.blocking, r.id))
    return merged[:6]
