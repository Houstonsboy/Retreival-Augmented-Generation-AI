"""Deterministic verification rules for the analyst result.

These rules double-check the LLM against the (masked) data it was shown and the
case file. They are code-only and never call the LLM:

- quote grounding (every evidence quote must trace to a message)
- M-Pesa cross-matching, reclassification and cross-evidence checks
- element status enforcement and no-evidence downgrade
- timeline referencing
- conservative verdict rule
- premature-breach detection
- deterministic follow-up request generation and merging

v2 additions (Phases 2–9 of SCCDeterminism.md):
- derive_statuses: derive A–K statuses from ContractFacts
- reconcile: merge call 1 and call 2 ContractFacts
- compute_integrity_rating: deterministic from flags + LLM concerns
- compute_confidence: deterministic from statuses
- deterministic_verdict: code-only verdict
- cert_eligibility: backend eligibility gate (replaces frontend 83% threshold)
- timestamp_bonus: additive bonus for visible timestamps
- check_refs_grounded: msg_index-based grounding check

All rules operate on the *masked* ``analysis_input`` so masking never causes
false quote mismatches.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel

from .models import (
    AnalystResult,
    ElementStatus,
    FollowUpRequest,
    OverallVerdict,
    PaymentClassification,
)


# ---------------------------------------------------------------------------
# M-Pesa finding model (Part B)
# ---------------------------------------------------------------------------

_BLOCKING_MPESA_KINDS = frozenset({
    "DUPLICATE_CONFLICT", "LLM_INVENTED_CODE", "LLM_INVENTED_FIELD",
})


class MpesaFinding(BaseModel):
    code: str | None = None
    kind: Literal[
        "DUPLICATE_SAME_TXN", "DUPLICATE_CONFLICT", "LLM_INVENTED_CODE",
        "LLM_INVENTED_FIELD", "PARSER_ONLY", "UNEXPLAINED_AMOUNT", "COUNTERPARTY_MISMATCH",
    ]
    amount_kes: int | None = None
    detail: str
    blocks_certificate: bool


# ---------------------------------------------------------------------------
# Certificate confidence floor (Part A Section 2)
# ---------------------------------------------------------------------------

CERT_CONFIDENCE_FLOOR = 0.70

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

def check_mpesa_cross_match(
    facts: Any,
    analysis_input: dict[str, Any],
) -> dict[str, Any]:
    """Categorised M-Pesa cross-check. Parser is source of truth.

    Only fabrication findings (LLM_INVENTED_CODE, LLM_INVENTED_FIELD,
    DUPLICATE_CONFLICT) block the certificate. All others are non-blocking.
    Returns a dict with findings list plus legacy-compat "consistent" key.
    """
    parsed_list: list[dict] = analysis_input.get("mpesa_extracted") or []
    # Build code → list[entry] from parser (preserving duplicates)
    parsed_by_code: dict[str, list[dict]] = {}
    for p in parsed_list:
        code = (p.get("code") or "").upper()
        if code:
            parsed_by_code.setdefault(code, []).append(p)

    facts_dict = facts if isinstance(facts, dict) else (
        facts.model_dump() if hasattr(facts, "model_dump") else {}
    )
    llm_payments = facts_dict.get("payments") or []

    findings: list[MpesaFinding] = []
    llm_codes: set[str] = set()

    # 1. Validate every code the LLM cited against parser output
    for pf in llm_payments:
        if isinstance(pf, dict):
            code = (pf.get("mpesa_code") or "").upper()
            amount = pf.get("amount")
        else:
            code = (getattr(pf, "mpesa_code", None) or "").upper()
            amount = getattr(pf, "amount", None)

        if not code:
            continue
        llm_codes.add(code)

        if code not in parsed_by_code:
            findings.append(MpesaFinding(
                code=code,
                kind="LLM_INVENTED_CODE",
                amount_kes=amount,
                detail=f"Code {code} not found in any parsed M-Pesa screenshot",
                blocks_certificate=True,
            ))
        elif amount is not None:
            parser_amounts = [
                e.get("amount_kes") for e in parsed_by_code[code]
                if e.get("amount_kes") is not None
            ]
            if parser_amounts and not any(
                abs(float(a) - float(amount)) <= _AMOUNT_TOL for a in parser_amounts
            ):
                findings.append(MpesaFinding(
                    code=code,
                    kind="LLM_INVENTED_FIELD",
                    amount_kes=amount,
                    detail=f"Code {code}: LLM says KES {amount}, parser found {parser_amounts}",
                    blocks_certificate=True,
                ))

    # 2. Duplicate codes in parser screenshots
    for code, entries in parsed_by_code.items():
        if len(entries) <= 1:
            continue
        amounts = [e.get("amount_kes") for e in entries if e.get("amount_kes") is not None]
        unique_amounts = set(float(a) for a in amounts)
        if len(unique_amounts) <= 1:
            findings.append(MpesaFinding(
                code=code,
                kind="DUPLICATE_SAME_TXN",
                amount_kes=int(amounts[0]) if amounts else None,
                detail=f"Code {code} appears {len(entries)}x in screenshots (same amount — likely duplicate photo)",
                blocks_certificate=False,
            ))
        else:
            findings.append(MpesaFinding(
                code=code,
                kind="DUPLICATE_CONFLICT",
                amount_kes=None,
                detail=f"Code {code} appears with conflicting amounts: {amounts}",
                blocks_certificate=True,
            ))

    # 3. Codes parser found but LLM never cited (non-blocking — LLM just missed)
    for code in sorted(set(parsed_by_code) - llm_codes):
        entries = parsed_by_code[code]
        amt = entries[0].get("amount_kes") if entries else None
        findings.append(MpesaFinding(
            code=code,
            kind="PARSER_ONLY",
            amount_kes=int(amt) if amt is not None else None,
            detail=f"Code {code} in screenshots but not cited by LLM — may be extra payment",
            blocks_certificate=False,
        ))

    # 4. Amount in screenshots not explained by any chat mention
    mentioned = {
        a.get("amount") for a in (analysis_input.get("mentioned_amounts") or [])
        if a.get("amount") is not None
    }
    for p in parsed_list:
        amt = p.get("amount_kes")
        if amt is None:
            continue
        if not any(abs(float(m) - float(amt)) <= _AMOUNT_TOL for m in mentioned):
            findings.append(MpesaFinding(
                code=p.get("code"),
                kind="UNEXPLAINED_AMOUNT",
                amount_kes=int(amt),
                detail=f"Parser found KES {amt} in screenshots but this amount is not mentioned in conversation",
                blocks_certificate=False,
            ))

    # 5. Till/paybill counterparty mismatch
    chat_tills = _chat_till_numbers(analysis_input.get("messages") or [])
    if chat_tills:
        for p in parsed_list:
            till = p.get("till_or_paybill")
            if till and till not in chat_tills:
                findings.append(MpesaFinding(
                    code=p.get("code"),
                    kind="COUNTERPARTY_MISMATCH",
                    amount_kes=p.get("amount_kes"),
                    detail=f"Screenshot till/paybill {till} differs from chat mentions {chat_tills}",
                    blocks_certificate=False,
                ))

    blocking = [f for f in findings if f.blocks_certificate]
    parsed_code_set = set(parsed_by_code)

    return {
        "findings": [f.model_dump() for f in findings],
        "blocks_certificate": bool(blocking),
        # Legacy compat
        "consistent": not bool(blocking),
        "codes_llm_only": sorted(llm_codes - parsed_code_set),
        "codes_parsed_only": sorted(parsed_code_set - llm_codes),
        "llm_payment_count": len(llm_payments),
        "parsed_payment_count": len(parsed_list),
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
    """Return (effective_verdict, downgraded, reasons).

    NOTE: missing dates are NOT a downgrade trigger (Phase 2d). Only integrity
    WEAK and edited_messages can downgrade LIKELY VALID → POSSIBLY VALID WITH GAPS.
    """
    original = result.overall_verdict
    verdict = original
    reasons: list[str] = []

    integrity = result.conversation_integrity
    if integrity and integrity.rating.value == "WEAK":
        reasons.append("record_integrity_weak")
    # Phase 2d: no_dates is NOT a downgrade; removed that trigger.
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
        # Phase 2d: removed no_dates trigger here

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
    """If performance is not yet due but K claims breach, reset K to UNCLEAR.

    Phase 2d: only act when the due date AND a reference date are BOTH resolvable.
    When dates are missing, leave K to the normal derivation; do not force UNCLEAR.
    """
    perf = date_checks.get("performance_due") or {}
    if perf.get("status") != "not_yet_due":
        return None
    # Phase 2d: require a resolvable due date before penalising.
    if not perf.get("date"):
        return None
    k = (result.elements or {}).get("K")
    summary_asserts = bool(_BREACH_HINTS.search(result.plain_summary or ""))
    if k is None and not summary_asserts:
        return None
    if k is not None and k.status not in (ElementStatus.PRESENT, ElementStatus.ABSENT) and not summary_asserts:
        return None
    if k is not None:
        k.status = ElementStatus.UNCLEAR
    notice = f"Performance is not due until {perf['date']}"
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


# ===========================================================================
# Phase 2c — Timestamp bonus (additive only, never a penalty)
# ===========================================================================

TIMESTAMP_BONUS_MAX = 0.06
_TS_SCORE = {"FULL": 1.0, "DATE_ONLY": 0.8, "RELATIVE": 0.5, "TIME_ONLY": 0.3, "NONE": 0.0}


def _chronology_conflict(
    key_refs: list[int],
    messages: list[dict[str, Any]],
    mpesa_events: list[dict[str, Any]],
) -> bool:
    """True only when two resolvable absolute dates contradict logical order."""
    resolved: list[tuple[int, str]] = []
    msg_by_idx = {m.get("index") or m.get("msg_index"): m for m in messages}
    mpesa_by_idx = {p.get("msg_index"): p for p in mpesa_events if p.get("msg_index")}

    for idx in key_refs:
        m = msg_by_idx.get(idx) or mpesa_by_idx.get(idx)
        if not m:
            continue
        tq = m.get("timestamp_quality") or "NONE"
        dt = m.get("resolved_date") or m.get("datetime")
        if tq in ("FULL", "DATE_ONLY") and dt:
            resolved.append((idx, str(dt)))

    if len(resolved) < 2:
        return False
    # Check if dates are in reverse order relative to msg_index order.
    for i in range(len(resolved) - 1):
        idx_a, dt_a = resolved[i]
        idx_b, dt_b = resolved[i + 1]
        if idx_a < idx_b and dt_a > dt_b:
            return True
    return False


def timestamp_bonus(
    key_event_refs: list[int],
    messages: list[dict[str, Any]],
    mpesa_events: list[dict[str, Any]] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute additive timestamp bonus for visible, consistent timestamps.

    key_event_refs: msg_index of offer, acceptance, first payment, first
    performance/breach event (whichever exist in the reconciled facts).
    Returns (bonus, {coverage, chronology_conflict}).
    """
    if not key_event_refs:
        return 0.0, {"coverage": 0.0, "chronology_conflict": False}

    msg_by_idx = {m.get("index") or m.get("msg_index"): m for m in messages}
    mpesa_by_idx = {
        p.get("msg_index"): p for p in (mpesa_events or []) if p.get("msg_index")
    }
    scores = []
    for idx in key_event_refs:
        m = msg_by_idx.get(idx)
        tq = (m.get("timestamp_quality") or "NONE") if m else "NONE"
        # M-Pesa parser entries with a parsed datetime always count as FULL
        if tq == "NONE" and idx in mpesa_by_idx:
            if mpesa_by_idx[idx].get("datetime"):
                tq = "FULL"
        scores.append(_TS_SCORE.get(tq, 0.0))

    coverage = round(sum(scores) / len(scores), 4)
    conflict = _chronology_conflict(key_event_refs, messages, mpesa_events or [])
    bonus = 0.0 if conflict else round(TIMESTAMP_BONUS_MAX * coverage, 4)
    return bonus, {"coverage": coverage, "chronology_conflict": conflict}


# ===========================================================================
# Phase 3 — msg_index-based reference grounding
# ===========================================================================

def check_refs_grounded(
    facts: Any,
    messages: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    """Check that all Ref objects in ContractFacts point to valid msg_index values.

    Returns (all_grounded, ungrounded_list).
    If rapidfuzz is available, also validates span text (soft: bad span is
    noted but does not make the ref ungrounded).
    """
    valid_indices = {
        m.get("index") or m.get("msg_index")
        for m in messages
        if m.get("index") or m.get("msg_index")
    }
    msg_text = {
        m.get("index") or m.get("msg_index"): (m.get("text") or "")
        for m in messages
    }

    try:
        from rapidfuzz.fuzz import partial_ratio as _pr
        def _span_ok(span: str, idx: int) -> bool:
            text = msg_text.get(idx, "")
            return _pr(span.lower(), text.lower()) >= 90
    except ImportError:
        def _span_ok(span: str, idx: int) -> bool:  # type: ignore[misc]
            return (span or "").lower() in (msg_text.get(idx) or "").lower()

    ungrounded: list[dict[str, Any]] = []
    facts_dict = facts if isinstance(facts, dict) else (
        facts.model_dump() if hasattr(facts, "model_dump") else {}
    )

    def _check_ref(ref_obj: Any, field: str) -> None:
        if ref_obj is None:
            return
        if isinstance(ref_obj, dict):
            idx = ref_obj.get("msg_index")
            span = ref_obj.get("span")
        elif hasattr(ref_obj, "msg_index"):
            idx = ref_obj.msg_index
            span = getattr(ref_obj, "span", None)
        else:
            return
        if idx not in valid_indices:
            ungrounded.append({"field": field, "msg_index": idx, "reason": "index_not_found"})
            return
        if span and not _span_ok(span, idx):
            # Soft failure: span mismatch — informational only.
            ungrounded.append({"field": field, "msg_index": idx, "reason": "span_mismatch", "severity": "info"})

    for field in ("offer", "acceptance", "price_ref", "subject_matter_ref",
                  "quantity_or_scope_ref", "due_date_ref"):
        _check_ref(facts_dict.get(field), field)

    for lst_field in ("performance_refs", "breach_refs", "capacity_concern_refs",
                      "consent_concern_refs", "illegality_concern_refs",
                      "scope_dispute_refs", "counter_allegation_refs", "amount_msgs"):
        for ref in facts_dict.get(lst_field) or []:
            _check_ref(ref, lst_field)

    for p in facts_dict.get("payments") or []:
        if isinstance(p, dict):
            _check_ref(p.get("ref"), "payments.ref")
        elif hasattr(p, "ref"):
            _check_ref(p.ref, "payments.ref")

    # Hard failures only (index_not_found)
    hard_failures = [u for u in ungrounded if u.get("reason") != "span_mismatch"]
    return len(hard_failures) == 0, ungrounded


# ===========================================================================
# Phase 5b — Derive element statuses from ContractFacts
# ===========================================================================

def _derive_A(f: dict, statements: dict | None) -> tuple[str, str]:
    """Return (status, basis) for element A — Party identifiability.

    Two routes to PRESENT:
    1. Both parties identifiable from conversation messages.
    2. Claimant declared their name in statements AND respondent is identifiable
       from conversation messages.
    """
    party_refs = f.get("party_identity_refs") or {}
    named = sum(1 for p in ("PARTY_1", "PARTY_2") if party_refs.get(p))
    if named == 2:
        return "PRESENT", "both_in_evidence"

    claimant_declared = bool((statements or {}).get("claimant_full_name"))
    respondent_party = f.get("respondent", "UNCLEAR")
    respondent_in_evidence = (
        respondent_party != "UNCLEAR" and bool(party_refs.get(respondent_party))
    )

    if named == 1 and claimant_declared and respondent_in_evidence:
        return "PRESENT", "declared+evidence"
    if named == 1 or claimant_declared:
        return "PARTIAL", "one_identified"
    return "UNCLEAR", "none_identified"


def derive_statuses(
    facts: Any,
    mpesa: list[dict[str, Any]] | None,
    date_info: dict[str, Any] | None,
    statements: dict | None = None,
) -> dict[str, str]:
    """Derive A–K element statuses deterministically from ContractFacts.

    Legal basis docstrings are inline. These rules are flagged for lawyer
    review in the PR (K-ABSENT and F commercial presumption decisions).
    """
    f = facts if isinstance(facts, dict) else (
        facts.model_dump() if hasattr(facts, "model_dump") else {}
    )
    date_info = date_info or {}
    s: dict[str, str] = {}

    # A — Parties (Kenyan Contract Act s.2: parties must be identifiable)
    s["A"], _ = _derive_A(f, statements)

    # B — Capacity (Cap 23 s.11: minors and persons of unsound mind cannot contract)
    s["B"] = "UNCLEAR" if f.get("capacity_concern_refs") else "NO_ISSUE_INDICATED"

    # C — Offer (requires communication of definite terms + intention to be bound)
    s["C"] = "PRESENT" if f.get("offer") else "ABSENT"

    # D — Acceptance (must be unconditional mirror of offer, by the other party)
    offer = f.get("offer")
    acceptance = f.get("acceptance")
    offer_by = f.get("offer_by", "UNCLEAR")
    acceptance_by = f.get("acceptance_by", "UNCLEAR")
    if acceptance and offer:
        offer_idx = (offer.get("msg_index") if isinstance(offer, dict)
                     else getattr(offer, "msg_index", None))
        acc_idx = (acceptance.get("msg_index") if isinstance(acceptance, dict)
                   else getattr(acceptance, "msg_index", None))
        ok = (
            acc_idx is not None and offer_idx is not None
            and acc_idx > offer_idx
            and acceptance_by != offer_by
            and "UNCLEAR" not in (acceptance_by, offer_by)
        )
        s["D"] = "PRESENT" if ok else "PARTIAL"
    else:
        s["D"] = "PARTIAL" if acceptance else "ABSENT"

    # E — Consideration (Cap 23 s.2: something of value moving from promisee)
    has_price_ref = bool(f.get("price_ref"))
    payments = f.get("payments") or []
    has_payment_ref = any(
        (p.get("ref") if isinstance(p, dict) else getattr(p, "ref", None))
        for p in payments
    )
    s["E"] = "PRESENT" if (has_price_ref or has_payment_ref) else "ABSENT"

    # F — Intention to create legal relations (commercial presumption; Balfour v Balfour)
    if f.get("social_or_domestic_context"):
        s["F"] = "UNCLEAR"
    else:
        s["F"] = "PRESENT" if s["E"] == "PRESENT" else "UNCLEAR"

    # G — Certainty of terms (May & Butcher v R; essential terms must be definite)
    # scope_dispute_refs: extra-charge demands or contested scope prevent PRESENT
    # even when price_ref + subject_matter_ref both exist.
    core_g = sum(x is not None for x in (f.get("price_ref"), f.get("subject_matter_ref")))
    scope_dispute_refs = f.get("scope_dispute_refs") or []
    if scope_dispute_refs:
        s["G"] = "PARTIAL" if core_g > 0 else "ABSENT"
    else:
        s["G"] = {2: "PRESENT", 1: "PARTIAL", 0: "ABSENT"}[core_g]

    # H — Genuine consent (absence of misrepresentation, duress, undue influence)
    s["H"] = "UNCLEAR" if f.get("consent_concern_refs") else "NO_ISSUE_INDICATED"

    # I — Legality (void contracts: illegal purpose, unlicensed trade)
    s["I"] = "UNCLEAR" if f.get("illegality_concern_refs") else "NO_ISSUE_INDICATED"

    # J — Formalities (Land Control Act; Hire Purchase Act require writing)
    s["J"] = "UNCLEAR" if f.get("formality_required") else "NOT_REQUIRED"

    # K — Performance & breach (LAWYER REVIEW: K-ABSENT does not block LIKELY VALID)
    # breach_refs must only contain respondent-failed-to-perform messages.
    # Counter-allegations (respondent accusing claimant of non-payment) go in
    # counter_allegation_refs and do NOT set K=PRESENT.
    breach_refs = f.get("breach_refs") or []
    performance_refs = f.get("performance_refs") or []
    if breach_refs:
        s["K"] = "PRESENT"
    elif performance_refs:
        s["K"] = "PARTIAL"
    else:
        s["K"] = "UNCLEAR"

    # Phase 2d: premature-breach rule only when BOTH due date and ref date are resolvable.
    if date_info.get("due_date_resolved") and date_info.get("performance_not_yet_due"):
        s["K"] = "UNCLEAR"

    return s


# ===========================================================================
# Phase 5a — Reconcile call 1 and call 2 ContractFacts
# ===========================================================================

_LIST_FIELDS = frozenset({
    "performance_refs", "breach_refs", "payments", "capacity_concern_refs",
    "consent_concern_refs", "illegality_concern_refs", "party_identity_refs",
    "scope_dispute_refs", "counter_allegation_refs", "amount_msgs",
})

_VERIFY_FIELDS = frozenset({
    "offer", "offer_by", "acceptance", "acceptance_by",
    "price_ref", "price_amount", "subject_matter_ref", "payments",
    "scope_dispute_refs",
})


def _canon(value: Any) -> Any:
    """Canonical form for comparison: Ref → msg_index; list → sorted tuple."""
    if value is None:
        return None
    if isinstance(value, dict):
        if "msg_index" in value:
            return value["msg_index"]
        return tuple(sorted(str(value)))
    if isinstance(value, list):
        def _item_key(v: Any) -> Any:
            if isinstance(v, dict):
                return (v.get("msg_index"), v.get("amount"), v.get("classification"))
            return str(v)
        return tuple(sorted(_item_key(v) for v in value))
    if hasattr(value, "msg_index"):
        return value.msg_index
    return value


def reconcile(
    c1: dict[str, Any],
    c2: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, set[str]]:
    """Merge call 1 and call 2 facts.

    Returns (final_facts, alt_facts, unstable_fields).
    alt_facts = c1 with c2's values for the verified fields (used only
    for confidence averaging). If call 2 didn't run, alt_facts is None.
    """
    if c2 is None:
        return c1, None, set()

    import copy
    final = copy.deepcopy(c1)
    alt = copy.deepcopy(c1)
    unstable: set[str] = set()

    for field in _VERIFY_FIELDS:
        v2 = c2.get(field)
        if v2 is None:
            continue
        alt[field] = v2
        v1 = c1.get(field)
        if _canon(v1) != _canon(v2):
            unstable.add(field)
            # Neutralise the field in final_facts
            if field in _LIST_FIELDS:
                final[field] = []
            else:
                final[field] = None

    return final, alt, unstable


# ===========================================================================
# Phase 6 — Deterministic integrity rating
# ===========================================================================

def compute_integrity_rating(
    flags: list[str] | set[str],
    integrity_concerns: list[Any] | None,
) -> str:
    """Compute STRONG/ADEQUATE/WEAK purely from code-detected flags and grounded concerns.

    No date-based rule (Phase 2d). LLM concerns only count after grounding.
    """
    flag_set = set(flags) if flags else set()
    concerns = integrity_concerns or []
    cats = set()
    for c in concerns:
        cat = (c.get("category") if isinstance(c, dict)
               else getattr(c, "category", None))
        if cat:
            cats.add(cat)

    if {"deleted_messages", "low_ocr_confidence"} & flag_set or "POSSIBLE_FABRICATION" in cats:
        return "WEAK"
    if ({"edited_messages", "screenshot_boundary", "gap_detected", "chronology_conflict"} & flag_set
            or cats & {"EDIT_INDICATOR", "DELETION_INDICATOR", "INCONSISTENT_ACCOUNT"}):
        return "ADEQUATE"
    return "STRONG"


# ===========================================================================
# Phase 7 — Deterministic confidence (display only, NOT a gate)
# ===========================================================================

ELEMENT_WEIGHTS = {
    "C": .18, "D": .18, "E": .14, "G": .12, "F": .10,
    "A": .10, "K": .10, "B": .02, "H": .02, "I": .02, "J": .02,
}
STATUS_SCORES = {
    "PRESENT": 1.0, "NO_ISSUE_INDICATED": 1.0, "NOT_REQUIRED": 1.0,
    "PARTIAL": 0.5, "UNCLEAR": 0.25, "ABSENT": 0.0,
}
INTEGRITY_FACTOR = {"STRONG": 1.0, "ADEQUATE": 0.95, "WEAK": 0.80}


def element_score(statuses: dict[str, str]) -> float:
    return sum(
        ELEMENT_WEIGHTS[k] * STATUS_SCORES.get(statuses.get(k, "UNCLEAR"), 0.25)
        for k in ELEMENT_WEIGHTS
    )


def compute_confidence(
    inspection_statuses: list[dict[str, str]],
    integrity_rating: str,
    ts_bonus: float,
) -> float:
    """Deterministic confidence from element statuses.

    inspection_statuses: [derive_statuses(call1_facts)] or, if call 2 ran,
    [derive_statuses(call1_facts), derive_statuses(alt_facts)].
    Result is display/ranking only — NOT used as a gate.
    """
    if not inspection_statuses:
        return 0.0
    base = sum(element_score(s) for s in inspection_statuses) / len(inspection_statuses)
    factor = INTEGRITY_FACTOR.get(integrity_rating, 1.0)
    return round(min(1.0, base * factor + ts_bonus), 4)


def confidence_breakdown(
    statuses: dict[str, str],
    integrity_rating: str,
    ts_bonus: float,
    ts_info: dict[str, Any],
) -> dict[str, Any]:
    """Return per-element contribution and factors for display."""
    per_element = {
        k: round(ELEMENT_WEIGHTS[k] * STATUS_SCORES.get(statuses.get(k, "UNCLEAR"), 0.25), 4)
        for k in ELEMENT_WEIGHTS
    }
    return {
        "per_element": per_element,
        "element_sum": round(sum(per_element.values()), 4),
        "integrity_rating": integrity_rating,
        "integrity_factor": INTEGRITY_FACTOR.get(integrity_rating, 1.0),
        "timestamp_bonus": ts_bonus,
        "timestamp_coverage": ts_info.get("coverage", 0.0),
        "chronology_conflict": ts_info.get("chronology_conflict", False),
    }


# ===========================================================================
# Phase 8 — Deterministic verdict
# ===========================================================================

_CORE = ("C", "D", "E", "F", "G")


def deterministic_verdict(
    st: dict[str, str],
    integrity: str,
    flags: list[str] | set[str],
) -> str:
    """Compute verdict from derived statuses. No LLM input.

    LAWYER REVIEW: K-ABSENT does NOT make the verdict LIKELY NOT VALID.
    LAWYER REVIEW: F uses a commercial presumption (social_or_domestic → UNCLEAR).
    """
    core = [st.get(k, "UNCLEAR") for k in _CORE]

    if (all(s in ("UNCLEAR", "ABSENT") for s in core)
            and not any(st.get(k) == "ABSENT" for k in ("C", "E"))):
        return "INSUFFICIENT EVIDENCE"

    if (st.get("C") == "ABSENT"
            or st.get("E") == "ABSENT"
            or (st.get("D") == "ABSENT" and st.get("C") == "PRESENT")):
        return "LIKELY NOT VALID"

    flag_set = set(flags) if flags else set()
    if (all(s == "PRESENT" for s in core)
            and st.get("A") in ("PRESENT", "PARTIAL")
            and integrity != "WEAK"
            and "edited_messages" not in flag_set):
        return "LIKELY VALID"

    return "POSSIBLY VALID WITH GAPS"


# ===========================================================================
# Phase 9 — Server-side certificate eligibility
# ===========================================================================

def cert_eligibility(
    verdict: str,
    st: dict[str, str],
    unstable_core: set[str] | list[str],
    integrity: str,
    refs_grounded: bool,
    mpesa_consistent: bool,
    incomplete: bool,
    verified: bool,
    confidence: float | None = None,
    mpesa_findings: list[dict] | None = None,
) -> dict[str, Any]:
    """Compute backend eligibility for a s.106B certificate.

    Replaces the frontend CERT_THRESHOLD = 0.83 gate.
    Returns {
        "eligible": bool,
        "reasons": list[str],
        "system_reason_codes": list[str],  # subset of reasons that are transient
    }.

    system_reason_codes entries are transient failures the user cannot fix by
    uploading more evidence; the UI should show a Retry button for these.
    """
    reasons: list[str] = []
    system_reason_codes: list[str] = []

    def _r(msg: str, *, system: bool = False) -> None:
        reasons.append(msg)
        if system:
            system_reason_codes.append(msg)

    if incomplete:
        _r("analysis incomplete — try again later", system=True)
    if not verified:
        _r("decisive facts not independently verified — needs review", system=True)
    if verdict in ("LIKELY NOT VALID", "INSUFFICIENT EVIDENCE"):
        _r(f"verdict: {verdict}")
    for k in ("C", "D", "E"):
        if st.get(k) != "PRESENT":
            _r(f"element {k} not established ({st.get(k, 'UNCLEAR')})")
    # G must not be ABSENT for a certificate (scope must be minimally defined)
    if st.get("G") == "ABSENT":
        _r("element G absent — contract scope not established")
    unstable = set(unstable_core) if unstable_core else set()
    if unstable:
        _r(f"inconsistent across independent inspections: {sorted(unstable)}")
    if integrity == "WEAK":
        _r("record integrity weak")
    if not refs_grounded:
        _r("ungrounded evidence references", system=True)
    # Only fabrication findings block the cert (non-blocking findings are informational)
    blocking_findings = [
        f for f in (mpesa_findings or [])
        if f.get("blocks_certificate")
    ]
    if blocking_findings or not mpesa_consistent:
        _r("M-Pesa cross-check: fabricated transaction data detected")
    # Confidence floor check — only triggers when no other reason is present
    # and caller supplies an explicit confidence (None = backwards-compat skip)
    if confidence is not None and confidence < CERT_CONFIDENCE_FLOOR and not reasons:
        _r(f"confidence {confidence:.0%} below minimum {CERT_CONFIDENCE_FLOOR:.0%}")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "system_reason_codes": system_reason_codes,
    }


# ===========================================================================
# Confidence band label
# ===========================================================================

def confidence_band(confidence: float) -> str:
    """Return a human-readable band label for a confidence score."""
    if confidence >= 0.85:
        return "HIGH"
    if confidence >= CERT_CONFIDENCE_FLOOR:
        return "MEDIUM"
    if confidence >= 0.50:
        return "LOW"
    return "VERY_LOW"


# ===========================================================================
# Model disagreements helper
# ===========================================================================

def compute_model_disagreements(
    element_opinions: dict[str, Any],
    derived_statuses: dict[str, str],
) -> dict[str, Any]:
    """Find elements where the LLM's advisory opinion differs from derived status."""
    disagreements: dict[str, Any] = {}
    for key, opinion in (element_opinions or {}).items():
        if isinstance(opinion, dict):
            model_status = opinion.get("status")
            reason = opinion.get("reason", "")
        elif hasattr(opinion, "status"):
            model_status = opinion.status
            reason = getattr(opinion, "reason", "")
        else:
            continue
        derived = derived_statuses.get(key)
        if model_status and derived and str(model_status) != str(derived):
            disagreements[key] = {
                "model_status": str(model_status),
                "derived_status": derived,
                "model_reason": reason,
            }
    return disagreements
