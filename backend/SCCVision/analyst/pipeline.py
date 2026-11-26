"""End-to-end contract-validity analysis pipeline (v2 — deterministic).

Flow:
  1. Build canonical analysis_input (candidates, timestamp_quality).
  2. Compute input_hash — check Firestore cache; return snapshot on hit.
  3. Call 1 (inspect) → ContractFacts.
  4. Ground references; run M-Pesa cross-match.
  5. Derive element statuses from facts.
  6. Check call-2 triggers (T1–T4).
  7. If triggered: Call 2 (verify) → reconcile facts → re-derive statuses.
  8. Compute integrity rating, confidence, verdict, cert_eligibility.
  9. Persist snapshot; return analysis response.

Call budget: at most MAX_CALLS=2 Groq calls per analysis (excluding 429 retries).
Cache hit: 0 Groq calls.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .client import AnalystError, call_inspect, call_verify
from .dates import compute_date_checks
from .enrich import build_analysis_input, build_analysis_input_from_case, merge_ocr_results
from .followup_catalog import build_followups
from .rules import (
    cert_eligibility,
    check_mpesa_cross_match,
    check_refs_grounded,
    compute_confidence,
    compute_integrity_rating,
    compute_model_disagreements,
    confidence_band,
    confidence_breakdown,
    deterministic_verdict,
    derive_statuses,
    reconcile,
    timestamp_bonus,
)
from .schema import ContractFacts
from .versions import (
    MAX_CALLS,
    MODEL,
    PROMPT_VERSION,
    RULES_VERSION,
    VERIFY_PROMPT_VERSION,
)

# Lazy import to avoid circular dependency
def _get_cache_store():
    from ..analysis_store import get_analysis_by_hash, save_analysis_snapshot
    return get_analysis_by_hash, save_analysis_snapshot


# ---------------------------------------------------------------------------
# Input hash
# ---------------------------------------------------------------------------

def compute_input_hash(canonical_json: str) -> str:
    salt = f"{PROMPT_VERSION}|{VERIFY_PROMPT_VERSION}|{RULES_VERSION}|{MODEL}"
    return hashlib.sha256((canonical_json + salt).encode("utf-8")).hexdigest()


def _canonical_json(analysis_input: dict[str, Any]) -> str:
    # Sort messages by index for stability; remove non-canonical fields.
    canonical = dict(analysis_input)
    messages = sorted(
        canonical.get("messages") or [],
        key=lambda m: m.get("index") or m.get("msg_index") or 0,
    )
    canonical["messages"] = messages
    return json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Call-2 trigger evaluation
# ---------------------------------------------------------------------------

_CORE_FACTS = ("C", "D", "E", "G")


def _check_triggers(
    provisional_eligibility: bool,
    derived_st: dict[str, str],
    model_disagreements: dict[str, Any],
    ungrounded_core: list[dict[str, Any]],
    verdict: str,
    facts: ContractFacts,
) -> list[str]:
    """Return list of applicable trigger codes (T1–T4). Call 2 runs if non-empty."""
    triggers: list[str] = []

    # T1: provisionally certificate-eligible
    if provisional_eligibility:
        triggers.append("T1")

    # T2: model opinion on core element differs from derived status
    for elem in _CORE_FACTS:
        if elem in model_disagreements:
            triggers.append("T2")
            break

    # T3: a core fact reference failed grounding
    if ungrounded_core:
        triggers.append("T3")

    # T4: verdict is good, but a core element has only one ref
    if verdict in ("LIKELY VALID", "POSSIBLY VALID WITH GAPS"):
        fd = facts.model_dump()
        single_support = {
            "C": bool(fd.get("offer")) and not fd.get("acceptance"),
            "E": bool(fd.get("price_ref")) and not fd.get("payments"),
            "G": sum(x is not None for x in (fd.get("price_ref"), fd.get("subject_matter_ref"))) == 1,
        }
        if any(single_support.values()):
            triggers.append("T4")

    return triggers


# ---------------------------------------------------------------------------
# Build legacy result dict for s106B gate compatibility
# ---------------------------------------------------------------------------

def _build_legacy_result(
    facts_dict: dict[str, Any],
    statuses: dict[str, str],
    integrity_rating: str,
    confidence: float,
    verdict: str,
    plain_summary: str,
    narrative: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the old AnalystResult-shaped dict for the s106B gate and PDF (Fix 4)."""
    narrative = narrative or {}
    elements: dict[str, Any] = {}
    for k, status in statuses.items():
        evidence: list[dict] = []
        for ref_field in ("offer", "acceptance", "price_ref", "subject_matter_ref"):
            ref = facts_dict.get(ref_field)
            if ref and k in _get_element_for_ref(ref_field):
                evidence.append({
                    "msg_index": ref.get("msg_index") if isinstance(ref, dict) else getattr(ref, "msg_index", None),
                    "quote": ref.get("span", "") if isinstance(ref, dict) else getattr(ref, "span", ""),
                    "sender": None, "time": None,
                })
        elements[k] = {
            "status": status,
            "evidence": evidence,
            "reasoning": "",
            "confidence": None,
        }

    # Parties from party_identity_refs
    parties: list[dict] = []
    for party_key, refs in (facts_dict.get("party_identity_refs") or {}).items():
        parties.append({"party": party_key, "refs": refs})

    # Red flags from LLM narrative
    red_flags = list(narrative.get("red_flag_notes") or [])

    # Edited offer is a special red flag (Part F)
    if facts_dict.get("offer") and facts_dict.get("integrity_concerns"):
        for concern in facts_dict.get("integrity_concerns") or []:
            cat = concern.get("category") if isinstance(concern, dict) else getattr(concern, "category", None)
            if cat == "EDIT_INDICATOR":
                offer_idx = None
                ref = facts_dict.get("offer")
                if isinstance(ref, dict):
                    offer_idx = ref.get("msg_index")
                for c_ref in (concern.get("refs") if isinstance(concern, dict) else getattr(concern, "refs", [])) or []:
                    c_idx = c_ref.get("msg_index") if isinstance(c_ref, dict) else getattr(c_ref, "msg_index", None)
                    if c_idx == offer_idx:
                        red_flags.append(f"The offer message (#{offer_idx}) appears to have been edited.")
                        break

    return {
        "conversation_integrity": {"rating": integrity_rating, "issues": []},
        "unrelated_messages": [],
        "parties": parties,
        "timeline": [],
        "elements": elements,
        "mpesa_payments": [],
        "red_flags": red_flags,
        "missing_information": [],
        "overall_verdict": verdict,
        "confidence": confidence,
        "changes_since_previous": [],
        "plain_summary": plain_summary,
        "strengthen_checklist": list(narrative.get("strengthen_suggestions") or []),
        "follow_up_requests": [],
        "questions_for_parties": list(narrative.get("questions_for_parties") or []),
        "disclaimer": (
            "This is an automated analysis, not legal advice. "
            "A qualified advocate should review high-value or contested matters."
        ),
    }


def _get_element_for_ref(field: str) -> set[str]:
    return {
        "offer": {"C"},
        "acceptance": {"D"},
        "price_ref": {"E", "G"},
        "subject_matter_ref": {"G"},
    }.get(field, set())


def _payments_for_cross_check(facts_dict: dict[str, Any]) -> list[dict]:
    out = []
    for p in facts_dict.get("payments") or []:
        if isinstance(p, dict):
            out.append({
                "code": p.get("mpesa_code"),
                "amount_kes": p.get("amount"),
                "sender": p.get("payer"),
                "recipient": None,
                "datetime": None,
                "evidence_id": None,
                "basis": ["confirmation"] if p.get("classification") == "VERIFIED-LOOKING" else [],
                "classification": p.get("classification", "CLAIMED ONLY"),
                "notes": "",
            })
    return out


# ---------------------------------------------------------------------------
# Key event refs for timestamp bonus
# ---------------------------------------------------------------------------

def _key_refs(facts_dict: dict[str, Any]) -> list[int]:
    refs: list[int] = []
    for field in ("offer", "acceptance"):
        ref = facts_dict.get(field)
        if isinstance(ref, dict) and ref.get("msg_index"):
            refs.append(ref["msg_index"])
    for p in facts_dict.get("payments") or []:
        pref = p.get("ref") if isinstance(p, dict) else None
        if pref and isinstance(pref, dict) and pref.get("msg_index"):
            refs.append(pref["msg_index"])
            break
    for p_field in ("performance_refs", "breach_refs"):
        lst = facts_dict.get(p_field) or []
        if lst:
            first = lst[0]
            if isinstance(first, dict) and first.get("msg_index"):
                refs.append(first["msg_index"])
            break
    return refs


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze_case(
    case_file: dict[str, Any],
    *,
    force_rerun: bool = False,
    uid: str | None = None,
    round_num: int | None = None,
    parent_analysis_id: str | None = None,
) -> dict[str, Any]:
    """Run the full deterministic analysis over a case file.

    Returns the analysis response dict. Saves the snapshot to Firestore if uid
    is provided. The snapshot stores input_hash for cache lookup.

    If force_rerun=False and a cached snapshot exists for this input_hash,
    returns the cached snapshot immediately (0 Groq calls).

    round_num: 1-based supplement round (1 = initial, 2 = first supplement, …).
      If None, inferred as len(case_file.analyses) + 1.
    parent_analysis_id: analysis_id of the previous round (for round diff).
    """
    if not case_file or not case_file.get("evidence"):
        raise ValueError("The case file must contain at least one evidence item")

    # Extract case_id and derive round number from the case file
    case_id = case_file.get("case_id") or ""
    _inferred_round = len(case_file.get("analyses") or []) + 1
    actual_round = round_num if round_num is not None else _inferred_round

    date_checks = compute_date_checks(case_file)
    analysis_input = build_analysis_input_from_case(case_file, date_checks)

    canon_json = _canonical_json(analysis_input)
    input_hash = compute_input_hash(canon_json)

    # Cache check
    if not force_rerun and uid:
        try:
            get_by_hash, _ = _get_cache_store()
            cached = get_by_hash(uid, input_hash)
            if cached:
                cached["from_cache"] = True
                return cached
        except Exception as exc:
            print(f"[WARN] cache lookup failed: {exc}")

    # Collect statements typed by the user (e.g. claimant_full_name from A_CLAIMANT_NAME request)
    statements: dict[str, str] = {
        f.get("key"): f.get("value")
        for f in (case_file.get("user_provided_facts") or [])
        if f.get("key") and f.get("value")
    }

    # --- Call 1 ---
    calls_used = 0
    triggers_fired: list[str] = []
    single_inspection = False
    incomplete = False

    try:
        c1_facts, calls_used = call_inspect(analysis_input)
    except AnalystError as exc:
        incomplete = True
        single_inspection = True
        # Build empty facts so the pipeline can still produce a result
        c1_facts = ContractFacts()
        calls_used = min(MAX_CALLS, calls_used + 1)
        print(f"[WARN] Call 1 failed: {exc}")

    c1_dict = c1_facts.model_dump()

    # --- Ref grounding (call 1) ---
    refs_grounded, ungrounded_list = check_refs_grounded(c1_dict, analysis_input.get("messages") or [])
    ungrounded_core = [u for u in ungrounded_list if u.get("reason") == "index_not_found"]

    # --- M-Pesa cross-match (call 1) ---
    try:
        mpesa_cross = check_mpesa_cross_match(c1_dict, analysis_input)
        mpesa_consistent = mpesa_cross.get("consistent", True)
        mpesa_findings = mpesa_cross.get("findings") or []
    except Exception as exc:
        print(f"[WARN] mpesa cross-match failed: {exc}")
        mpesa_cross = {"consistent": True, "findings": []}
        mpesa_consistent = True
        mpesa_findings = []

    # --- Derive statuses (call 1) ---
    date_info = {
        "due_date_resolved": bool(
            (date_checks.get("performance_due") or {}).get("date")
        ),
        "performance_not_yet_due": (
            (date_checks.get("performance_due") or {}).get("status") == "not_yet_due"
        ),
    }
    st1 = derive_statuses(c1_dict, analysis_input.get("mpesa_extracted") or [], date_info, statements=statements)

    # --- Integrity rating (call 1) ---
    integrity_flags = analysis_input.get("integrity_flags") or []
    integrity_rating = compute_integrity_rating(
        integrity_flags,
        c1_dict.get("integrity_concerns") or [],
    )

    # --- Provisional verdict and eligibility (for trigger check) ---
    prov_verdict = deterministic_verdict(st1, integrity_rating, integrity_flags)
    model_disagreements = compute_model_disagreements(
        c1_dict.get("element_opinions") or {}, st1
    )

    prov_cert = cert_eligibility(
        prov_verdict, st1,
        unstable_core=set(),
        integrity=integrity_rating,
        refs_grounded=refs_grounded,
        mpesa_consistent=mpesa_consistent,
        incomplete=incomplete,
        verified=False,
    )

    # --- Check triggers ---
    triggers_fired = _check_triggers(
        provisional_eligibility=prov_cert["eligible"],
        derived_st=st1,
        model_disagreements=model_disagreements,
        ungrounded_core=ungrounded_core,
        verdict=prov_verdict,
        facts=c1_facts,
    ) if not incomplete else []

    # --- Call 2 (conditional) ---
    c2_dict: dict[str, Any] | None = None
    verified = False
    targeted_fields: list[str] = []

    if triggers_fired and calls_used < MAX_CALLS:
        targeted_fields = list({
            "offer", "offer_by", "acceptance", "acceptance_by",
            "price_ref", "price_amount", "subject_matter_ref", "payments",
            "scope_dispute_refs",
        })
        c2_dict_raw, c2_calls = call_verify(analysis_input, targeted_fields)
        calls_used += c2_calls
        if c2_dict_raw is not None:
            c2_dict = c2_dict_raw
            # Call 2 running and returning data = independently verified
            verified = not single_inspection

    # --- Reconcile ---
    final_dict, alt_dict, unstable_fields = reconcile(c1_dict, c2_dict)

    # --- Re-derive statuses on final and alt facts ---
    mpesa_extracted = analysis_input.get("mpesa_extracted") or []
    st_final = derive_statuses(final_dict, mpesa_extracted, date_info, statements=statements)

    inspection_statuses: list[dict[str, str]] = [st1]
    if alt_dict is not None:
        st_alt = derive_statuses(alt_dict, mpesa_extracted, date_info, statements=statements)
        inspection_statuses.append(st_alt)

    # Mark unstable elements as UNCLEAR in final statuses
    unstable_core_elements: set[str] = set()
    field_to_elem = {
        "offer": "C", "acceptance": "D", "offer_by": "D", "acceptance_by": "D",
        "price_ref": "E", "payments": "E", "subject_matter_ref": "G",
        "price_amount": "G", "party_identity_refs": "A",
        "performance_refs": "K", "breach_refs": "K", "due_date_ref": "K",
    }
    for uf in unstable_fields:
        elem = field_to_elem.get(uf)
        if elem:
            st_final[elem] = "UNCLEAR"
            if elem in _CORE_FACTS:
                unstable_core_elements.add(elem)

    # --- Re-compute integrity on final facts (uses call 1 concerns only) ---
    integrity_rating = compute_integrity_rating(
        integrity_flags,
        final_dict.get("integrity_concerns") or [],
    )

    # --- Timestamp bonus ---
    key_refs = _key_refs(final_dict)
    ts_bonus, ts_info = timestamp_bonus(
        key_refs, analysis_input.get("messages") or [], analysis_input.get("mpesa_extracted") or []
    )
    if ts_info.get("chronology_conflict"):
        if "chronology_conflict" not in integrity_flags:
            integrity_flags = list(integrity_flags) + ["chronology_conflict"]
        integrity_rating = compute_integrity_rating(integrity_flags, final_dict.get("integrity_concerns") or [])

    # --- Final confidence ---
    conf = compute_confidence(inspection_statuses, integrity_rating, ts_bonus)
    conf_breakdown = confidence_breakdown(st_final, integrity_rating, ts_bonus, ts_info)

    # --- Final verdict ---
    verdict = deterministic_verdict(st_final, integrity_rating, integrity_flags)

    # --- LLM verdict opinion (from call 1 element_opinions) ---
    llm_verdict_opinion = deterministic_verdict(st1, integrity_rating, integrity_flags)

    # --- Model disagreements (final) ---
    model_disagreements = compute_model_disagreements(
        c1_dict.get("element_opinions") or {}, st_final
    )

    # --- Certificate eligibility ---
    cert = cert_eligibility(
        verdict, st_final,
        unstable_core=unstable_core_elements,
        integrity=integrity_rating,
        refs_grounded=refs_grounded,
        mpesa_consistent=mpesa_consistent,
        incomplete=incomplete,
        verified=verified,
        confidence=conf,
        mpesa_findings=mpesa_findings,
    )

    # --- Guided follow-up requests (deterministic catalog) ---
    # Build a partial snapshot for the catalog — only the fields it reads.
    _partial_snapshot: dict[str, Any] = {
        "derived_statuses": st_final,
        "unstable_fields": sorted(unstable_fields),
        "contract_facts": final_dict,
        "integrity_flags": integrity_flags,
        "timestamp_info": ts_info,
        "cert_eligibility": cert,
        "verification": {
            "mpesa_cross_check": mpesa_cross,
        },
    }
    _messages_for_catalog = analysis_input.get("messages") or []
    try:
        guided_follow_ups = [
            r.model_dump()
            for r in build_followups(_partial_snapshot, case_file, _messages_for_catalog)
        ]
    except Exception as _exc:
        print(f"[WARN] build_followups failed: {_exc}")
        guided_follow_ups = []

    # --- Build narrative (from call 1) ---
    narrative = c1_dict.get("narrative") or {}

    # Append date advice to strengthen suggestions if no timestamps
    if date_checks.get("no_dates") and narrative.get("strengthen_suggestions") is not None:
        date_advice = (
            "No timestamps visible on key messages — adding the dates "
            "(e.g. scrolling to the date header before capturing) would strengthen this evidence."
        )
        sugg = list(narrative.get("strengthen_suggestions") or [])
        if date_advice not in sugg:
            sugg.append(date_advice)
        narrative["strengthen_suggestions"] = sugg

    # plain_summary: LLM narrative if available, else code-generated template (Fix 6)
    llm_summary = (narrative.get("facts_summary") or "").strip()
    if llm_summary:
        plain_summary = llm_summary
    else:
        present_core = [k for k in "CDEFG" if st_final.get(k) == "PRESENT"]
        absent_core = [k for k in "CDEFG" if st_final.get(k) == "ABSENT"]
        parts = []
        price = final_dict.get("price_amount")
        if price:
            parts.append(f"Price: KES {price:,}.")
        if present_core:
            parts.append(f"Contract elements established: {', '.join(present_core)}.")
        if absent_core:
            parts.append(f"Elements not found: {', '.join(absent_core)}.")
        parts.append(f"Verdict: {verdict}. Confidence: {conf:.0%}.")
        plain_summary = " ".join(parts)

    # --- Legacy result dict for s106B gate compatibility ---
    legacy_result = _build_legacy_result(
        final_dict, st_final, integrity_rating, conf, verdict,
        plain_summary,
        narrative=narrative,
    )

    # --- Verification block ---
    verification = {
        "refs_grounded": refs_grounded,
        "quotes_grounded": refs_grounded,  # kept for gate compat
        "ungrounded_refs": ungrounded_list,
        "mpesa_cross_check": mpesa_cross,
        "mpesa_findings": mpesa_findings,
        "unstable_fields": sorted(unstable_fields),
        "triggers_fired": triggers_fired,
        "calls_used": calls_used,
        "verified": verified,
    }

    response = {
        "input_hash": input_hash,
        "prompt_version": PROMPT_VERSION,
        "verify_prompt_version": VERIFY_PROMPT_VERSION,
        "rules_version": RULES_VERSION,
        "model": MODEL,
        "calls_used": calls_used,
        "triggers_fired": triggers_fired,
        "single_inspection": single_inspection,
        "verified": verified,
        "contract_facts": final_dict,
        "unstable_fields": sorted(unstable_fields),
        "derived_statuses": st_final,
        "integrity_flags": integrity_flags,
        "model_disagreements": model_disagreements,
        "integrity_rating": integrity_rating,
        "confidence": conf,
        "confidence_band": confidence_band(conf),
        "confidence_breakdown": conf_breakdown,
        "verdict": verdict,
        "plain_summary": plain_summary,
        "llm_verdict_opinion": llm_verdict_opinion,
        "cert_eligibility": cert,
        "timestamp_info": ts_info,
        "narrative": narrative,
        "mpesa_findings": mpesa_findings,
        # Guided follow-ups (rich deterministic requests)
        "guided_follow_ups": guided_follow_ups,
        # Case tracking fields
        "case_id": case_id,
        "round": actual_round,
        "parent_analysis_id": parent_analysis_id,
        # Legacy fields kept for s106B gate and other consumers
        "result": legacy_result,
        "verification": verification,
        "analysis_input": analysis_input,
    }

    return response


# ---------------------------------------------------------------------------
# Legacy path (deprecated endpoints) — preserved for /sccvision/analyze
# ---------------------------------------------------------------------------

def analyze(
    ocr_results: list[dict[str, Any]],
    case_context: dict[str, Any] | None = None,
    *,
    llm: Any = None,  # kept for signature compat; ignored
) -> dict[str, Any]:
    """Deprecated: run analysis over OCR results (no case file)."""
    if not ocr_results:
        raise ValueError("At least one OCR result is required")

    cleaned = merge_ocr_results(ocr_results)
    image_count = len(ocr_results)
    analysis_input = build_analysis_input(cleaned, case_context, image_count)

    try:
        c1_facts, calls_used = call_inspect(analysis_input)
    except AnalystError as exc:
        raise AnalystError(f"Analysis failed: {exc}") from exc

    c1_dict = c1_facts.model_dump()
    integrity_flags = analysis_input.get("integrity_flags") or []
    st = derive_statuses(c1_dict, [], {})
    integrity_rating = compute_integrity_rating(integrity_flags, c1_dict.get("integrity_concerns") or [])
    verdict = deterministic_verdict(st, integrity_rating, integrity_flags)
    conf = compute_confidence([st], integrity_rating, 0.0)

    return {
        "analysis_input": analysis_input,
        "result": _build_legacy_result(c1_dict, st, integrity_rating, conf, verdict,
                                        (c1_dict.get("narrative") or {}).get("facts_summary") or ""),
        "verification": {"calls_used": calls_used},
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
    }
