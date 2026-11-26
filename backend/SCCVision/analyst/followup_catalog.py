"""Deterministic follow-up request catalog.

`build_followups(snapshot, case_file, messages)` maps every gap in a finished
analysis to a specific, actionable request. The catalog is pure code — no LLM
involvement. The LLM's `questions_for_parties` and `strengthen_suggestions` are
shown separately as "additional tips" (Section 3, principle 1).

Gap precedence: requests in `cert_eligibility.reasons` are priority 1 (block
the certificate). Core elements not PRESENT are priority 2. Everything else is
priority 3. Within a tier, sort by ELEMENT_WEIGHTS descending so the most
impactful gap leads.

Returned list is sorted and capped at MAX_OPEN=6 open requests. Requests
already RESOLVED / UNAVAILABLE / SUPERSEDED in the case_file are suppressed.
"""

from __future__ import annotations

import re
from typing import Any

from .followups import FollowUpRequestV2, InputField, ReqStatus
from .rules import (
    CERT_CONFIDENCE_FLOOR,
    ELEMENT_WEIGHTS,
    INTEGRITY_FACTOR,
    STATUS_SCORES,
)

MAX_OPEN = 6

# Element weight order for sorting within priority tier (matches rules.py)
_ELEM_WEIGHT_ORDER = ["C", "D", "E", "G", "F", "A", "K", "B", "H", "I", "J"]
_ELEM_RANK = {e: i for i, e in enumerate(_ELEM_WEIGHT_ORDER)}

# Statuses that are NOT open
_CLOSED = frozenset({"RESOLVED", "UNAVAILABLE", "SUPERSEDED"})


# ---------------------------------------------------------------------------
# C3 — Simulated expected_effect
# ---------------------------------------------------------------------------

def _simulate_effect(
    snapshot: dict[str, Any],
    resolve_to: dict[str, str],
) -> str:
    """Compute the expected_effect string by simulating status changes.

    resolve_to: element → new status if this request were answered (e.g. {"C": "PRESENT"}).
    Returns a human-readable description of the confidence delta.
    """
    old_st = dict(snapshot.get("derived_statuses") or {})
    new_st = dict(old_st)
    new_st.update(resolve_to)

    integrity = snapshot.get("integrity_rating") or "ADEQUATE"
    factor = INTEGRITY_FACTOR.get(integrity, 1.0)
    ts_bonus = (snapshot.get("timestamp_info") or {}).get("coverage", 0.0) * 0.06

    def _score(st: dict) -> float:
        return sum(
            ELEMENT_WEIGHTS[k] * STATUS_SCORES.get(st.get(k, "UNCLEAR"), 0.25)
            for k in ELEMENT_WEIGHTS
        )

    old_conf = round(min(1.0, _score(old_st) * factor + ts_bonus), 2)
    new_conf = round(min(1.0, _score(new_st) * factor + ts_bonus), 2)
    delta = round(new_conf - old_conf, 2)

    old_elig = new_conf < CERT_CONFIDENCE_FLOOR
    new_elig = new_conf >= CERT_CONFIDENCE_FLOOR

    if new_elig and old_elig:
        return f"Could raise confidence to ~{new_conf:.0%} (+{delta:.0%}) and cross the certificate threshold"
    if delta > 0.05:
        return f"Could raise confidence from ~{old_conf:.0%} to ~{new_conf:.0%} (+{delta:.0%})"
    resolved_names = " and ".join(resolve_to.keys())
    return f"Would improve element {resolved_names} from {old_st.get(list(resolve_to.keys())[0], 'UNCLEAR')} to {list(resolve_to.values())[0]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg_preview(messages: list[dict[str, Any]], idx: int, max_chars: int = 60) -> str:
    for m in messages:
        if m.get("index") == idx or m.get("msg_index") == idx:
            text = (m.get("text") or "").strip()[:max_chars]
            sender = m.get("sender") or "?"
            if len(text) == max_chars:
                text += "…"
            return f"#{idx} ({sender}): \"{text}\""
    return f"#{idx}"


def _closed_ids(case_file: dict[str, Any]) -> set[str]:
    """Return request IDs that are already closed in the case file."""
    closed: set[str] = set()
    for r in case_file.get("requests") or []:
        if r.get("status") in _CLOSED or r.get("status") in ("answered", "dismissed"):
            closed.add(str(r.get("id", "")))
    return closed


def _existing_types(case_file: dict[str, Any]) -> set[str]:
    return {(ev.get("type") or "") for ev in (case_file.get("evidence") or [])}


# ---------------------------------------------------------------------------
# Individual gap builders
# ---------------------------------------------------------------------------

def _c_offer_missing(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    st = snapshot.get("derived_statuses") or {}
    if st.get("C") == "PRESENT":
        return None
    is_cert_blocker = any("element C" in r for r in cert_reasons)
    return FollowUpRequestV2(
        id="C_OFFER_MISSING:offer",
        gap_code="C_OFFER_MISSING",
        elements=["C", "G"],
        priority=1 if is_cert_blocker else 2,
        blocks_certificate=is_cert_blocker,
        kind="UPLOAD",
        title="Upload the message where the price was proposed",
        why=(
            "No clear offer was found — the message where the price or terms were "
            "first proposed is missing or unclear."
        ),
        how=(
            "If you have an earlier screenshot, upload it. If the price was discussed "
            "in a different chat or over the phone, describe what was agreed in the text "
            "field below."
        ),
        upload_types=["CHAT_SCREENSHOT"],
        fields=[
            InputField(
                key="offer_description",
                label="What was offered and at what price?",
                type="TEXT",
                help="e.g. 'Catering for 50 guests at KES 27,750 all inclusive'",
            ),
            InputField(
                key="offer_date",
                label="Roughly when was the price agreed?",
                type="DATE",
            ),
        ],
        example="A screenshot showing 'I charge 27,750 for catering 50 guests'",
        expected_effect="Can move Offer (C) from ABSENT to PRESENT and may resolve Certainty (G)",
    )


def _d_accept_missing(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    st = snapshot.get("derived_statuses") or {}
    if st.get("D") not in ("ABSENT", "PARTIAL"):
        return None
    is_cert_blocker = any("element D" in r for r in cert_reasons)
    partial = st.get("D") == "PARTIAL"
    return FollowUpRequestV2(
        id="D_ACCEPT_MISSING:acceptance",
        gap_code="D_ACCEPT_MISSING",
        elements=["D"],
        priority=1 if is_cert_blocker else 2,
        blocks_certificate=is_cert_blocker,
        kind="UPLOAD",
        title="Upload the message where the other party agreed",
        why=(
            "No clear acceptance was found — a message from the other party "
            "confirming the deal ('ok', 'sawa', 'done', or making payment) is missing."
            if not partial else
            "The acceptance message was found but it's unclear who sent it or "
            "whether it came after the offer."
        ),
        how=(
            "If you have a screenshot showing the other party's reply to the price, "
            "upload it. The M-Pesa payment confirmation can also serve as evidence of "
            "acceptance if it was made in response to the offer."
        ),
        upload_types=["CHAT_SCREENSHOT", "MPESA_SMS"],
        example="A reply saying 'ok sawa' or 'nimeagree' or the M-Pesa payment confirmation",
        expected_effect="Can move Acceptance (D) from "
                        + ("ABSENT" if not partial else "PARTIAL") + " to PRESENT",
    )


def _d_party_unclear(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    st = snapshot.get("derived_statuses") or {}
    facts = snapshot.get("contract_facts") or {}
    if st.get("D") != "PARTIAL":
        return None
    # Only fire if ambiguity is due to party labelling
    offer_by = facts.get("offer_by", "UNCLEAR")
    accept_by = facts.get("acceptance_by", "UNCLEAR")
    if "UNCLEAR" not in (offer_by, accept_by):
        return None
    # Build MSG_PICK options from offer and acceptance indices
    offer = facts.get("offer") or {}
    acceptance = facts.get("acceptance") or {}
    offer_idx = offer.get("msg_index") if isinstance(offer, dict) else None
    acc_idx = acceptance.get("msg_index") if isinstance(acceptance, dict) else None
    options: list[dict] = []
    if offer_idx:
        options.append({"msg_index": offer_idx, "preview": _msg_preview(messages, offer_idx)})
    if acc_idx and acc_idx != offer_idx:
        options.append({"msg_index": acc_idx, "preview": _msg_preview(messages, acc_idx)})

    return FollowUpRequestV2(
        id="D_PARTY_UNCLEAR:parties",
        gap_code="D_PARTY_UNCLEAR",
        elements=["D"],
        priority=2,
        blocks_certificate=False,
        kind="CHOICE",
        title="Confirm who sent the offer and acceptance",
        why=(
            "The system can't tell with certainty which side made the offer and which "
            "accepted. Confirming this removes the ambiguity."
        ),
        how="For each message shown, tell us whether you (the claimant) or the other party sent it.",
        fields=[
            InputField(
                key="offer_sender",
                label="Who sent the offer?",
                type="MSG_PICK",
                options=options or None,
                help="Select 'You (claimant)' or 'Other party (respondent)'",
            ),
        ],
        related_msg_indices=[i for i in (offer_idx, acc_idx) if i is not None],
        expected_effect="Can stabilise Acceptance (D) to PRESENT once parties are confirmed",
    )


def _e_payment_proof(
    snapshot: dict, messages: list, cert_reasons: list[str], existing_types: set
) -> FollowUpRequestV2 | None:
    st = snapshot.get("derived_statuses") or {}
    if st.get("E") == "PRESENT":
        # Still fire if all payments are CLAIMED ONLY / PRESENT BUT UNVERIFIED
        result = snapshot.get("result") or {}
        payments = result.get("mpesa_payments") or []
        weak = [p for p in payments if p.get("classification") in (
            "CLAIMED ONLY", "PRESENT BUT UNVERIFIED"
        )]
        if not weak:
            return None  # confirmed payment, no request needed
    is_cert_blocker = any("element E" in r for r in cert_reasons)
    has_mpesa = "mpesa_screenshot" in existing_types
    return FollowUpRequestV2(
        id="E_PAYMENT_PROOF:payment",
        gap_code="E_PAYMENT_PROOF",
        elements=["E"],
        priority=1 if is_cert_blocker else 2,
        blocks_certificate=is_cert_blocker,
        kind="UPLOAD",
        title="Upload M-Pesa payment confirmation" if not has_mpesa else "Upload clearer payment proof",
        why=(
            "No M-Pesa confirmation was found. "
            "A claim that payment was made is not the same as evidence of payment."
            if not has_mpesa else
            "The uploaded M-Pesa screenshot could not be verified against the conversation. "
            "A statement from M-Pesa confirms the transaction independently."
        ),
        how=(
            "Open M-Pesa on the phone and find the SMS confirmation for this payment. "
            "Screenshot it and upload here. Alternatively, go to M-Pesa → My Account → "
            "M-Pesa statement and export the PDF covering the payment date."
        ),
        upload_types=["MPESA_SMS", "MPESA_STATEMENT", "RECEIPT"],
        example="SMS from M-Pesa saying 'Confirmed. KES 27,750 sent to [name] on [date]'",
        expected_effect="Can move Consideration (E) to PRESENT and upgrade payment classification",
    )


def _mpesa_inconsistent(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    """Legacy code: fire when consistent=False but no blocking findings caught by _mpesa_conflict."""
    verification = snapshot.get("verification") or {}
    mpesa_check = verification.get("mpesa_cross_check") or {}
    if mpesa_check.get("consistent", True):
        return None
    # _mpesa_conflict will handle fabrication findings; this only fires for the
    # legacy non-structured case (codes_llm_only is non-empty with no findings)
    findings = mpesa_check.get("findings") or []
    blocking = [f for f in findings if f.get("blocks_certificate")]
    if blocking:
        return None  # _mpesa_conflict handles this
    llm_only = mpesa_check.get("codes_llm_only") or []
    return FollowUpRequestV2(
        id="MPESA_INCONSISTENT:cross_check",
        gap_code="MPESA_INCONSISTENT",
        elements=["INTEGRITY"],
        priority=1,
        blocks_certificate=True,
        kind="UPLOAD",
        title="Upload M-Pesa statement to confirm payment",
        why=(
            f"The analysis found {len(llm_only)} payment code(s) that appear in the "
            "conversation but not in the uploaded M-Pesa screenshots. This creates an "
            "integrity issue."
        ),
        how=(
            "Open the M-Pesa app → My Account → Statements. Export the period covering "
            "the transaction dates and upload the PDF or screenshot here."
        ),
        upload_types=["MPESA_STATEMENT"],
        expected_effect="Resolves M-Pesa cross-check inconsistency and removes the integrity block",
    )


def _mpesa_conflict(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    """Fire when blocking M-Pesa findings (LLM_INVENTED_CODE/FIELD, DUPLICATE_CONFLICT) are present."""
    verification = snapshot.get("verification") or {}
    mpesa_check = verification.get("mpesa_cross_check") or {}
    findings = mpesa_check.get("findings") or []
    blocking = [f for f in findings if f.get("blocks_certificate")]
    if not blocking:
        return None
    # Describe the first blocking finding
    first = blocking[0]
    kind = first.get("kind", "")
    code = first.get("code")
    amount = first.get("amount_kes")
    if kind in ("LLM_INVENTED_CODE", "LLM_INVENTED_FIELD"):
        reason = (
            f"The analysis cited M-Pesa code {code} "
            + (f"(KES {amount:,}) " if amount else "")
            + "but it does not appear in the uploaded screenshots. "
            "This suggests a discrepancy between the conversation and the evidence."
        )
    elif kind == "DUPLICATE_CONFLICT":
        reason = (
            f"Code {code} appears in multiple screenshots with conflicting amounts. "
            "This may indicate different transactions or an error."
        )
    else:
        reason = "M-Pesa data in the conversation does not match the uploaded screenshots."

    return FollowUpRequestV2(
        id=f"MPESA_CONFLICT:{code or 'unknown'}",
        gap_code="MPESA_CONFLICT",
        elements=["INTEGRITY"],
        priority=1,
        blocks_certificate=True,
        kind="UPLOAD",
        title="Upload the original M-Pesa confirmation screenshot",
        why=reason,
        how=(
            "Upload the original M-Pesa SMS or screenshot from your phone's gallery. "
            "If you have an M-Pesa statement PDF, upload that instead — it lists all transactions."
        ),
        upload_types=["MPESA_SMS", "MPESA_STATEMENT"],
        expected_effect="Resolves M-Pesa fabrication flag; required before a certificate can be issued",
    )


def _g_scope_missing(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    st = snapshot.get("derived_statuses") or {}
    if st.get("G") not in ("PARTIAL", "ABSENT"):
        return None
    # Don't double-fire if G_SCOPE_DISPUTE already fires
    facts = snapshot.get("contract_facts") or {}
    if facts.get("scope_dispute_refs"):
        return None
    is_cert_blocker = any("terms of the agreement" in r or "element G" in r for r in cert_reasons)
    return FollowUpRequestV2(
        id="G_SCOPE_MISSING:scope",
        gap_code="G_SCOPE_MISSING",
        elements=["G"],
        priority=1 if is_cert_blocker else 2,
        blocks_certificate=is_cert_blocker,
        kind="UPLOAD",
        title="Upload the quote or item list that was agreed",
        why=(
            "The analysis could not fully confirm what was included in the agreed price. "
            "A quote, invoice or item list makes the terms certain."
        ),
        how=(
            "If you have a written quote, invoice or list of items that were agreed, "
            "upload a photo or screenshot here. If the price was agreed verbally, "
            "describe what was included in the text field below."
        ),
        upload_types=["INVOICE_OR_QUOTE", "CHAT_SCREENSHOT", "RECEIPT"],
        fields=[
            InputField(
                key="scope_description",
                label="What was included in the agreed price?",
                type="TEXT",
                help="e.g. 'Catering for 50 guests: food, serving staff, cutlery'",
            ),
        ],
        example="A WhatsApp message listing items: 'Price includes: food, 2 chefs, cutlery'",
        expected_effect="Can move Certainty of terms (G) from PARTIAL to PRESENT",
    )


def _g_scope_dispute(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    facts = snapshot.get("contract_facts") or {}
    dispute_refs = facts.get("scope_dispute_refs") or []
    if not dispute_refs:
        return None
    dispute_indices = [
        (r.get("msg_index") if isinstance(r, dict) else getattr(r, "msg_index", None))
        for r in dispute_refs
    ]
    dispute_indices = [i for i in dispute_indices if i is not None]
    previews = [_msg_preview(messages, i) for i in dispute_indices[:3]]
    is_cert_blocker = any("element G" in r or "terms of the agreement" in r for r in cert_reasons)
    price_ref = facts.get("price_ref")
    price_amount = facts.get("price_amount")
    amount_str = f"KES {price_amount:,}" if price_amount else "the agreed amount"

    return FollowUpRequestV2(
        id=f"G_SCOPE_DISPUTE:{dispute_indices[0] if dispute_indices else 'dispute'}",
        gap_code="G_SCOPE_DISPUTE",
        elements=["G"],
        priority=1 if is_cert_blocker else 2,
        blocks_certificate=is_cert_blocker,
        kind="UPLOAD",
        title="Provide the original quote to resolve scope dispute",
        why=(
            "One or more messages show a disagreement about what was included in "
            f"{amount_str}. Without the original terms the agreed scope is uncertain."
        ),
        how=(
            "Upload the original quote, item list or message where the full price and "
            "what it covered were first agreed. "
            + (f"The disputed messages are: {'; '.join(previews)}." if previews else "")
        ),
        upload_types=["INVOICE_OR_QUOTE", "CHAT_SCREENSHOT"],
        fields=[
            InputField(
                key="scope_original",
                label=f"What was included in {amount_str}?",
                type="TEXT",
                help="List every item or service that was part of the original agreement",
            ),
            InputField(
                key="extra_charge_agreed",
                label="Was the extra charge mentioned before you paid?",
                type="YES_NO",
                help="'Yes' if the extra was part of the original deal; 'No' if added later",
            ),
        ],
        related_msg_indices=dispute_indices,
        expected_effect="Can move Certainty of terms (G) from PARTIAL to PRESENT by clarifying scope",
    )


def _a_claimant_name(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    """Fire when the claimant's identity is not established from messages."""
    st = snapshot.get("derived_statuses") or {}
    if st.get("A") == "PRESENT":
        return None
    facts = snapshot.get("contract_facts") or {}
    claimant_party = facts.get("claimant", "UNCLEAR")
    party_refs = facts.get("party_identity_refs") or {}
    # If claimant IS identifiable from messages, skip this card
    if claimant_party != "UNCLEAR" and party_refs.get(claimant_party):
        return None
    return FollowUpRequestV2(
        id="A_CLAIMANT_NAME:claimant",
        gap_code="A_CLAIMANT_NAME",
        elements=["A"],
        priority=2,
        blocks_certificate=False,
        kind="INPUT",
        title="Provide your full name",
        why=(
            "Your name (the claimant) could not be found in the conversation. "
            "Full names are required for the certificate."
        ),
        how="Type your full legal name below — it will be recorded as your self-declaration.",
        fields=[
            InputField(
                key="claimant_full_name",
                label="Your full name (claimant)",
                type="TEXT",
                required=True,
            ),
        ],
        expected_effect=_simulate_effect(snapshot, {"A": "PARTIAL"}),
    )


def _a_respondent_identity(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    """Fire when the respondent's identity is not established from messages."""
    st = snapshot.get("derived_statuses") or {}
    if st.get("A") == "PRESENT":
        return None
    facts = snapshot.get("contract_facts") or {}
    respondent_party = facts.get("respondent", "UNCLEAR")
    party_refs = facts.get("party_identity_refs") or {}
    # If respondent IS identifiable from messages (e.g. M-Pesa recipient in SYSTEM_SMS), skip
    if respondent_party != "UNCLEAR" and party_refs.get(respondent_party):
        return None
    return FollowUpRequestV2(
        id="A_RESPONDENT_IDENTITY:respondent",
        gap_code="A_RESPONDENT_IDENTITY",
        elements=["A"],
        priority=2,
        blocks_certificate=False,
        kind="UPLOAD",
        title="Upload the other party's contact info",
        why=(
            "The other party's (respondent's) name could not be confirmed from the "
            "conversation. Their identity is required for the certificate."
        ),
        how=(
            "Upload a screenshot of the contact page showing the other party's saved name "
            "and phone number, or type their details below."
        ),
        upload_types=["CONTACT_INFO_SCREENSHOT"],
        fields=[
            InputField(
                key="respondent_full_name",
                label="Other party's full name (respondent)",
                type="TEXT",
                required=True,
            ),
            InputField(
                key="respondent_phone",
                label="Other party's phone number",
                type="PHONE",
            ),
            InputField(
                key="respondent_business",
                label="Business name (if a business)",
                type="TEXT",
                help="Leave blank for individuals",
            ),
        ],
        example="Screenshot of WhatsApp contact page showing name and +254 number",
        expected_effect=_simulate_effect(snapshot, {"A": "PRESENT"}),
    )


def _a_party_identity(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    """Backward-compat: fires when BOTH claimant and respondent need to be named."""
    st = snapshot.get("derived_statuses") or {}
    if st.get("A") == "PRESENT":
        return None
    facts = snapshot.get("contract_facts") or {}
    party_refs = facts.get("party_identity_refs") or {}
    # Only fire the combined card when neither party is identifiable AND
    # the split cards (claimant/respondent) haven't already fired
    claimant_party = facts.get("claimant", "UNCLEAR")
    respondent_party = facts.get("respondent", "UNCLEAR")
    claimant_in_msgs = claimant_party != "UNCLEAR" and bool(party_refs.get(claimant_party))
    respondent_in_msgs = respondent_party != "UNCLEAR" and bool(party_refs.get(respondent_party))
    # If the split cards will handle the situation, don't fire the combined one
    if not claimant_in_msgs or not respondent_in_msgs:
        return None
    # Both are in evidence but A is not PRESENT (unusual edge case)
    return FollowUpRequestV2(
        id="A_PARTY_IDENTITY:parties",
        gap_code="A_PARTY_IDENTITY",
        elements=["A"],
        priority=2,
        blocks_certificate=False,
        kind="UPLOAD",
        title="Confirm party identities",
        why="Party identities found but need confirmation for the certificate.",
        how="Upload contact info screenshots or fill in the names below.",
        upload_types=["CONTACT_INFO_SCREENSHOT"],
        fields=[
            InputField(key="claimant_full_name", label="Your full name", type="TEXT", required=True),
            InputField(key="respondent_full_name", label="Other party's full name", type="TEXT", required=True),
            InputField(key="respondent_phone", label="Other party's phone number", type="PHONE"),
        ],
        expected_effect=_simulate_effect(snapshot, {"A": "PRESENT"}),
    )


def _k_breach_evidence(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    st = snapshot.get("derived_statuses") or {}
    if st.get("K") not in ("UNCLEAR",):
        return None
    # Only fire if the case seems to involve a claim (can't tell without context, skip if no hint)
    facts = snapshot.get("contract_facts") or {}
    if not facts.get("offer") and not facts.get("price_ref"):
        return None  # too little evidence to know what K should be about
    return FollowUpRequestV2(
        id="K_BREACH_EVIDENCE:breach",
        gap_code="K_BREACH_EVIDENCE",
        elements=["K"],
        priority=2,
        blocks_certificate=False,
        kind="UPLOAD",
        title="Upload evidence of non-delivery or breach",
        why=(
            "Whether the other party performed or breached the agreement is unclear. "
            "Evidence of what happened (or didn't) strengthens the case narrative."
        ),
        how=(
            "Upload any messages where you complained about non-delivery, the other party "
            "refused to perform, or where delivery was confirmed. "
            "Also fill in the details below."
        ),
        upload_types=["CHAT_SCREENSHOT"],
        fields=[
            InputField(
                key="breach_description",
                label="What happened? (What was not delivered or done?)",
                type="TEXT",
                required=True,
            ),
            InputField(
                key="due_date",
                label="When was delivery/performance due?",
                type="DATE",
            ),
        ],
        expected_effect="Can move Performance & breach (K) from UNCLEAR to PRESENT or PARTIAL",
    )


def _k_due_date(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    facts = snapshot.get("contract_facts") or {}
    if facts.get("due_date_ref"):
        return None  # already found
    st = snapshot.get("derived_statuses") or {}
    # Only useful when K is not yet established and we have a contract
    if st.get("K") == "PRESENT":
        return None
    if st.get("C") != "PRESENT" and st.get("E") != "PRESENT":
        return None
    return FollowUpRequestV2(
        id="K_DUE_DATE:due_date",
        gap_code="K_DUE_DATE",
        elements=["K"],
        priority=3,
        blocks_certificate=False,
        kind="INPUT",
        title="Provide the agreed delivery or payment date",
        why=(
            "No specific due date was found in the conversation. Knowing when "
            "performance was due strengthens the timeline."
        ),
        how=(
            "If a date was agreed, enter it below. If you have a message mentioning the "
            "date, upload a screenshot."
        ),
        upload_types=["CHAT_SCREENSHOT"],
        fields=[
            InputField(
                key="due_date",
                label="When was delivery / performance due?",
                type="DATE",
                required=True,
            ),
        ],
        expected_effect="Adds a performance deadline to the timeline; may strengthen K",
    )


def _ts_no_dates(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    ts_info = snapshot.get("timestamp_info") or {}
    coverage = ts_info.get("coverage", 1.0)
    if coverage >= 0.5:
        return None
    facts = snapshot.get("contract_facts") or {}
    # Collect key event message indices
    key_indices: list[int] = []
    for field in ("offer", "acceptance"):
        ref = facts.get(field)
        if isinstance(ref, dict) and ref.get("msg_index"):
            key_indices.append(ref["msg_index"])
    previews = [_msg_preview(messages, i) for i in key_indices[:3]]
    return FollowUpRequestV2(
        id="TS_NO_DATES:timestamps",
        gap_code="TS_NO_DATES",
        elements=[],
        priority=2,
        blocks_certificate=False,
        kind="UPLOAD",
        title="Recapture key messages with the date header visible",
        why=(
            f"Only {int(coverage * 100)}% of key messages have visible dates. "
            "Visible dates strengthen the evidence record and raise the confidence score."
        ),
        how=(
            "Scroll up in the chat until you see the date chip (e.g. 'Sunday, 30 Oct 2024'). "
            "Take a new screenshot that includes that chip and the key messages below it. "
            "The messages you need to recapture are: "
            + ("; ".join(previews) or "the offer and acceptance messages") + "."
        ),
        upload_types=["CHAT_SCREENSHOT"],
        example="Screenshot showing 'Sunday, 30 Oct 2024' above the message where the price was agreed",
        expected_effect="Raises timestamp coverage and the confidence score",
    )


def _ts_conflict(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    ts_info = snapshot.get("timestamp_info") or {}
    if not ts_info.get("chronology_conflict"):
        return None
    return FollowUpRequestV2(
        id="TS_CONFLICT:chronology",
        gap_code="TS_CONFLICT",
        elements=["INTEGRITY"],
        priority=1,
        blocks_certificate=True,
        kind="UPLOAD",
        title="Resolve conflicting message dates",
        why=(
            "Two or more key messages have dates that contradict the expected order "
            "(a later message appears to be dated before an earlier one). This may "
            "indicate screenshots from different dates were mixed."
        ),
        how=(
            "Upload clearer screenshots of the messages with conflicting dates, ensuring "
            "the full date header is visible. If the dates are from different conversations, "
            "explain below."
        ),
        upload_types=["CHAT_SCREENSHOT"],
        fields=[
            InputField(
                key="date_conflict_explanation",
                label="Why do the dates appear out of order?",
                type="TEXT",
                help="e.g. 'The 2nd screenshot is from a separate chat about the deposit'",
            ),
        ],
        expected_effect="Resolves chronology conflict; may change integrity rating from ADEQUATE to STRONG",
    )


def _int_edited(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    flags = snapshot.get("integrity_flags") or []
    if "edited_messages" not in flags:
        return None
    edited = [m for m in messages if m.get("edited")]
    edited_indices = [m.get("index") or m.get("msg_index") for m in edited[:3]]
    previews = [_msg_preview(messages, i) for i in edited_indices if i]
    return FollowUpRequestV2(
        id="INT_EDITED:edited_messages",
        gap_code="INT_EDITED",
        elements=["INTEGRITY"],
        priority=2,
        blocks_certificate=False,
        kind="UPLOAD",
        title="Provide earlier screenshot of the edited message",
        why=(
            f"{len(edited)} message(s) in the conversation were edited after being sent. "
            "The original wording is unknown and may have changed the agreed terms."
        ),
        how=(
            "If you took a screenshot of the conversation before the message was edited, "
            "upload it here. Otherwise, tell us what the original message said."
            + (f" Edited messages: {'; '.join(previews)}." if previews else "")
        ),
        upload_types=["CHAT_SCREENSHOT"],
        fields=[
            InputField(
                key="edited_original_text",
                label="What did the message say before it was edited?",
                type="TEXT",
            ),
        ],
        related_msg_indices=[i for i in edited_indices if i],
        expected_effect="May change integrity rating from ADEQUATE to STRONG if original text shows no change to terms",
    )


def _int_deleted(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    flags = snapshot.get("integrity_flags") or []
    if "deleted_messages" not in flags:
        return None
    return FollowUpRequestV2(
        id="INT_DELETED:deleted_messages",
        gap_code="INT_DELETED",
        elements=["INTEGRITY"],
        priority=2,
        blocks_certificate=False,
        kind="INPUT",
        title="Tell us what the deleted message said",
        why=(
            "One or more messages were deleted. If the deleted content is relevant "
            "to the agreement, it should be on record."
        ),
        how=(
            "If you can recall what the deleted message said and who deleted it, "
            "provide that below. This is recorded as a claimant statement (unverified)."
        ),
        fields=[
            InputField(
                key="deleted_content",
                label="What did the deleted message say?",
                type="TEXT",
            ),
            InputField(
                key="deleted_by",
                label="Who deleted it?",
                type="TEXT",
                help="'You' or 'The other party' or 'Unknown'",
            ),
        ],
        expected_effect="Provides context; recorded as unverified statement only",
    )


def _int_low_ocr(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    flags = snapshot.get("integrity_flags") or []
    if "low_ocr_confidence" not in flags:
        return None
    return FollowUpRequestV2(
        id="INT_LOW_OCR:ocr_quality",
        gap_code="INT_LOW_OCR",
        elements=["INTEGRITY"],
        priority=1,
        blocks_certificate=True,
        kind="UPLOAD",
        title="Re-upload the original screenshots at full quality",
        why=(
            "One or more screenshots could not be read clearly. This lowers the "
            "integrity rating and may prevent the certificate."
        ),
        how=(
            "Upload the original screenshots directly from your phone's gallery. "
            "Do not crop, forward or compress them. Upload .PNG or .JPG at full size."
        ),
        upload_types=["CHAT_SCREENSHOT", "MPESA_SMS"],
        example="Full-size .PNG screenshot from your phone's camera roll, not a compressed forward",
        expected_effect="Raises OCR confidence and may change integrity rating from WEAK to ADEQUATE",
    )


def _int_gap(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    flags = snapshot.get("integrity_flags") or []
    if not ({"gap_detected", "screenshot_boundary"} & set(flags)):
        return None
    # Try to identify gap location from message indices
    sorted_msgs = sorted(messages, key=lambda m: m.get("index") or m.get("msg_index") or 0)
    gap_previews = []
    for i in range(len(sorted_msgs) - 1):
        a = sorted_msgs[i].get("index") or sorted_msgs[i].get("msg_index") or 0
        b = sorted_msgs[i + 1].get("index") or sorted_msgs[i + 1].get("msg_index") or 0
        if b - a > 10:  # large gap in indices
            gap_previews.append(f"between {_msg_preview(messages, a)} and {_msg_preview(messages, b)}")
    return FollowUpRequestV2(
        id="INT_GAP:screenshot_boundary",
        gap_code="INT_GAP",
        elements=["INTEGRITY"],
        priority=2,
        blocks_certificate=False,
        kind="UPLOAD",
        title="Upload the missing part of the conversation",
        why=(
            "The conversation appears to start or end mid-way. Missing sections may "
            "contain agreed terms or relevant context."
        ),
        how=(
            "Scroll to the section that is missing and take a screenshot. Upload it here. "
            + (f"The gap appears {gap_previews[0]}." if gap_previews else "")
        ),
        upload_types=["CHAT_SCREENSHOT"],
        expected_effect="Closes the conversation gap; may change integrity from ADEQUATE to STRONG",
    )


def _unstable_fact(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> list[FollowUpRequestV2]:
    unstable = snapshot.get("unstable_fields") or []
    disagreements = snapshot.get("model_disagreements") or {}
    facts = snapshot.get("contract_facts") or {}
    out: list[FollowUpRequestV2] = []

    # Map unstable fields to the two model readings
    field_to_desc = {
        "offer": "the offer message",
        "acceptance": "the acceptance message",
        "price_ref": "the price message",
        "subject_matter_ref": "the subject matter",
    }

    for field in unstable:
        if field not in field_to_desc:
            continue
        ref = facts.get(field)
        idx = ref.get("msg_index") if isinstance(ref, dict) else None
        element = {"offer": "C", "acceptance": "D", "price_ref": "E",
                   "subject_matter_ref": "G"}.get(field)
        options = []
        if idx:
            options.append({"msg_index": idx, "preview": _msg_preview(messages, idx), "source": "reading_1"})
        # We don't have alt_dict readily available in the snapshot; skip 2nd reading
        # for now — the user can pick "neither"

        out.append(FollowUpRequestV2(
            id=f"UNSTABLE_FACT:{field}",
            gap_code="UNSTABLE_FACT",
            elements=[element] if element else [],
            priority=2,
            blocks_certificate=False,
            kind="CHOICE",
            title=f"Confirm {field_to_desc.get(field, field)}",
            why=(
                f"Two independent readings disagreed about {field_to_desc.get(field, field)}. "
                "Your confirmation breaks the tie."
            ),
            how=(
                "Select the message that best represents the "
                + field_to_desc.get(field, field)
                + ", or select 'Neither' if neither is correct."
            ),
            fields=[
                InputField(
                    key=f"tiebreak_{field}",
                    label=f"Which message is {field_to_desc.get(field, field)}?",
                    type="MSG_PICK",
                    options=options if options else None,
                    help="Pick one reading, or 'Neither' to flag this for review",
                ),
            ],
            related_msg_indices=[idx] if idx else [],
            expected_effect=f"Stabilises {field_to_desc.get(field, field)} (removes from unstable fields)",
        ))

    return out


def _s106b_hash(
    snapshot: dict, messages: list, cert_reasons: list[str]
) -> FollowUpRequestV2 | None:
    # Only fire if explicitly in cert_eligibility reasons
    if not any("4a" in r or "hash" in r.lower() for r in cert_reasons):
        return None
    return FollowUpRequestV2(
        id="S106B_HASH:file_integrity",
        gap_code="S106B_HASH",
        elements=["106B_4a"],
        priority=1,
        blocks_certificate=True,
        kind="UPLOAD",
        title="Re-upload the original screenshot files",
        why=(
            "One or more screenshot files appear to have been modified after upload. "
            "The certificate requires files that match the originals exactly."
        ),
        how=(
            "Upload the screenshots directly from your phone's gallery without any "
            "editing, cropping or compression. Use the original .jpg or .png files."
        ),
        upload_types=["CHAT_SCREENSHOT", "MPESA_SMS"],
        expected_effect="Restores file integrity (4a); required before a certificate can be issued",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Minimum element weight rank (lower = more impactful = shown first within tier)
def _elem_sort_key(req: FollowUpRequestV2) -> tuple[int, int]:
    elem_ranks = [_ELEM_RANK.get(e, 99) for e in req.elements] or [99]
    return (req.priority, min(elem_ranks))


def build_followups(
    snapshot: dict[str, Any],
    case_file: dict[str, Any],
    messages: list[dict[str, Any]],
) -> list[FollowUpRequestV2]:
    """Build deterministic follow-up requests from an analysis snapshot.

    Returns sorted list (priority asc, element weight desc). At most MAX_OPEN
    requests are returned with status OPEN; the rest carry status OPEN but are
    behind a "show more" gate on the frontend.

    Already-closed requests (answered, dismissed, RESOLVED, UNAVAILABLE) are
    suppressed entirely.
    """
    closed_ids = _closed_ids(case_file)
    existing = _existing_types(case_file)
    cert_info = snapshot.get("cert_eligibility") or {}
    cert_reasons: list[str] = cert_info.get("reasons") or []
    # C2: system reasons are transient failures — do NOT show a card, the UI shows a Retry button
    system_reason_codes: set[str] = set(cert_info.get("system_reason_codes") or [])

    all_reqs: list[FollowUpRequestV2] = []

    # Build all candidates
    candidates = [
        _c_offer_missing(snapshot, messages, cert_reasons),
        _d_accept_missing(snapshot, messages, cert_reasons),
        _d_party_unclear(snapshot, messages, cert_reasons),
        _e_payment_proof(snapshot, messages, cert_reasons, existing),
        _mpesa_conflict(snapshot, messages, cert_reasons),
        _mpesa_inconsistent(snapshot, messages, cert_reasons),
        _g_scope_missing(snapshot, messages, cert_reasons),
        _g_scope_dispute(snapshot, messages, cert_reasons),
        # Part D: split A_PARTY_IDENTITY
        _a_claimant_name(snapshot, messages, cert_reasons),
        _a_respondent_identity(snapshot, messages, cert_reasons),
        _a_party_identity(snapshot, messages, cert_reasons),
        _k_breach_evidence(snapshot, messages, cert_reasons),
        _k_due_date(snapshot, messages, cert_reasons),
        _ts_no_dates(snapshot, messages, cert_reasons),
        _ts_conflict(snapshot, messages, cert_reasons),
        _int_edited(snapshot, messages, cert_reasons),
        _int_deleted(snapshot, messages, cert_reasons),
        _int_low_ocr(snapshot, messages, cert_reasons),
        _int_gap(snapshot, messages, cert_reasons),
        _s106b_hash(snapshot, messages, cert_reasons),
    ]

    for req in candidates:
        if req is None:
            continue
        if req.id in closed_ids:
            continue
        all_reqs.append(req)

    # Unstable facts (may produce multiple)
    for req in _unstable_fact(snapshot, messages, cert_reasons):
        if req.id not in closed_ids:
            all_reqs.append(req)

    # Deduplicate by id
    seen_ids: set[str] = set()
    deduped: list[FollowUpRequestV2] = []
    for req in all_reqs:
        if req.id not in seen_ids:
            seen_ids.add(req.id)
            deduped.append(req)

    # Sort: priority (1 first), then element weight (C first)
    deduped.sort(key=_elem_sort_key)

    return deduped


# ---------------------------------------------------------------------------
# Round diff computation
# ---------------------------------------------------------------------------

def compute_round_diff(
    prev_snapshot: dict[str, Any],
    new_snapshot: dict[str, Any],
    prev_requests: list[FollowUpRequestV2],
    new_requests: list[FollowUpRequestV2],
    round_num: int,
) -> dict[str, Any]:
    """Compare two analysis snapshots and return a round diff dict."""
    prev_st = prev_snapshot.get("derived_statuses") or {}
    new_st = new_snapshot.get("derived_statuses") or {}

    elements_changed = []
    for elem, new_val in new_st.items():
        old_val = prev_st.get(elem)
        if old_val != new_val:
            elements_changed.append({
                "element": elem,
                "from": old_val or "UNCLEAR",
                "to": new_val,
                "because": "",  # filled by pipeline if possible
            })

    prev_ids = {r.id for r in prev_requests}
    new_ids = {r.id for r in new_requests}

    resolved = [r.id for r in prev_requests if r.id not in new_ids]
    still_open = [{"id": r.id, "why": r.why} for r in new_requests if r.id in prev_ids]
    new_reqs = [r.id for r in new_requests if r.id not in prev_ids]

    prev_conf = prev_snapshot.get("confidence") or 0.0
    new_conf = new_snapshot.get("confidence") or 0.0
    prev_elig = (prev_snapshot.get("cert_eligibility") or {}).get("eligible", False)
    new_elig = (new_snapshot.get("cert_eligibility") or {}).get("eligible", False)

    # Evidence tracking (Part E)
    prev_manifest = {
        e["evidence_id"]: e
        for e in (prev_snapshot.get("analysis_input") or {}).get("evidence_manifest") or []
        if e.get("evidence_id")
    }
    new_manifest = {
        e["evidence_id"]: e
        for e in (new_snapshot.get("analysis_input") or {}).get("evidence_manifest") or []
        if e.get("evidence_id")
    }
    evidence_added = [
        new_manifest[eid] for eid in new_manifest if eid not in prev_manifest
    ]
    evidence_removed = [
        prev_manifest[eid] for eid in prev_manifest if eid not in new_manifest
    ]

    return {
        "round": round_num,
        "previous_analysis_id": prev_snapshot.get("analysis_id"),
        "confidence": {"from": prev_conf, "to": new_conf},
        "verdict": {
            "from": prev_snapshot.get("verdict", ""),
            "to": new_snapshot.get("verdict", ""),
        },
        "elements_changed": elements_changed,
        "requests_resolved": resolved,
        "requests_still_open": still_open,
        "new_requests": new_reqs,
        "cert_eligibility": {"from": prev_elig, "to": new_elig},
        "evidence_added": evidence_added,
        "evidence_removed": evidence_removed,
    }
