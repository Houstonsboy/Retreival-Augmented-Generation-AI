"""Deterministic Section 106B compliance gate.

Takes a stored analysis snapshot and litigant-supplied DeviceFacts and
returns a GateResult with one ConditionResult per s.106B requirement.

No LLM calls.  Every decision is auditable from the basis[] strings.
"""

from __future__ import annotations

from typing import Any

from . import GATE_VERSION
from .models import ConditionResult, DeviceFacts, GateResult, ProductionMethod


# ---------------------------------------------------------------------------
# Individual condition evaluators
# ---------------------------------------------------------------------------

def _check_4a(snapshot: dict, facts: DeviceFacts) -> ConditionResult:
    """4a — Electronic record identified, manner of production described."""
    evidence = snapshot.get("evidence") or []
    basis: list[str] = []
    remedies: list[str] = []
    met = True

    if not evidence:
        met = False
        basis.append("No evidence items found in the analysis snapshot.")
        remedies.append("Re-upload the original screenshot files directly from your phone.")
    else:
        for ev in evidence:
            fid = ev.get("evidence_id", "?")
            fn = ev.get("file_name") or ""
            sha = ev.get("server_sha256") or ""
            hm = ev.get("hash_match")

            if not fn:
                met = False
                basis.append(f"Evidence item {fid}: file_name is missing.")
                remedies.append("Re-upload the original screenshot files directly from your phone.")
            else:
                basis.append(f"Evidence item '{fn}' identified.")

            if not sha:
                met = False
                basis.append(f"Evidence item '{fn}': no server-computed SHA-256 hash (file not uploaded via /sccvision/extract).")
                remedies.append("Re-upload the original screenshot files directly from your phone.")
            else:
                basis.append(f"Evidence item '{fn}': SHA-256 hash verified by server.")

            if hm is False:
                met = False
                basis.append(f"Evidence item '{fn}': client hash does not match server hash — file may have been modified after upload.")
                remedies.append("Re-upload the original screenshot files directly from your phone.")
            elif hm is True:
                basis.append(f"Evidence item '{fn}': hash integrity confirmed (client = server).")

    if facts.production_method == ProductionMethod.OTHER:
        met = False
        basis.append("Production method is 'other' — the method cannot be described in the certificate.")
        remedies.append("Describe how the screenshots were produced.")

    remedy = " ".join(dict.fromkeys(remedies)) if remedies else None
    return ConditionResult(
        code="4a",
        title="Electronic record identified and manner of production described",
        met=met,
        basis=basis,
        remedy=remedy,
    )


def _check_4b(facts: DeviceFacts) -> ConditionResult:
    """4b — Particulars of the device provided."""
    basis = [
        f"Device make: '{facts.device_make}'.",
        f"Device model: '{facts.device_model}'.",
    ]
    if facts.phone_number_last4:
        basis.append(f"Phone number last 4 digits supplied (stored masked as ******{facts.phone_number_last4}).")
    return ConditionResult(
        code="4b",
        title="Particulars of the device",
        met=True,
        basis=basis,
        remedy=None,
    )


def _check_2a(facts: DeviceFacts) -> ConditionResult:
    """2a — Device regularly used by a person with lawful control."""
    basis: list[str] = []
    met = facts.lawful_control and facts.regular_use

    if facts.lawful_control:
        basis.append("Declarant confirmed lawful control of the device during the relevant period.")
    else:
        basis.append("Declarant did NOT confirm lawful control of the device.")

    if facts.regular_use:
        basis.append("Declarant confirmed regular use of the device during the relevant period.")
    else:
        basis.append("Declarant did NOT confirm regular use of the device.")

    remedy = (
        "The certificate must be given by the person who controlled and regularly "
        "used the phone at the time."
        if not met else None
    )
    return ConditionResult(
        code="2a",
        title="Device regularly used by a person with lawful control",
        met=met,
        basis=basis,
        remedy=remedy,
    )


def _check_2b(facts: DeviceFacts) -> ConditionResult:
    """2b — Information regularly fed into the device."""
    met = facts.regular_feed
    basis = [
        "Declarant confirmed that messages/payments of this kind were regularly "
        "received or stored on this phone in the ordinary course of activities."
        if met else
        "Declarant did NOT confirm regular feed of this type of information."
    ]
    remedy = (
        "Confirm that messages/payments of this kind were regularly received "
        "or stored on this phone."
        if not met else None
    )
    return ConditionResult(
        code="2b",
        title="Information regularly fed into the device",
        met=met,
        basis=basis,
        remedy=remedy,
    )


def _check_2c(facts: DeviceFacts) -> ConditionResult:
    """2c — Device operating properly."""
    basis: list[str] = []
    met: bool

    if facts.operating_properly:
        met = True
        basis.append("Declarant confirmed the device was operating properly throughout the relevant period.")
    else:
        details = facts.malfunction_details or "unspecified malfunction"
        affects = facts.malfunction_affects_record
        if affects is False:
            met = True
            basis.append(
                f"Malfunction reported ('{details}') but declarant confirmed it did not "
                "affect the record or its accuracy — condition still met per s.106B(2)(c)."
            )
        else:
            met = False
            basis.append(
                f"Malfunction reported ('{details}') and declarant confirmed it affected "
                "the accuracy of the record."
            )

    remedy = (
        "A malfunction that affected the record prevents certification; obtain the record "
        "from another source (e.g. an M-Pesa statement from Safaricom)."
        if not met else None
    )
    return ConditionResult(
        code="2c",
        title="Device operating properly",
        met=met,
        basis=basis,
        remedy=remedy,
    )


def _check_2d(snapshot: dict, facts: DeviceFacts) -> ConditionResult:
    """2d — Record derived from information fed into the device."""
    basis: list[str] = []
    remedies: list[str] = []
    met = True

    if not facts.messages_on_this_device:
        met = False
        basis.append("Declarant did NOT confirm that messages/payments were sent or received directly on this device.")
        remedies.append("Take the screenshots on the phone that sent or received the messages, not from forwarded copies.")

    acceptable_methods = {ProductionMethod.SCREENSHOT_ON_DEVICE, ProductionMethod.PRINTED_STATEMENT}
    if facts.production_method not in acceptable_methods:
        met = False
        basis.append(
            f"Production method '{facts.production_method.value}' means the record was not "
            "taken directly from the originating device."
        )
        remedies.append("Take the screenshots on the phone that sent or received the messages, not from forwarded copies.")
    else:
        basis.append(f"Production method '{facts.production_method.value}' is consistent with direct derivation.")

    # For M-Pesa items, verify at least one parsed transaction code exists
    evidence = snapshot.get("evidence") or []
    mpesa_items = [e for e in evidence if e.get("type") == "mpesa_screenshot"]
    if mpesa_items:
        result = snapshot.get("result") or {}
        payments = result.get("mpesa_payments") or []
        codes = [p.get("code") for p in payments if p.get("code")]
        if codes:
            basis.append(
                f"M-Pesa evidence present: {len(codes)} parsed transaction code(s) found — "
                "derivation from Safaricom system confirmed."
            )
        else:
            met = False
            basis.append(
                "M-Pesa screenshot present but no transaction codes could be parsed — "
                "cannot confirm derivation."
            )
            remedies.append(
                "Upload a clear screenshot of the M-Pesa confirmation message showing the transaction code."
            )

    if facts.messages_on_this_device and facts.production_method in acceptable_methods:
        basis.insert(0, "Declarant confirmed messages/payments were on this device.")

    remedy = " ".join(dict.fromkeys(remedies)) if remedies else None
    return ConditionResult(
        code="2d",
        title="Record derived from information fed into the device",
        met=met,
        basis=basis,
        remedy=remedy,
    )


def _check_4d(facts: DeviceFacts) -> ConditionResult:
    """4d — Signed by a person in a responsible position."""
    basis: list[str] = []
    remedies: list[str] = []
    met = True

    if not facts.owner_is_declarant:
        met = False
        basis.append("Declarant is not the owner/controller of the device.")
        remedies.append("The person who owns and controls the phone must make and sign the declaration.")
    else:
        basis.append(f"Declarant '{facts.declarant_name}' confirmed they own/control the device.")

    if not facts.declaration_confirmed:
        met = False
        basis.append("Declarant did not confirm the declaration checkbox.")
        remedies.append("The declarant must tick the confirmation checkbox to certify the statements are true.")
    else:
        basis.append("Declarant confirmed the declaration is true to the best of their knowledge.")

    basis.append(f"Position stated: '{facts.declarant_position}'.")

    remedy = " ".join(dict.fromkeys(remedies)) if remedies else None
    return ConditionResult(
        code="4d",
        title="Signed by a person in a responsible position",
        met=met,
        basis=basis,
        remedy=remedy,
    )


def _check_integrity(snapshot: dict) -> ConditionResult:
    """INTEGRITY — deterministic verification checks from the stored analysis."""
    verification = snapshot.get("verification") or {}
    result = snapshot.get("result") or {}
    basis: list[str] = []
    remedies: list[str] = []
    met = True

    quotes_ok = verification.get("quotes_grounded", True)
    if not quotes_ok:
        met = False
        ungrounded = verification.get("ungrounded_quotes") or []
        basis.append(
            f"Evidence integrity: {len(ungrounded)} quote(s) could not be traced back "
            "to source messages (quotes_grounded=False)."
        )
        remedies.append("Upload the complete, unedited conversation.")
    else:
        basis.append("All quotes traced back to source messages (quotes_grounded=True).")

    mpesa_check = verification.get("mpesa_cross_check") or {}
    consistent = mpesa_check.get("consistent", True)
    invented = mpesa_check.get("invented_fields", False)
    if not consistent or invented:
        met = False
        llm_only = mpesa_check.get("codes_llm_only") or []
        basis.append(
            f"M-Pesa cross-check failed: consistent={consistent}, invented_fields={invented}. "
            f"LLM-only codes: {llm_only}."
        )
        remedies.append("Upload the complete, unedited conversation.")
    else:
        basis.append("M-Pesa payment cross-check passed.")

    rating = (result.get("conversation_integrity") or {}).get("rating", "ADEQUATE")
    if rating == "WEAK":
        met = False
        issues = (result.get("conversation_integrity") or {}).get("issues") or []
        basis.append(f"Conversation integrity rated WEAK. Issues: {issues}.")
        remedies.append("Upload the complete, unedited conversation.")
    else:
        basis.append(f"Conversation integrity rated {rating}.")

    remedy = " ".join(dict.fromkeys(remedies)) if remedies else None
    return ConditionResult(
        code="INTEGRITY",
        title="Evidence integrity checks passed",
        met=met,
        basis=basis,
        remedy=remedy,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(analysis_snapshot: dict, facts: DeviceFacts) -> GateResult:
    """Run all s.106B conditions and return a GateResult.

    ``analysis_snapshot`` is the dict stored by analysis_store.save_analysis_snapshot().
    ``facts`` is a validated DeviceFacts instance.
    """
    conditions = [
        _check_4a(analysis_snapshot, facts),
        _check_4b(facts),
        _check_2a(facts),
        _check_2b(facts),
        _check_2c(facts),
        _check_2d(analysis_snapshot, facts),
        _check_4d(facts),
        _check_integrity(analysis_snapshot),
    ]

    unmet = [c.code for c in conditions if not c.met]

    # Outcome mapping:
    # all met → CERTIFICATE, certificate_ready
    # only questionnaire-type unmet (2a/2b/2c/2d/4b/4d) while INTEGRITY+4a met → needs_strengthening
    # INTEGRITY or 4a unmet → weak
    if not unmet:
        doc_type = "CERTIFICATE"
        outcome = "certificate_ready"
    elif "INTEGRITY" in unmet or "4a" in unmet:
        doc_type = "DEFICIENCY_REPORT"
        outcome = "weak"
    else:
        doc_type = "DEFICIENCY_REPORT"
        outcome = "needs_strengthening"

    return GateResult(
        document_type=doc_type,
        outcome=outcome,
        conditions=conditions,
        unmet=unmet,
        gate_version=GATE_VERSION,
    )
