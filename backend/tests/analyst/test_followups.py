"""Tests for analyst/followup_catalog.py.

Checks that build_followups() fires the right gap codes given specific
derived status / integrity flag combinations, and suppresses requests that
are already closed.
"""

from __future__ import annotations

import pytest

from SCCVision.analyst.followup_catalog import MAX_OPEN, build_followups
from SCCVision.analyst.followups import FollowUpRequestV2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot(
    *,
    statuses: dict | None = None,
    unstable: list | None = None,
    flags: list | None = None,
    ts_info: dict | None = None,
    facts: dict | None = None,
    cert: dict | None = None,
    mpesa_cross: dict | None = None,
) -> dict:
    return {
        "derived_statuses": statuses or {},
        "unstable_fields": unstable or [],
        "integrity_flags": flags or [],
        "timestamp_info": ts_info or {"coverage": 1.0, "chronology_conflict": False},
        "contract_facts": facts or {},
        "cert_eligibility": cert or {"eligible": True, "reasons": []},
        "verification": {"mpesa_cross_check": mpesa_cross or {"consistent": True}},
    }


def _gap_codes(reqs: list[FollowUpRequestV2]) -> list[str]:
    return [r.gap_code for r in reqs]


def _case_file(requests: list[dict] | None = None, evidence: list | None = None) -> dict:
    return {"evidence": evidence or [{"evidence_id": "ev1"}], "requests": requests or []}


MSGS: list[dict] = []


# ---------------------------------------------------------------------------
# Perfect snapshot — no requests expected
# ---------------------------------------------------------------------------

def test_no_requests_when_all_present():
    snap = _snapshot(statuses={k: "PRESENT" for k in "ABCDEFGHIJK"})
    result = build_followups(snap, _case_file(), MSGS)
    assert result == []


# ---------------------------------------------------------------------------
# Individual gap triggers
# ---------------------------------------------------------------------------

def test_c_offer_missing_fires_when_absent():
    snap = _snapshot(statuses={"C": "ABSENT"})
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "C_OFFER_MISSING" in _gap_codes(reqs)


def test_c_offer_missing_not_fired_when_present():
    snap = _snapshot(statuses={"C": "PRESENT"})
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "C_OFFER_MISSING" not in _gap_codes(reqs)


def test_d_accept_absent():
    snap = _snapshot(statuses={"D": "ABSENT"})
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "D_ACCEPT_MISSING" in _gap_codes(reqs)


def test_d_party_unclear_fires_on_partial_with_unclear_parties():
    snap = _snapshot(
        statuses={"D": "PARTIAL"},
        facts={"offer_by": "UNCLEAR", "acceptance_by": "UNCLEAR"},
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    codes = _gap_codes(reqs)
    # D_PARTY_UNCLEAR fires; D_ACCEPT_MISSING also fires for PARTIAL
    assert "D_PARTY_UNCLEAR" in codes


def test_d_party_unclear_not_fired_when_parties_known():
    snap = _snapshot(
        statuses={"D": "PARTIAL"},
        facts={"offer_by": "claimant", "acceptance_by": "respondent"},
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "D_PARTY_UNCLEAR" not in _gap_codes(reqs)


def test_e_payment_proof_fires_when_absent():
    snap = _snapshot(statuses={"E": "ABSENT"})
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "E_PAYMENT_PROOF" in _gap_codes(reqs)


def test_e_payment_proof_not_fired_when_present_and_verified():
    snap = _snapshot(
        statuses={"E": "PRESENT"},
        facts={"payments": [{"classification": "VERIFIED-LOOKING"}]},
    )
    # With no weak payments in result.mpesa_payments
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "E_PAYMENT_PROOF" not in _gap_codes(reqs)


def test_mpesa_inconsistent_fires():
    snap = _snapshot(
        mpesa_cross={"consistent": False, "codes_llm_only": ["QDF3A"], "codes_parsed_only": []},
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "MPESA_INCONSISTENT" in _gap_codes(reqs)


def test_mpesa_consistent_not_fired():
    snap = _snapshot(mpesa_cross={"consistent": True})
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "MPESA_INCONSISTENT" not in _gap_codes(reqs)


def test_g_scope_missing_on_partial():
    snap = _snapshot(statuses={"G": "PARTIAL"})
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "G_SCOPE_MISSING" in _gap_codes(reqs)


def test_g_scope_dispute_fires_when_dispute_refs_present():
    snap = _snapshot(
        statuses={"G": "PARTIAL"},
        facts={"scope_dispute_refs": [{"msg_index": 5}]},
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    codes = _gap_codes(reqs)
    assert "G_SCOPE_DISPUTE" in codes
    # G_SCOPE_MISSING should NOT fire when dispute refs present
    assert "G_SCOPE_MISSING" not in codes


def test_a_party_identity_on_partial():
    snap = _snapshot(statuses={"A": "PARTIAL"})
    reqs = build_followups(snap, _case_file(), MSGS)
    codes = _gap_codes(reqs)
    # Either the split codes or the combined code must fire
    assert (
        "A_CLAIMANT_NAME" in codes
        or "A_RESPONDENT_IDENTITY" in codes
        or "A_PARTY_IDENTITY" in codes
    )


def test_k_breach_evidence_fires_on_unclear_with_contract():
    snap = _snapshot(
        statuses={"K": "UNCLEAR"},
        facts={"offer": {"msg_index": 1, "span": "deliver goods"}, "price_ref": {"msg_index": 2}},
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "K_BREACH_EVIDENCE" in _gap_codes(reqs)


def test_k_due_date_fires_when_no_due_date_and_contract_exists():
    snap = _snapshot(
        statuses={"C": "PRESENT", "E": "PRESENT"},
        facts={},  # no due_date_ref
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "K_DUE_DATE" in _gap_codes(reqs)


def test_ts_no_dates_fires_on_low_coverage():
    snap = _snapshot(ts_info={"coverage": 0.1, "chronology_conflict": False})
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "TS_NO_DATES" in _gap_codes(reqs)


def test_ts_no_dates_not_fired_on_good_coverage():
    snap = _snapshot(ts_info={"coverage": 0.9, "chronology_conflict": False})
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "TS_NO_DATES" not in _gap_codes(reqs)


def test_ts_conflict_fires():
    snap = _snapshot(ts_info={"coverage": 1.0, "chronology_conflict": True})
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "TS_CONFLICT" in _gap_codes(reqs)


def test_int_edited_fires():
    snap = _snapshot(flags=["edited_messages"])
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "INT_EDITED" in _gap_codes(reqs)


def test_int_deleted_fires():
    snap = _snapshot(flags=["deleted_messages"])
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "INT_DELETED" in _gap_codes(reqs)


def test_int_low_ocr_fires():
    snap = _snapshot(flags=["low_ocr_confidence"])
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "INT_LOW_OCR" in _gap_codes(reqs)


def test_int_gap_fires():
    snap = _snapshot(flags=["gap_detected"])
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "INT_GAP" in _gap_codes(reqs)


def test_int_screenshot_boundary_fires():
    snap = _snapshot(flags=["screenshot_boundary"])
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "INT_GAP" in _gap_codes(reqs)


def test_unstable_fact_fires_for_offer():
    snap = _snapshot(
        unstable=["offer"],
        facts={"offer": {"msg_index": 3, "span": "blah"}},
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "UNSTABLE_FACT" in _gap_codes(reqs)
    # id should be UNSTABLE_FACT:offer
    ids = [r.id for r in reqs]
    assert "UNSTABLE_FACT:offer" in ids


def test_unstable_fact_fires_multiple():
    snap = _snapshot(
        unstable=["offer", "acceptance"],
        facts={
            "offer": {"msg_index": 1, "span": "a"},
            "acceptance": {"msg_index": 2, "span": "b"},
        },
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    unstable_ids = [r.id for r in reqs if r.gap_code == "UNSTABLE_FACT"]
    assert len(unstable_ids) == 2


def test_s106b_hash_fires_on_cert_reason():
    snap = _snapshot(
        cert={"eligible": False, "reasons": ["4a — file hash mismatch detected"]},
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "S106B_HASH" in _gap_codes(reqs)


def test_s106b_hash_not_fired_without_reason():
    snap = _snapshot(cert={"eligible": True, "reasons": []})
    reqs = build_followups(snap, _case_file(), MSGS)
    assert "S106B_HASH" not in _gap_codes(reqs)


# ---------------------------------------------------------------------------
# Priority and ordering
# ---------------------------------------------------------------------------

def test_priority1_before_priority2():
    snap = _snapshot(
        statuses={"C": "ABSENT", "D": "ABSENT"},
        flags=["low_ocr_confidence"],  # priority 1
        cert={"eligible": False, "reasons": ["element C not present"]},
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    priorities = [r.priority for r in reqs]
    for i in range(len(priorities) - 1):
        assert priorities[i] <= priorities[i + 1], "Requests must be sorted by priority"


def test_blocks_certificate_set_on_cert_reason_items():
    snap = _snapshot(
        statuses={"C": "ABSENT"},
        cert={"eligible": False, "reasons": ["element C not present"]},
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    c_req = next((r for r in reqs if r.gap_code == "C_OFFER_MISSING"), None)
    assert c_req is not None
    assert c_req.blocks_certificate is True
    assert c_req.priority == 1


# ---------------------------------------------------------------------------
# Closed request suppression
# ---------------------------------------------------------------------------

def test_closed_request_is_suppressed():
    snap = _snapshot(statuses={"C": "ABSENT"})
    cf = _case_file(requests=[{"id": "C_OFFER_MISSING:offer", "status": "answered"}])
    reqs = build_followups(snap, cf, MSGS)
    assert "C_OFFER_MISSING" not in _gap_codes(reqs)


def test_dismissed_request_is_suppressed():
    snap = _snapshot(statuses={"C": "ABSENT"})
    cf = _case_file(requests=[{"id": "C_OFFER_MISSING:offer", "status": "dismissed"}])
    reqs = build_followups(snap, cf, MSGS)
    assert "C_OFFER_MISSING" not in _gap_codes(reqs)


def test_resolved_request_is_suppressed():
    snap = _snapshot(statuses={"C": "ABSENT"})
    cf = _case_file(requests=[{"id": "C_OFFER_MISSING:offer", "status": "RESOLVED"}])
    reqs = build_followups(snap, cf, MSGS)
    assert "C_OFFER_MISSING" not in _gap_codes(reqs)


def test_open_request_is_not_suppressed():
    snap = _snapshot(statuses={"C": "ABSENT"})
    cf = _case_file(requests=[{"id": "C_OFFER_MISSING:offer", "status": "open"}])
    reqs = build_followups(snap, cf, MSGS)
    assert "C_OFFER_MISSING" in _gap_codes(reqs)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_no_duplicate_ids():
    snap = _snapshot(
        statuses={"C": "ABSENT", "D": "ABSENT", "G": "PARTIAL"},
        flags=["edited_messages", "low_ocr_confidence"],
    )
    reqs = build_followups(snap, _case_file(), MSGS)
    ids = [r.id for r in reqs]
    assert len(ids) == len(set(ids)), "Duplicate request IDs found"


# ---------------------------------------------------------------------------
# to_legacy compatibility
# ---------------------------------------------------------------------------

def test_to_legacy_produces_expected_shape():
    snap = _snapshot(statuses={"C": "ABSENT"})
    reqs = build_followups(snap, _case_file(), MSGS)
    c_req = next(r for r in reqs if r.gap_code == "C_OFFER_MISSING")
    legacy = c_req.to_legacy()
    assert "id" in legacy
    assert "element" in legacy
    assert "type" in legacy
    assert "prompt" in legacy
    assert "why_it_matters" in legacy
    assert "blocking" in legacy
    assert isinstance(legacy["blocking"], bool)


# ---------------------------------------------------------------------------
# Field model validation
# ---------------------------------------------------------------------------

def test_input_fields_have_required_attributes():
    snap = _snapshot(statuses={"A": "ABSENT"})
    reqs = build_followups(snap, _case_file(), MSGS)
    # Accept either the split codes or the combined code
    _A_CODES = {"A_PARTY_IDENTITY", "A_CLAIMANT_NAME", "A_RESPONDENT_IDENTITY"}
    party_reqs = [r for r in reqs if r.gap_code in _A_CODES]
    assert party_reqs, f"Expected at least one A party request, got: {_gap_codes(reqs)}"
    for party_req in party_reqs:
        assert len(party_req.fields) >= 1
        for field in party_req.fields:
            assert field.key
            assert field.label
            assert field.type in ("TEXT", "PHONE", "DATE", "AMOUNT_KES", "MSG_PICK", "YES_NO")


# ---------------------------------------------------------------------------
# Idempotency: same snapshot → same result
# ---------------------------------------------------------------------------

def test_idempotent():
    snap = _snapshot(
        statuses={"C": "ABSENT", "E": "ABSENT"},
        flags=["edited_messages"],
    )
    r1 = build_followups(snap, _case_file(), MSGS)
    r2 = build_followups(snap, _case_file(), MSGS)
    assert [r.id for r in r1] == [r.id for r in r2]


# ---------------------------------------------------------------------------
# MAX_OPEN constant is enforced by the catalog constant (not truncated here)
# ---------------------------------------------------------------------------

def test_max_open_constant():
    assert MAX_OPEN == 6
