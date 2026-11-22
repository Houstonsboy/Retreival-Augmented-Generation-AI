"""Unit tests for deterministic functions introduced in SCCDeterminism.md.

Covers:
- derive_statuses (A–K, one test per branch)
- reconcile (stable / unstable)
- _check_triggers (T1–T4)
- compute_integrity_rating
- compute_confidence
- deterministic_verdict
- cert_eligibility
- timestamp_bonus
- check_refs_grounded
- compute_input_hash canonicalization
- _check_4b blank/whitespace fix
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from types import SimpleNamespace

from SCCVision.analyst.rules import (
    ELEMENT_WEIGHTS,
    INTEGRITY_FACTOR,
    STATUS_SCORES,
    cert_eligibility,
    check_refs_grounded,
    compute_confidence,
    compute_integrity_rating,
    compute_model_disagreements,
    deterministic_verdict,
    derive_statuses,
    reconcile,
    timestamp_bonus,
)
from SCCVision.analyst.pipeline import _check_triggers, compute_input_hash, _canonical_json
from SCCVision.analyst.schema import ContractFacts, Ref, PaymentFact
from SCCVision.s106b.gate import _check_4b
from SCCVision.s106b.models import DeviceFacts, ProductionMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref(idx: int) -> dict:
    return {"msg_index": idx, "span": None}


def _msgs(*indices) -> list[dict]:
    return [{"index": i, "text": f"msg {i}", "timestamp_quality": "NONE"} for i in indices]


def _full_facts(**kwargs) -> dict:
    """Minimal ContractFacts dict with all fields defaulted."""
    base = {
        "claimant": "PARTY_1",
        "respondent": "PARTY_2",
        "party_identity_refs": {},
        "offer": None, "offer_by": "UNCLEAR",
        "acceptance": None, "acceptance_by": "UNCLEAR",
        "price_ref": None, "price_amount": None,
        "subject_matter_ref": None,
        "quantity_or_scope_ref": None,
        "due_date_ref": None,
        "performance_refs": [],
        "breach_refs": [],
        "payments": [],
        "social_or_domestic_context": False,
        "capacity_concern_refs": [],
        "consent_concern_refs": [],
        "illegality_concern_refs": [],
        "formality_required": False,
        "integrity_concerns": [],
        "element_opinions": {},
        "narrative": {},
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# derive_statuses — A
# ---------------------------------------------------------------------------

class TestDeriveStatusesA:
    def test_both_parties_identified(self):
        f = _full_facts(party_identity_refs={"PARTY_1": [_ref(1)], "PARTY_2": [_ref(2)]})
        st = derive_statuses(f, None, None)
        assert st["A"] == "PRESENT"

    def test_one_party_identified(self):
        f = _full_facts(party_identity_refs={"PARTY_1": [_ref(1)]})
        st = derive_statuses(f, None, None)
        assert st["A"] == "PARTIAL"

    def test_no_parties_identified(self):
        f = _full_facts(party_identity_refs={})
        st = derive_statuses(f, None, None)
        assert st["A"] == "UNCLEAR"


# ---------------------------------------------------------------------------
# derive_statuses — B
# ---------------------------------------------------------------------------

class TestDeriveStatusesB:
    def test_no_capacity_concern(self):
        f = _full_facts()
        assert derive_statuses(f, None, None)["B"] == "NO_ISSUE_INDICATED"

    def test_has_capacity_concern(self):
        f = _full_facts(capacity_concern_refs=[_ref(3)])
        assert derive_statuses(f, None, None)["B"] == "UNCLEAR"


# ---------------------------------------------------------------------------
# derive_statuses — C
# ---------------------------------------------------------------------------

class TestDeriveStatusesC:
    def test_offer_present(self):
        f = _full_facts(offer=_ref(1))
        assert derive_statuses(f, None, None)["C"] == "PRESENT"

    def test_offer_absent(self):
        f = _full_facts(offer=None)
        assert derive_statuses(f, None, None)["C"] == "ABSENT"


# ---------------------------------------------------------------------------
# derive_statuses — D
# ---------------------------------------------------------------------------

class TestDeriveStatusesD:
    def test_full_acceptance_present(self):
        f = _full_facts(
            offer=_ref(1), offer_by="PARTY_1",
            acceptance=_ref(2), acceptance_by="PARTY_2",
        )
        assert derive_statuses(f, None, None)["D"] == "PRESENT"

    def test_acceptance_same_party_as_offer(self):
        f = _full_facts(
            offer=_ref(1), offer_by="PARTY_1",
            acceptance=_ref(2), acceptance_by="PARTY_1",
        )
        assert derive_statuses(f, None, None)["D"] == "PARTIAL"

    def test_acceptance_before_offer_index(self):
        f = _full_facts(
            offer=_ref(5), offer_by="PARTY_1",
            acceptance=_ref(2), acceptance_by="PARTY_2",
        )
        assert derive_statuses(f, None, None)["D"] == "PARTIAL"

    def test_acceptance_unclear_party(self):
        f = _full_facts(
            offer=_ref(1), offer_by="PARTY_1",
            acceptance=_ref(2), acceptance_by="UNCLEAR",
        )
        assert derive_statuses(f, None, None)["D"] == "PARTIAL"

    def test_no_offer_no_acceptance(self):
        f = _full_facts()
        assert derive_statuses(f, None, None)["D"] == "ABSENT"

    def test_acceptance_without_offer(self):
        f = _full_facts(acceptance=_ref(2), acceptance_by="PARTY_2")
        assert derive_statuses(f, None, None)["D"] == "PARTIAL"


# ---------------------------------------------------------------------------
# derive_statuses — E
# ---------------------------------------------------------------------------

class TestDeriveStatusesE:
    def test_price_ref_present(self):
        f = _full_facts(price_ref=_ref(3))
        assert derive_statuses(f, None, None)["E"] == "PRESENT"

    def test_payment_ref_present(self):
        f = _full_facts(payments=[{"ref": _ref(4), "classification": "CLAIMED ONLY"}])
        assert derive_statuses(f, None, None)["E"] == "PRESENT"

    def test_no_consideration(self):
        f = _full_facts()
        assert derive_statuses(f, None, None)["E"] == "ABSENT"


# ---------------------------------------------------------------------------
# derive_statuses — F
# ---------------------------------------------------------------------------

class TestDeriveStatusesF:
    def test_social_domestic(self):
        f = _full_facts(social_or_domestic_context=True)
        assert derive_statuses(f, None, None)["F"] == "UNCLEAR"

    def test_commercial_with_consideration(self):
        f = _full_facts(price_ref=_ref(1))
        assert derive_statuses(f, None, None)["F"] == "PRESENT"

    def test_commercial_no_consideration(self):
        f = _full_facts(social_or_domestic_context=False)
        assert derive_statuses(f, None, None)["F"] == "UNCLEAR"


# ---------------------------------------------------------------------------
# derive_statuses — G
# ---------------------------------------------------------------------------

class TestDeriveStatusesG:
    def test_both_refs_present(self):
        f = _full_facts(price_ref=_ref(2), subject_matter_ref=_ref(3))
        assert derive_statuses(f, None, None)["G"] == "PRESENT"

    def test_one_ref_present(self):
        f = _full_facts(price_ref=_ref(2))
        assert derive_statuses(f, None, None)["G"] == "PARTIAL"

    def test_no_refs(self):
        f = _full_facts()
        assert derive_statuses(f, None, None)["G"] == "ABSENT"


# ---------------------------------------------------------------------------
# derive_statuses — H, I, J
# ---------------------------------------------------------------------------

class TestDeriveStatusesHIJ:
    def test_no_consent_concern(self):
        f = _full_facts()
        assert derive_statuses(f, None, None)["H"] == "NO_ISSUE_INDICATED"

    def test_has_consent_concern(self):
        f = _full_facts(consent_concern_refs=[_ref(5)])
        assert derive_statuses(f, None, None)["H"] == "UNCLEAR"

    def test_no_illegality_concern(self):
        f = _full_facts()
        assert derive_statuses(f, None, None)["I"] == "NO_ISSUE_INDICATED"

    def test_has_illegality_concern(self):
        f = _full_facts(illegality_concern_refs=[_ref(6)])
        assert derive_statuses(f, None, None)["I"] == "UNCLEAR"

    def test_formality_not_required(self):
        f = _full_facts(formality_required=False)
        assert derive_statuses(f, None, None)["J"] == "NOT_REQUIRED"

    def test_formality_required(self):
        f = _full_facts(formality_required=True)
        assert derive_statuses(f, None, None)["J"] == "UNCLEAR"


# ---------------------------------------------------------------------------
# derive_statuses — K
# ---------------------------------------------------------------------------

class TestDeriveStatusesK:
    def test_breach_refs_present(self):
        f = _full_facts(breach_refs=[_ref(7)])
        assert derive_statuses(f, None, None)["K"] == "PRESENT"

    def test_performance_refs_partial(self):
        f = _full_facts(performance_refs=[_ref(8)])
        assert derive_statuses(f, None, None)["K"] == "PARTIAL"

    def test_no_refs_unclear(self):
        f = _full_facts()
        assert derive_statuses(f, None, None)["K"] == "UNCLEAR"

    def test_premature_breach_resets_to_unclear(self):
        f = _full_facts(breach_refs=[_ref(7)])
        date_info = {"due_date_resolved": True, "performance_not_yet_due": True}
        assert derive_statuses(f, None, date_info)["K"] == "UNCLEAR"

    def test_premature_breach_no_date_no_reset(self):
        f = _full_facts(breach_refs=[_ref(7)])
        date_info = {"due_date_resolved": False, "performance_not_yet_due": True}
        assert derive_statuses(f, None, date_info)["K"] == "PRESENT"


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------

class TestReconcile:
    def test_no_call2_returns_stable(self):
        c1 = _full_facts(offer=_ref(1))
        final, alt, unstable = reconcile(c1, None)
        assert alt is None
        assert unstable == set()
        assert final["offer"] == _ref(1)

    def test_agree_on_all_fields(self):
        c1 = _full_facts(offer=_ref(1), offer_by="PARTY_1")
        c2 = {"offer": _ref(1), "offer_by": "PARTY_1"}
        final, alt, unstable = reconcile(c1, c2)
        assert unstable == set()
        assert final["offer"] == _ref(1)

    def test_disagree_marks_unstable(self):
        c1 = _full_facts(offer=_ref(1))
        c2 = {"offer": _ref(3)}  # different msg_index
        final, alt, unstable = reconcile(c1, c2)
        assert "offer" in unstable
        # Field neutralised in final
        assert final["offer"] is None
        # Alt reflects c2's value
        assert alt["offer"] == _ref(3)

    def test_disagree_on_payment_list(self):
        payment_a = {"ref": _ref(2), "classification": "CLAIMED ONLY"}
        payment_b = {"ref": _ref(5), "classification": "VERIFIED-LOOKING"}
        c1 = _full_facts(payments=[payment_a])
        c2 = {"payments": [payment_b]}
        final, alt, unstable = reconcile(c1, c2)
        assert "payments" in unstable
        assert final["payments"] == []


# ---------------------------------------------------------------------------
# _check_triggers
# ---------------------------------------------------------------------------

class TestCheckTriggers:
    def _make_facts(self, **kwargs):
        base = {
            "offer": None, "offer_by": "UNCLEAR",
            "acceptance": None, "acceptance_by": "UNCLEAR",
            "price_ref": None, "price_amount": None,
            "subject_matter_ref": None, "quantity_or_scope_ref": None,
            "due_date_ref": None, "performance_refs": [],
            "breach_refs": [], "payments": [], "social_or_domestic_context": False,
            "capacity_concern_refs": [], "consent_concern_refs": [],
            "illegality_concern_refs": [], "formality_required": False,
            "integrity_concerns": [], "element_opinions": {}, "narrative": {},
            "party_identity_refs": {}, "claimant": "UNCLEAR", "respondent": "UNCLEAR",
        }
        base.update(kwargs)
        return ContractFacts(**base)

    def test_T1_provisionally_eligible(self):
        facts = self._make_facts()
        st = {"C": "PRESENT", "D": "PRESENT", "E": "PRESENT", "F": "PRESENT",
              "G": "PRESENT", "A": "PRESENT", "B": "NO_ISSUE_INDICATED",
              "H": "NO_ISSUE_INDICATED", "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED",
              "K": "UNCLEAR"}
        trigger = _check_triggers(
            provisional_eligibility=True,
            derived_st=st,
            model_disagreements={},
            ungrounded_core=[],
            verdict="LIKELY VALID",
            facts=facts,
        )
        assert "T1" in trigger

    def test_T2_model_disagrees_on_core(self):
        facts = self._make_facts()
        st = {"C": "PRESENT", "D": "ABSENT", "E": "PRESENT", "F": "UNCLEAR",
              "G": "PARTIAL", "A": "UNCLEAR", "B": "NO_ISSUE_INDICATED",
              "H": "NO_ISSUE_INDICATED", "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED",
              "K": "UNCLEAR"}
        trigger = _check_triggers(
            provisional_eligibility=False,
            derived_st=st,
            model_disagreements={"D": {"model_status": "PRESENT", "derived_status": "ABSENT"}},
            ungrounded_core=[],
            verdict="POSSIBLY VALID WITH GAPS",
            facts=facts,
        )
        assert "T2" in trigger

    def test_T3_ungrounded_core_ref(self):
        facts = self._make_facts()
        st = {"C": "PRESENT", "D": "ABSENT", "E": "ABSENT", "F": "UNCLEAR",
              "G": "ABSENT", "A": "UNCLEAR", "B": "NO_ISSUE_INDICATED",
              "H": "NO_ISSUE_INDICATED", "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED",
              "K": "UNCLEAR"}
        trigger = _check_triggers(
            provisional_eligibility=False,
            derived_st=st,
            model_disagreements={},
            ungrounded_core=[{"field": "offer", "msg_index": 99, "reason": "index_not_found"}],
            verdict="LIKELY NOT VALID",
            facts=facts,
        )
        assert "T3" in trigger

    def test_no_trigger_for_clear_negative(self):
        facts = self._make_facts()
        st = {"C": "ABSENT", "D": "ABSENT", "E": "ABSENT", "F": "UNCLEAR",
              "G": "ABSENT", "A": "UNCLEAR", "B": "NO_ISSUE_INDICATED",
              "H": "NO_ISSUE_INDICATED", "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED",
              "K": "UNCLEAR"}
        trigger = _check_triggers(
            provisional_eligibility=False,
            derived_st=st,
            model_disagreements={},
            ungrounded_core=[],
            verdict="LIKELY NOT VALID",
            facts=facts,
        )
        assert trigger == []


# ---------------------------------------------------------------------------
# compute_integrity_rating
# ---------------------------------------------------------------------------

class TestComputeIntegrityRating:
    def test_no_flags_no_concerns_strong(self):
        assert compute_integrity_rating([], None) == "STRONG"

    def test_edited_messages_adequate(self):
        assert compute_integrity_rating(["edited_messages"], None) == "ADEQUATE"

    def test_deleted_messages_weak(self):
        assert compute_integrity_rating(["deleted_messages"], None) == "WEAK"

    def test_low_ocr_confidence_weak(self):
        assert compute_integrity_rating(["low_ocr_confidence"], None) == "WEAK"

    def test_fabrication_concern_weak(self):
        concerns = [{"category": "POSSIBLE_FABRICATION", "refs": []}]
        assert compute_integrity_rating([], concerns) == "WEAK"

    def test_edit_indicator_concern_adequate(self):
        concerns = [{"category": "EDIT_INDICATOR", "refs": []}]
        assert compute_integrity_rating([], concerns) == "ADEQUATE"

    def test_screenshot_boundary_adequate(self):
        assert compute_integrity_rating(["screenshot_boundary"], None) == "ADEQUATE"


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def _all_present_st(self):
        return {k: "PRESENT" for k in ELEMENT_WEIGHTS}

    def test_all_present_strong_no_bonus(self):
        st = self._all_present_st()
        st["B"] = "NO_ISSUE_INDICATED"
        st["H"] = "NO_ISSUE_INDICATED"
        st["I"] = "NO_ISSUE_INDICATED"
        st["J"] = "NOT_REQUIRED"
        conf = compute_confidence([st], "STRONG", 0.0)
        assert conf == round(sum(ELEMENT_WEIGHTS.values()), 4)  # 1.0

    def test_adequate_factor_applied(self):
        st = self._all_present_st()
        conf_strong = compute_confidence([st], "STRONG", 0.0)
        conf_adeq = compute_confidence([st], "ADEQUATE", 0.0)
        assert conf_adeq < conf_strong
        assert abs(conf_adeq - conf_strong * INTEGRITY_FACTOR["ADEQUATE"]) < 0.001

    def test_bonus_added(self):
        st = self._all_present_st()
        conf_no_bonus = compute_confidence([st], "STRONG", 0.0)
        conf_with_bonus = compute_confidence([st], "STRONG", 0.03)
        # Clamped to 1.0 since already at max
        assert conf_with_bonus == min(1.0, conf_no_bonus + 0.03)

    def test_two_inspection_statuses_averaged(self):
        st_all = {k: "PRESENT" for k in ELEMENT_WEIGHTS}
        st_absent = {k: "ABSENT" for k in ELEMENT_WEIGHTS}
        conf = compute_confidence([st_all, st_absent], "STRONG", 0.0)
        # Should be average
        from SCCVision.analyst.rules import element_score
        expected = (element_score(st_all) + element_score(st_absent)) / 2
        assert abs(conf - expected) < 0.001

    def test_deterministic_same_input(self):
        st = {"C": "PRESENT", "D": "PRESENT", "E": "PARTIAL", "F": "PRESENT",
              "G": "PARTIAL", "A": "PARTIAL", "K": "UNCLEAR", "B": "NO_ISSUE_INDICATED",
              "H": "NO_ISSUE_INDICATED", "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED"}
        c1 = compute_confidence([st], "ADEQUATE", 0.02)
        c2 = compute_confidence([st], "ADEQUATE", 0.02)
        assert c1 == c2


# ---------------------------------------------------------------------------
# deterministic_verdict
# ---------------------------------------------------------------------------

class TestDeterministicVerdict:
    def _present_core(self):
        return {k: "PRESENT" for k in ("C", "D", "E", "F", "G")}

    def test_likely_valid(self):
        st = {**self._present_core(), "A": "PRESENT",
              "B": "NO_ISSUE_INDICATED", "H": "NO_ISSUE_INDICATED",
              "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED", "K": "UNCLEAR"}
        assert deterministic_verdict(st, "STRONG", []) == "LIKELY VALID"

    def test_likely_not_valid_c_absent(self):
        st = {**self._present_core(), "C": "ABSENT", "A": "UNCLEAR", "B": "NO_ISSUE_INDICATED",
              "H": "NO_ISSUE_INDICATED", "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED", "K": "UNCLEAR"}
        assert deterministic_verdict(st, "STRONG", []) == "LIKELY NOT VALID"

    def test_likely_not_valid_e_absent(self):
        st = {**self._present_core(), "E": "ABSENT", "A": "UNCLEAR", "B": "NO_ISSUE_INDICATED",
              "H": "NO_ISSUE_INDICATED", "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED", "K": "UNCLEAR"}
        assert deterministic_verdict(st, "STRONG", []) == "LIKELY NOT VALID"

    def test_likely_not_valid_d_absent_c_present(self):
        st = {**self._present_core(), "D": "ABSENT", "A": "UNCLEAR", "B": "NO_ISSUE_INDICATED",
              "H": "NO_ISSUE_INDICATED", "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED", "K": "UNCLEAR"}
        assert deterministic_verdict(st, "STRONG", []) == "LIKELY NOT VALID"

    def test_possibly_valid_weak_integrity(self):
        st = {**self._present_core(), "A": "PRESENT", "B": "NO_ISSUE_INDICATED",
              "H": "NO_ISSUE_INDICATED", "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED", "K": "UNCLEAR"}
        assert deterministic_verdict(st, "WEAK", []) == "POSSIBLY VALID WITH GAPS"

    def test_possibly_valid_edited_messages(self):
        st = {**self._present_core(), "A": "PRESENT", "B": "NO_ISSUE_INDICATED",
              "H": "NO_ISSUE_INDICATED", "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED", "K": "UNCLEAR"}
        assert deterministic_verdict(st, "STRONG", ["edited_messages"]) == "POSSIBLY VALID WITH GAPS"

    def test_possibly_valid_partial_element(self):
        st = {**self._present_core(), "G": "PARTIAL", "A": "PARTIAL", "B": "NO_ISSUE_INDICATED",
              "H": "NO_ISSUE_INDICATED", "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED", "K": "UNCLEAR"}
        assert deterministic_verdict(st, "STRONG", []) == "POSSIBLY VALID WITH GAPS"

    def test_insufficient_evidence_all_unclear(self):
        st = {k: "UNCLEAR" for k in "ABCDEFGHIJK"}
        assert deterministic_verdict(st, "STRONG", []) == "INSUFFICIENT EVIDENCE"


# ---------------------------------------------------------------------------
# cert_eligibility
# ---------------------------------------------------------------------------

class TestCertEligibility:
    def _eligible_st(self):
        return {k: "PRESENT" for k in ("C", "D", "E", "F", "G", "A", "K",
                                        "B", "H", "I", "J")}

    def test_all_conditions_met(self):
        result = cert_eligibility(
            verdict="LIKELY VALID",
            st=self._eligible_st(),
            unstable_core=set(),
            integrity="STRONG",
            refs_grounded=True,
            mpesa_consistent=True,
            incomplete=False,
            verified=True,
        )
        assert result["eligible"] is True
        assert result["reasons"] == []

    def test_unverified_not_eligible(self):
        result = cert_eligibility(
            verdict="LIKELY VALID",
            st=self._eligible_st(),
            unstable_core=set(),
            integrity="STRONG",
            refs_grounded=True,
            mpesa_consistent=True,
            incomplete=False,
            verified=False,  # <-- key
        )
        assert result["eligible"] is False
        assert any("verified" in r for r in result["reasons"])

    def test_incomplete_not_eligible(self):
        result = cert_eligibility(
            verdict="LIKELY VALID",
            st=self._eligible_st(),
            unstable_core=set(),
            integrity="STRONG",
            refs_grounded=True,
            mpesa_consistent=True,
            incomplete=True,
            verified=True,
        )
        assert result["eligible"] is False

    def test_bad_verdict_not_eligible(self):
        result = cert_eligibility(
            verdict="LIKELY NOT VALID",
            st=self._eligible_st(),
            unstable_core=set(),
            integrity="STRONG",
            refs_grounded=True,
            mpesa_consistent=True,
            incomplete=False,
            verified=True,
        )
        assert result["eligible"] is False
        assert any("verdict" in r for r in result["reasons"])

    def test_element_c_not_present(self):
        st = {**self._eligible_st(), "C": "PARTIAL"}
        result = cert_eligibility(
            verdict="LIKELY VALID",
            st=st,
            unstable_core=set(),
            integrity="STRONG",
            refs_grounded=True,
            mpesa_consistent=True,
            incomplete=False,
            verified=True,
        )
        assert result["eligible"] is False
        assert any("element C" in r for r in result["reasons"])

    def test_unstable_core_not_eligible(self):
        result = cert_eligibility(
            verdict="LIKELY VALID",
            st=self._eligible_st(),
            unstable_core={"C"},
            integrity="STRONG",
            refs_grounded=True,
            mpesa_consistent=True,
            incomplete=False,
            verified=True,
        )
        assert result["eligible"] is False

    def test_weak_integrity_not_eligible(self):
        result = cert_eligibility(
            verdict="LIKELY VALID",
            st=self._eligible_st(),
            unstable_core=set(),
            integrity="WEAK",
            refs_grounded=True,
            mpesa_consistent=True,
            incomplete=False,
            verified=True,
        )
        assert result["eligible"] is False
        assert any("integrity" in r for r in result["reasons"])

    def test_ungrounded_refs_not_eligible(self):
        result = cert_eligibility(
            verdict="LIKELY VALID",
            st=self._eligible_st(),
            unstable_core=set(),
            integrity="STRONG",
            refs_grounded=False,
            mpesa_consistent=True,
            incomplete=False,
            verified=True,
        )
        assert result["eligible"] is False

    def test_mpesa_inconsistent_not_eligible(self):
        result = cert_eligibility(
            verdict="LIKELY VALID",
            st=self._eligible_st(),
            unstable_core=set(),
            integrity="STRONG",
            refs_grounded=True,
            mpesa_consistent=False,
            incomplete=False,
            verified=True,
        )
        assert result["eligible"] is False


# ---------------------------------------------------------------------------
# timestamp_bonus
# ---------------------------------------------------------------------------

class TestTimestampBonus:
    def test_no_key_refs_zero_bonus(self):
        bonus, info = timestamp_bonus([], [], None)
        assert bonus == 0.0
        assert info["coverage"] == 0.0
        assert info["chronology_conflict"] is False

    def test_all_full_timestamps_max_bonus(self):
        msgs = [
            {"index": 1, "text": "offer", "timestamp_quality": "FULL"},
            {"index": 2, "text": "accept", "timestamp_quality": "FULL"},
        ]
        bonus, info = timestamp_bonus([1, 2], msgs, None)
        assert abs(bonus - 0.06) < 0.001
        assert info["coverage"] == 1.0

    def test_no_timestamps_zero_bonus(self):
        msgs = [
            {"index": 1, "text": "offer", "timestamp_quality": "NONE"},
            {"index": 2, "text": "accept", "timestamp_quality": "NONE"},
        ]
        bonus, info = timestamp_bonus([1, 2], msgs, None)
        assert bonus == 0.0

    def test_chronology_conflict_zero_bonus_with_flag(self):
        msgs = [
            {"index": 1, "text": "offer", "timestamp_quality": "FULL", "resolved_date": "2024-02-10"},
            {"index": 2, "text": "accept", "timestamp_quality": "FULL", "resolved_date": "2024-01-05"},
        ]
        bonus, info = timestamp_bonus([1, 2], msgs, None)
        assert bonus == 0.0
        assert info["chronology_conflict"] is True

    def test_partial_timestamps_partial_bonus(self):
        msgs = [
            {"index": 1, "text": "offer", "timestamp_quality": "FULL"},
            {"index": 2, "text": "accept", "timestamp_quality": "NONE"},
        ]
        bonus, info = timestamp_bonus([1, 2], msgs, None)
        assert 0.0 < bonus < 0.06
        assert info["coverage"] == 0.5


# ---------------------------------------------------------------------------
# check_refs_grounded
# ---------------------------------------------------------------------------

class TestCheckRefsGrounded:
    def test_valid_refs_grounded(self):
        facts = _full_facts(offer=_ref(1), acceptance=_ref(2))
        msgs = _msgs(1, 2, 3)
        grounded, ungrounded = check_refs_grounded(facts, msgs)
        assert grounded is True
        hard = [u for u in ungrounded if u.get("reason") == "index_not_found"]
        assert hard == []

    def test_out_of_range_ref_not_grounded(self):
        facts = _full_facts(offer=_ref(99))
        msgs = _msgs(1, 2, 3)
        grounded, ungrounded = check_refs_grounded(facts, msgs)
        assert grounded is False
        assert any(u["field"] == "offer" for u in ungrounded)

    def test_no_refs_all_grounded(self):
        facts = _full_facts()
        msgs = _msgs(1, 2)
        grounded, ungrounded = check_refs_grounded(facts, msgs)
        assert grounded is True


# ---------------------------------------------------------------------------
# Input hash canonicalization
# ---------------------------------------------------------------------------

class TestInputHash:
    def _make_input(self, messages: list[dict]) -> dict:
        return {
            "messages": messages,
            "mpesa_extracted": [],
            "case_context": {"user_role": "claimant"},
        }

    def test_same_input_same_hash(self):
        msgs = [{"index": 1, "text": "hello"}, {"index": 2, "text": "world"}]
        ai = self._make_input(msgs)
        h1 = compute_input_hash(_canonical_json(ai))
        h2 = compute_input_hash(_canonical_json(ai))
        assert h1 == h2

    def test_shuffled_messages_same_hash(self):
        msgs_a = [{"index": 1, "text": "hello"}, {"index": 2, "text": "world"}]
        msgs_b = [{"index": 2, "text": "world"}, {"index": 1, "text": "hello"}]
        h1 = compute_input_hash(_canonical_json(self._make_input(msgs_a)))
        h2 = compute_input_hash(_canonical_json(self._make_input(msgs_b)))
        assert h1 == h2

    def test_different_messages_different_hash(self):
        ai_a = self._make_input([{"index": 1, "text": "offer to sell"}])
        ai_b = self._make_input([{"index": 1, "text": "offer to BUY"}])
        h1 = compute_input_hash(_canonical_json(ai_a))
        h2 = compute_input_hash(_canonical_json(ai_b))
        assert h1 != h2


# ---------------------------------------------------------------------------
# _check_4b blank / whitespace fix
# ---------------------------------------------------------------------------
# Note: DeviceFacts Pydantic model validates non-blank at construction time,
# so production code can never pass blank values. We test _check_4b's internal
# logic directly via SimpleNamespace to verify the gate function itself is correct.
# The integration test (test_both_provided_met) uses the real DeviceFacts.

class TestCheck4bGateFix:
    def _real_facts(self):
        return DeviceFacts(
            device_make="Samsung",
            device_model="Galaxy A32",
            declarant_name="Test User",
            declarant_position="Owner",
            phone_number_last4="1234",
            production_method=ProductionMethod.SCREENSHOT_ON_DEVICE,
            lawful_control=True,
            regular_use=True,
            regular_feed=True,
            operating_properly=True,
            messages_on_this_device=True,
            owner_is_declarant=True,
            declaration_confirmed=True,
        )

    def _ns(self, make, model):
        return SimpleNamespace(device_make=make, device_model=model, phone_number_last4=None)

    def test_both_provided_met(self):
        result = _check_4b(self._real_facts())
        assert result.met is True

    def test_blank_make_not_met(self):
        result = _check_4b(self._ns("", "Galaxy A32"))
        assert result.met is False

    def test_whitespace_make_not_met(self):
        result = _check_4b(self._ns("   ", "Galaxy A32"))
        assert result.met is False

    def test_blank_model_not_met(self):
        result = _check_4b(self._ns("Samsung", ""))
        assert result.met is False

    def test_whitespace_model_not_met(self):
        result = _check_4b(self._ns("Samsung", "  \t  "))
        assert result.met is False

    def test_both_blank_not_met(self):
        result = _check_4b(self._ns("", ""))
        assert result.met is False
        # Both remedies should be in the output
        assert any("make" in (r or "").lower() for r in result.basis)
        assert any("model" in (r or "").lower() for r in result.basis)


# ---------------------------------------------------------------------------
# compute_model_disagreements
# ---------------------------------------------------------------------------

class TestModelDisagreements:
    def test_no_disagreement(self):
        opinions = {"C": {"status": "PRESENT", "reason": ""}}
        derived = {"C": "PRESENT"}
        result = compute_model_disagreements(opinions, derived)
        assert result == {}

    def test_disagreement_recorded(self):
        opinions = {"C": {"status": "ABSENT", "reason": "not seen"}}
        derived = {"C": "PRESENT"}
        result = compute_model_disagreements(opinions, derived)
        assert "C" in result
        assert result["C"]["model_status"] == "ABSENT"
        assert result["C"]["derived_status"] == "PRESENT"
