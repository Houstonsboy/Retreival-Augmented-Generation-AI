"""v2.2.0 regression and new-feature tests.

Test groups:
1. M-Pesa cross-match unit tests (new categorised MpesaFinding design)
2. G derivation with scope_dispute_refs
3. K derivation — counter_allegation_refs do not set K=PRESENT
4. A derivation 2.2.0 — claimant declaration lifts status
5. cert_eligibility — confidence floor, G-not-ABSENT, system_reason_codes
6. confidence_band helper
7. timestamp_bonus — M-Pesa datetime counts as FULL
8. Part E merge invariant — M-Pesa screenshots produce messages
9. evidence_manifest structure
10. breach_refs vs counter_allegation_refs (breach direction)
"""

from __future__ import annotations

import pytest

from SCCVision.analyst.rules import (
    CERT_CONFIDENCE_FLOOR,
    MpesaFinding,
    _derive_A,
    cert_eligibility,
    check_mpesa_cross_match,
    check_refs_grounded,
    confidence_band,
    derive_statuses,
    timestamp_bonus,
)
from SCCVision.analyst.enrich import build_analysis_input_from_case
from SCCVision.analyst.dates import compute_date_checks


# ===========================================================================
# 1. M-Pesa cross-match unit tests
# ===========================================================================

class TestMpesaCrossMatch:
    def _ai(self, mpesa_extracted=None, mentioned_amounts=None, messages=None):
        return {
            "mpesa_extracted": mpesa_extracted or [],
            "mentioned_amounts": mentioned_amounts or [],
            "messages": messages or [],
        }

    def test_llm_invented_code_blocks_cert(self):
        facts = {"payments": [{"mpesa_code": "QDF3A123", "amount": 5000}]}
        ai = self._ai(mpesa_extracted=[])
        result = check_mpesa_cross_match(facts, ai)
        assert result["blocks_certificate"] is True
        assert result["consistent"] is False
        findings = result["findings"]
        assert any(f["kind"] == "LLM_INVENTED_CODE" and f["blocks_certificate"] for f in findings)

    def test_parser_only_does_not_block(self):
        facts = {"payments": []}  # LLM cited nothing
        ai = self._ai(mpesa_extracted=[{"code": "QDF3A123", "amount_kes": 5000}])
        result = check_mpesa_cross_match(facts, ai)
        assert result["blocks_certificate"] is False
        assert result["consistent"] is True
        assert any(f["kind"] == "PARSER_ONLY" for f in result["findings"])

    def test_duplicate_same_txn_does_not_block(self):
        # Same code appears twice in parser (duplicate screenshot)
        facts = {"payments": [{"mpesa_code": "QDF3A123", "amount": 5000}]}
        ai = self._ai(mpesa_extracted=[
            {"code": "QDF3A123", "amount_kes": 5000},
            {"code": "QDF3A123", "amount_kes": 5000},
        ])
        result = check_mpesa_cross_match(facts, ai)
        assert result["blocks_certificate"] is False
        assert any(f["kind"] == "DUPLICATE_SAME_TXN" for f in result["findings"])

    def test_duplicate_conflict_blocks_cert(self):
        # Same code with different amounts → conflict
        facts = {"payments": [{"mpesa_code": "QDF3A123", "amount": 5000}]}
        ai = self._ai(mpesa_extracted=[
            {"code": "QDF3A123", "amount_kes": 5000},
            {"code": "QDF3A123", "amount_kes": 2500},
        ])
        result = check_mpesa_cross_match(facts, ai)
        assert result["blocks_certificate"] is True
        assert any(f["kind"] == "DUPLICATE_CONFLICT" and f["blocks_certificate"] for f in result["findings"])

    def test_llm_invented_field_blocks_cert(self):
        # Code exists in parser but LLM cites a different amount
        facts = {"payments": [{"mpesa_code": "QDF3A123", "amount": 10000}]}
        ai = self._ai(mpesa_extracted=[{"code": "QDF3A123", "amount_kes": 5000}])
        result = check_mpesa_cross_match(facts, ai)
        assert result["blocks_certificate"] is True
        assert any(f["kind"] == "LLM_INVENTED_FIELD" and f["blocks_certificate"] for f in result["findings"])

    def test_counterparty_mismatch_does_not_block(self):
        facts = {"payments": []}
        messages = [{"text": "pay to till 123456", "index": 1}]
        ai = self._ai(
            mpesa_extracted=[{"code": "ABC001", "amount_kes": 1000, "till_or_paybill": "999999"}],
            messages=messages,
        )
        result = check_mpesa_cross_match(facts, ai)
        assert result["blocks_certificate"] is False
        assert any(f["kind"] == "COUNTERPARTY_MISMATCH" for f in result["findings"])

    def test_unexplained_amount_does_not_block(self):
        # Parser found KES 1500 but it's never mentioned in conversation
        facts = {"payments": []}
        ai = self._ai(
            mpesa_extracted=[{"code": "XYZ123", "amount_kes": 1500}],
            mentioned_amounts=[{"amount": 5000}],
        )
        result = check_mpesa_cross_match(facts, ai)
        assert result["blocks_certificate"] is False
        assert any(f["kind"] == "UNEXPLAINED_AMOUNT" for f in result["findings"])

    def test_matching_code_and_amount_no_findings(self):
        # LLM and parser agree on code and amount → no fabrication findings
        facts = {"payments": [{"mpesa_code": "QDF3A123", "amount": 5000}]}
        ai = self._ai(
            mpesa_extracted=[{"code": "QDF3A123", "amount_kes": 5000}],
            mentioned_amounts=[{"amount": 5000}],
        )
        result = check_mpesa_cross_match(facts, ai)
        assert result["blocks_certificate"] is False
        assert result["consistent"] is True
        blocking = [f for f in result["findings"] if f["blocks_certificate"]]
        assert blocking == []

    def test_no_mpesa_data_at_all(self):
        facts = {"payments": []}
        result = check_mpesa_cross_match(facts, {"mpesa_extracted": [], "messages": []})
        assert result["blocks_certificate"] is False
        assert result["consistent"] is True

    def test_mpesafinding_model_valid(self):
        f = MpesaFinding(
            code="ABC123",
            kind="LLM_INVENTED_CODE",
            amount_kes=5000,
            detail="test",
            blocks_certificate=True,
        )
        d = f.model_dump()
        assert d["kind"] == "LLM_INVENTED_CODE"
        assert d["blocks_certificate"] is True


# ===========================================================================
# 2. G derivation with scope_dispute_refs
# ===========================================================================

class TestGDerivation:
    def _f(self, **kw):
        base = {
            "offer": None, "offer_by": "UNCLEAR", "acceptance": None,
            "acceptance_by": "UNCLEAR", "price_ref": None, "price_amount": None,
            "subject_matter_ref": None, "quantity_or_scope_ref": None,
            "due_date_ref": None, "performance_refs": [], "breach_refs": [],
            "scope_dispute_refs": [], "counter_allegation_refs": [], "amount_msgs": [],
            "payments": [], "social_or_domestic_context": False,
            "capacity_concern_refs": [], "consent_concern_refs": [],
            "illegality_concern_refs": [], "formality_required": False,
            "integrity_concerns": [], "element_opinions": {}, "narrative": {},
            "party_identity_refs": {}, "claimant": "UNCLEAR", "respondent": "UNCLEAR",
        }
        base.update(kw)
        return base

    def test_g_present_when_price_and_subject_no_dispute(self):
        f = self._f(
            price_ref={"msg_index": 1},
            subject_matter_ref={"msg_index": 2},
        )
        st = derive_statuses(f, [], {})
        assert st["G"] == "PRESENT"

    def test_g_partial_when_scope_dispute_even_with_both_refs(self):
        f = self._f(
            price_ref={"msg_index": 1},
            subject_matter_ref={"msg_index": 2},
            scope_dispute_refs=[{"msg_index": 3}],
        )
        st = derive_statuses(f, [], {})
        assert st["G"] == "PARTIAL"

    def test_g_absent_when_scope_dispute_and_no_refs(self):
        f = self._f(scope_dispute_refs=[{"msg_index": 3}])
        st = derive_statuses(f, [], {})
        assert st["G"] == "ABSENT"

    def test_g_partial_when_only_price_ref(self):
        f = self._f(price_ref={"msg_index": 1})
        st = derive_statuses(f, [], {})
        assert st["G"] == "PARTIAL"

    def test_g_absent_when_no_refs(self):
        f = self._f()
        st = derive_statuses(f, [], {})
        assert st["G"] == "ABSENT"


# ===========================================================================
# 3. K derivation — counter_allegations do NOT set K=PRESENT
# ===========================================================================

class TestKDerivation:
    def _f(self, breach_refs=None, counter_allegation_refs=None, performance_refs=None):
        return {
            "offer": None, "offer_by": "UNCLEAR", "acceptance": None,
            "acceptance_by": "UNCLEAR", "price_ref": None, "price_amount": None,
            "subject_matter_ref": None, "quantity_or_scope_ref": None,
            "due_date_ref": None, "performance_refs": performance_refs or [],
            "breach_refs": breach_refs or [],
            "scope_dispute_refs": [], "counter_allegation_refs": counter_allegation_refs or [],
            "amount_msgs": [], "payments": [], "social_or_domestic_context": False,
            "capacity_concern_refs": [], "consent_concern_refs": [],
            "illegality_concern_refs": [], "formality_required": False,
            "integrity_concerns": [], "element_opinions": {}, "narrative": {},
            "party_identity_refs": {}, "claimant": "UNCLEAR", "respondent": "UNCLEAR",
        }

    def test_k_present_when_breach_refs_exist(self):
        f = self._f(breach_refs=[{"msg_index": 5}])
        st = derive_statuses(f, [], {})
        assert st["K"] == "PRESENT"

    def test_k_unclear_when_only_counter_allegation(self):
        # Counter-allegation ("I haven't received payment") is NOT a breach ref
        f = self._f(counter_allegation_refs=[{"msg_index": 5}])
        st = derive_statuses(f, [], {})
        assert st["K"] == "UNCLEAR"

    def test_k_partial_when_only_performance_refs(self):
        f = self._f(performance_refs=[{"msg_index": 3}])
        st = derive_statuses(f, [], {})
        assert st["K"] == "PARTIAL"

    def test_k_unclear_when_no_refs_at_all(self):
        f = self._f()
        st = derive_statuses(f, [], {})
        assert st["K"] == "UNCLEAR"


# ===========================================================================
# 4. A derivation 2.2.0
# ===========================================================================

class TestADerivation:
    def _f(self, party_identity_refs=None, claimant="UNCLEAR", respondent="UNCLEAR"):
        return {
            "party_identity_refs": party_identity_refs or {},
            "claimant": claimant,
            "respondent": respondent,
        }

    def test_a_present_when_both_in_evidence(self):
        f = self._f(party_identity_refs={"PARTY_1": [{"msg_index": 1}], "PARTY_2": [{"msg_index": 2}]})
        status, basis = _derive_A(f, None)
        assert status == "PRESENT"

    def test_a_partial_when_one_in_evidence(self):
        f = self._f(party_identity_refs={"PARTY_1": [{"msg_index": 1}]})
        status, basis = _derive_A(f, None)
        assert status == "PARTIAL"

    def test_a_unclear_when_none_identified(self):
        f = self._f()
        status, basis = _derive_A(f, None)
        assert status == "UNCLEAR"

    def test_a_partial_when_claimant_declared_no_respondent(self):
        f = self._f(claimant="PARTY_1", respondent="PARTY_2")
        status, basis = _derive_A(f, {"claimant_full_name": "John Doe"})
        assert status == "PARTIAL"

    def test_a_present_when_claimant_declared_and_respondent_in_evidence(self):
        f = self._f(
            party_identity_refs={"PARTY_2": [{"msg_index": 5}]},
            claimant="PARTY_1",
            respondent="PARTY_2",
        )
        status, basis = _derive_A(f, {"claimant_full_name": "John Doe"})
        assert status == "PRESENT"
        assert "declared" in basis

    def test_a_present_still_when_both_in_evidence_and_declared(self):
        f = self._f(
            party_identity_refs={"PARTY_1": [{"msg_index": 1}], "PARTY_2": [{"msg_index": 2}]},
        )
        status, basis = _derive_A(f, {"claimant_full_name": "Jane Doe"})
        assert status == "PRESENT"


# ===========================================================================
# 5. cert_eligibility — confidence floor, G-not-ABSENT, system_reason_codes
# ===========================================================================

class TestCertEligibility220:
    def _eligible_st(self):
        return {
            "A": "PRESENT", "B": "NO_ISSUE_INDICATED", "C": "PRESENT", "D": "PRESENT",
            "E": "PRESENT", "F": "PRESENT", "G": "PRESENT", "H": "NO_ISSUE_INDICATED",
            "I": "NO_ISSUE_INDICATED", "J": "NOT_REQUIRED", "K": "PRESENT",
        }

    def test_confidence_floor_blocks_cert(self):
        result = cert_eligibility(
            verdict="LIKELY VALID",
            st=self._eligible_st(),
            unstable_core=set(),
            integrity="STRONG",
            refs_grounded=True,
            mpesa_consistent=True,
            incomplete=False,
            verified=True,
            confidence=0.65,  # below 0.70 floor
        )
        assert result["eligible"] is False
        assert any("confidence" in r for r in result["reasons"])

    def test_confidence_floor_passes_above_threshold(self):
        result = cert_eligibility(
            verdict="LIKELY VALID",
            st=self._eligible_st(),
            unstable_core=set(),
            integrity="STRONG",
            refs_grounded=True,
            mpesa_consistent=True,
            incomplete=False,
            verified=True,
            confidence=0.72,
        )
        assert result["eligible"] is True

    def test_confidence_floor_not_applied_without_confidence_param(self):
        # Callers that don't pass confidence (old code) should still work
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

    def test_g_absent_blocks_cert(self):
        st = dict(self._eligible_st())
        st["G"] = "ABSENT"
        result = cert_eligibility(
            verdict="LIKELY VALID", st=st,
            unstable_core=set(), integrity="STRONG",
            refs_grounded=True, mpesa_consistent=True,
            incomplete=False, verified=True,
        )
        assert result["eligible"] is False
        assert any("G absent" in r for r in result["reasons"])

    def test_incomplete_is_system_reason(self):
        result = cert_eligibility(
            verdict="LIKELY VALID", st=self._eligible_st(),
            unstable_core=set(), integrity="STRONG",
            refs_grounded=True, mpesa_consistent=True,
            incomplete=True, verified=True,
        )
        assert result["eligible"] is False
        sys_codes = result["system_reason_codes"]
        assert any("incomplete" in c for c in sys_codes)

    def test_ungrounded_refs_is_system_reason(self):
        result = cert_eligibility(
            verdict="LIKELY VALID", st=self._eligible_st(),
            unstable_core=set(), integrity="STRONG",
            refs_grounded=False, mpesa_consistent=True,
            incomplete=False, verified=True,
        )
        assert result["eligible"] is False
        assert any("ungrounded" in c for c in result["system_reason_codes"])

    def test_non_system_reasons_not_in_system_codes(self):
        st = dict(self._eligible_st())
        st["C"] = "PARTIAL"
        result = cert_eligibility(
            verdict="LIKELY VALID", st=st,
            unstable_core=set(), integrity="STRONG",
            refs_grounded=True, mpesa_consistent=True,
            incomplete=False, verified=True,
        )
        assert result["eligible"] is False
        # C not established is NOT a system reason
        assert not any("element C" in c for c in result["system_reason_codes"])

    def test_mpesa_fabrication_findings_block(self):
        result = cert_eligibility(
            verdict="LIKELY VALID", st=self._eligible_st(),
            unstable_core=set(), integrity="STRONG",
            refs_grounded=True, mpesa_consistent=True,
            incomplete=False, verified=True,
            mpesa_findings=[{"blocks_certificate": True, "kind": "LLM_INVENTED_CODE"}],
        )
        assert result["eligible"] is False
        assert any("M-Pesa" in r for r in result["reasons"])

    def test_non_blocking_mpesa_findings_do_not_block(self):
        result = cert_eligibility(
            verdict="LIKELY VALID", st=self._eligible_st(),
            unstable_core=set(), integrity="STRONG",
            refs_grounded=True, mpesa_consistent=True,
            incomplete=False, verified=True,
            confidence=0.75,
            mpesa_findings=[{"blocks_certificate": False, "kind": "PARSER_ONLY"}],
        )
        assert result["eligible"] is True

    def test_cert_confidence_floor_constant(self):
        assert CERT_CONFIDENCE_FLOOR == 0.70


# ===========================================================================
# 6. confidence_band helper
# ===========================================================================

class TestConfidenceBand:
    def test_high(self):
        assert confidence_band(0.90) == "HIGH"
        assert confidence_band(0.85) == "HIGH"

    def test_medium(self):
        assert confidence_band(0.75) == "MEDIUM"
        assert confidence_band(0.70) == "MEDIUM"

    def test_low(self):
        assert confidence_band(0.60) == "LOW"
        assert confidence_band(0.50) == "LOW"

    def test_very_low(self):
        assert confidence_band(0.40) == "VERY_LOW"
        assert confidence_band(0.00) == "VERY_LOW"


# ===========================================================================
# 7. timestamp_bonus — M-Pesa datetime counts as FULL
# ===========================================================================

class TestTimestampBonusMpesa:
    def test_mpesa_datetime_counts_as_full(self):
        # Message index 5 has no timestamp in the chat...
        messages = [{"index": 5, "text": "paid", "timestamp_quality": "NONE"}]
        # ...but the M-Pesa parser entry for index 5 has a datetime
        mpesa_events = [{"msg_index": 5, "code": "ABC123", "datetime": "2024-11-01 14:23"}]
        bonus, info = timestamp_bonus([5], messages, mpesa_events)
        assert info["coverage"] == 1.0  # FULL score for the M-Pesa datetime
        assert bonus > 0.0

    def test_mpesa_no_datetime_no_bonus(self):
        messages = [{"index": 5, "text": "paid", "timestamp_quality": "NONE"}]
        mpesa_events = [{"msg_index": 5, "code": "ABC123", "datetime": None}]
        bonus, info = timestamp_bonus([5], messages, mpesa_events)
        assert info["coverage"] == 0.0
        assert bonus == 0.0

    def test_chat_timestamp_quality_still_works(self):
        messages = [{"index": 3, "text": "offer", "timestamp_quality": "FULL"}]
        bonus, info = timestamp_bonus([3], messages, [])
        assert info["coverage"] == 1.0
        assert bonus > 0.0


# ===========================================================================
# 8. Part E merge invariant — M-Pesa screenshots produce messages
# ===========================================================================

class TestMergeInvariant:
    def _case_file(self, evidence):
        return {
            "case_id": "test-case",
            "case_context": {"user_role": "claimant"},
            "evidence": evidence,
            "user_provided_facts": [],
            "analyses": [],
            "requests": [],
        }

    def test_mpesa_screenshot_produces_message(self):
        case_file = self._case_file([
            {
                "evidence_id": "ev-mpesa-1",
                "type": "mpesa_screenshot",
                "parsed": [{
                    "code": "QDF3A123",
                    "amount_kes": 27750,
                    "recipient": "NEPTUNE SHELTERS",
                    "datetime": "2024-11-01 14:23",
                    "parse_confidence": 0.95,
                }],
            }
        ])
        date_checks = compute_date_checks(case_file)
        ai = build_analysis_input_from_case(case_file, date_checks)
        messages = ai.get("messages") or []
        assert len(messages) >= 1
        # The M-Pesa confirmation should produce a SYSTEM_SMS message
        system_msgs = [m for m in messages if m.get("msg_kind") == "SYSTEM_SMS"]
        assert len(system_msgs) >= 1
        # The recipient name should appear in the message text
        combined_text = " ".join(m.get("text", "") for m in system_msgs)
        assert "QDF3A123" in combined_text or "27,750" in combined_text

    def test_mpesa_screenshot_also_in_mpesa_extracted(self):
        case_file = self._case_file([
            {
                "evidence_id": "ev-mpesa-1",
                "type": "mpesa_screenshot",
                "parsed": [{"code": "QDF3A123", "amount_kes": 5000}],
            }
        ])
        date_checks = compute_date_checks(case_file)
        ai = build_analysis_input_from_case(case_file, date_checks)
        # M-Pesa data should ALSO be in mpesa_extracted
        mpesa = ai.get("mpesa_extracted") or []
        assert any(p.get("code") == "QDF3A123" for p in mpesa)

    def test_chat_screenshot_messages_unaffected(self):
        # Existing chat screenshot processing must still produce messages
        chat_ocr = {
            "messages": [
                {"text": "Hello", "sender": "LEFT"},
                {"text": "Hi there", "sender": "RIGHT"},
            ],
        }
        case_file = self._case_file([
            {
                "evidence_id": "ev-chat-1",
                "type": "chat_screenshot",
                "ocr": chat_ocr,
            }
        ])
        date_checks = compute_date_checks(case_file)
        ai = build_analysis_input_from_case(case_file, date_checks)
        chat_msgs = [m for m in (ai.get("messages") or []) if m.get("msg_kind") != "SYSTEM_SMS"]
        assert len(chat_msgs) == 2


# ===========================================================================
# 9. evidence_manifest structure
# ===========================================================================

class TestEvidenceManifest:
    def _case_file(self, evidence):
        return {
            "case_id": "test-case",
            "case_context": {"user_role": "claimant"},
            "evidence": evidence,
            "user_provided_facts": [],
            "analyses": [],
            "requests": [],
        }

    def test_manifest_has_entry_per_evidence(self):
        case_file = self._case_file([
            {"evidence_id": "ev-1", "type": "chat_screenshot", "ocr": {"messages": []}},
            {"evidence_id": "ev-2", "type": "mpesa_screenshot", "parsed": []},
        ])
        date_checks = compute_date_checks(case_file)
        ai = build_analysis_input_from_case(case_file, date_checks)
        manifest = ai.get("evidence_manifest") or []
        assert len(manifest) == 2
        ids = {e["evidence_id"] for e in manifest}
        assert ids == {"ev-1", "ev-2"}

    def test_manifest_entry_has_required_fields(self):
        case_file = self._case_file([
            {"evidence_id": "ev-1", "type": "chat_screenshot", "file_hash": "abc123",
             "ocr": {"messages": []}, "uploaded_round": 1},
        ])
        date_checks = compute_date_checks(case_file)
        ai = build_analysis_input_from_case(case_file, date_checks)
        entry = ai["evidence_manifest"][0]
        assert entry["evidence_id"] == "ev-1"
        assert entry["file_hash"] == "abc123"
        assert "message_index_range" in entry
        assert "mpesa_codes" in entry

    def test_manifest_dedup_flagged(self):
        case_file = self._case_file([
            {"evidence_id": "ev-1", "type": "chat_screenshot", "file_hash": "SAMEHASH",
             "ocr": {"messages": []}},
            {"evidence_id": "ev-2", "type": "chat_screenshot", "file_hash": "SAMEHASH",
             "ocr": {"messages": []}},
        ])
        date_checks = compute_date_checks(case_file)
        ai = build_analysis_input_from_case(case_file, date_checks)
        manifest = ai["evidence_manifest"]
        second = next(e for e in manifest if e["evidence_id"] == "ev-2")
        assert second.get("duplicate_of") == "ev-1"

    def test_manifest_mpesa_codes_recorded(self):
        case_file = self._case_file([
            {
                "evidence_id": "ev-m",
                "type": "mpesa_screenshot",
                "parsed": [
                    {"code": "CODE001", "amount_kes": 100},
                    {"code": "CODE002", "amount_kes": 200},
                ],
            }
        ])
        date_checks = compute_date_checks(case_file)
        ai = build_analysis_input_from_case(case_file, date_checks)
        entry = next(e for e in ai["evidence_manifest"] if e["evidence_id"] == "ev-m")
        assert "CODE001" in entry["mpesa_codes"]
        assert "CODE002" in entry["mpesa_codes"]


# ===========================================================================
# 10. Breach direction — only respondent-failed-to-perform counts
# ===========================================================================

class TestBreachDirection:
    def _make_facts(self, breach_refs, counter_refs):
        return {
            "offer": {"msg_index": 1}, "offer_by": "PARTY_1",
            "acceptance": {"msg_index": 2}, "acceptance_by": "PARTY_2",
            "price_ref": {"msg_index": 3}, "price_amount": 5000,
            "subject_matter_ref": {"msg_index": 4},
            "quantity_or_scope_ref": None, "due_date_ref": None,
            "performance_refs": [], "breach_refs": breach_refs,
            "scope_dispute_refs": [], "counter_allegation_refs": counter_refs,
            "amount_msgs": [], "payments": [], "social_or_domestic_context": False,
            "capacity_concern_refs": [], "consent_concern_refs": [],
            "illegality_concern_refs": [], "formality_required": False,
            "integrity_concerns": [], "element_opinions": {}, "narrative": {},
            "party_identity_refs": {"PARTY_1": [{"msg_index": 1}], "PARTY_2": [{"msg_index": 2}]},
            "claimant": "PARTY_1", "respondent": "PARTY_2",
        }

    def test_respondent_breach_sets_k_present(self):
        f = self._make_facts(
            breach_refs=[{"msg_index": 7}],
            counter_refs=[],
        )
        st = derive_statuses(f, [], {})
        assert st["K"] == "PRESENT"

    def test_claimant_counter_allegation_does_not_set_k_present(self):
        f = self._make_facts(
            breach_refs=[],
            counter_refs=[{"msg_index": 7}],
        )
        st = derive_statuses(f, [], {})
        # counter_allegation goes in counter_allegation_refs — does not make K=PRESENT
        assert st["K"] == "UNCLEAR"

    def test_scope_dispute_does_not_set_k_present(self):
        f = self._make_facts(
            breach_refs=[],
            counter_refs=[],
        )
        f["scope_dispute_refs"] = [{"msg_index": 8}]
        st = derive_statuses(f, [], {})
        assert st["K"] == "UNCLEAR"
        # scope_dispute should make G at most PARTIAL (both price+subject exist)
        assert st["G"] == "PARTIAL"

    def test_check_refs_grounded_includes_new_fields(self):
        messages = [
            {"msg_index": 1, "text": "offer"},
            {"msg_index": 2, "text": "accepted"},
            {"msg_index": 3, "text": "price"},
            {"msg_index": 4, "text": "subject"},
            {"msg_index": 5, "text": "new charge"},
            {"msg_index": 6, "text": "you owe more"},
        ]
        f = self._make_facts(
            breach_refs=[{"msg_index": 1, "span": "offer"}],
            counter_refs=[{"msg_index": 6}],
        )
        f["scope_dispute_refs"] = [{"msg_index": 5}]
        all_grounded, ungrounded = check_refs_grounded(f, messages)
        # All refs point to valid indices → should be grounded
        hard_failures = [u for u in ungrounded if u.get("reason") == "index_not_found"]
        assert hard_failures == []
