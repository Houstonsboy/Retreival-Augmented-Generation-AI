"""
Unit tests for the SCCVision Contract Evidence Analyst.

No network or OCR is required: the LLM is stubbed and the enrichment/verification
layers are exercised with synthetic WhatsApp data.

Run with:
    cd backend
    PYTHONPATH=. pytest tests/test_sccvision_analyst.py -v
"""

import os
import sys

_BACKEND = os.path.dirname(os.path.dirname(__file__))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from SCCVision.analyst.dates import compute_date_checks  # noqa: E402
from SCCVision.analyst.enrich import (  # noqa: E402
    build_analysis_input,
    build_analysis_input_from_case,
    build_integrity_flags,
    extract_amounts,
    mask_id_number,
    mask_phone,
    mask_pii,
    merge_ocr_results,
    parse_mpesa_message,
)
from SCCVision.analyst.models import (  # noqa: E402
    AnalystResult,
    ElementStatus,
    OverallVerdict,
    PaymentClassification,
)
from SCCVision.analyst.mpssa import (  # noqa: E402
    classify_evidence,
    find_codes,
    parse_mpesa_screenshot,
)
from SCCVision.analyst.pipeline import analyze, analyze_case  # noqa: E402
from SCCVision.analyst.rules import (  # noqa: E402
    build_deterministic_followups,
    check_mpesa_cross_match,
    check_quotes_grounded,
    compute_verdict,
    merge_followups,
)
from SCCVision.analyst.verify import verify  # noqa: E402


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def test_mask_phone():
    assert mask_phone("call +254721445100 now") == "call 2547***100 now"
    assert mask_phone("0721445100") == "0721***100"


def test_mask_id_number():
    assert mask_id_number("ID 12345678") == "ID 123***678"


def test_mask_pii_phones_before_ids():
    out = mask_pii("call 0721445100 id 12345678")
    assert "0721***100" in out
    assert "123***678" in out


def test_parse_mpesa_full_confirmation():
    text = (
        "QHB9MQD7R2 Confirmed. Ksh2,500.00 sent to NEPTUNE SHELTERS CATERERS "
        "+254721445100 on 8/11/26 at 6:15 PM. New M-PESA balance is Ksh4,180.50."
    )
    p = parse_mpesa_message(text, 3)
    assert p is not None
    assert p["code"] == "QHB9MQD7R2"
    assert p["amount_kes"] == 2500.0
    assert p["recipient_number"] == "+254721445100"
    assert p["parse_confidence"] == 1.0


def test_parse_mpesa_requires_confirmed():
    assert parse_mpesa_message("Ksh2,500.00 sent to Joe", 1) is None


def test_extract_amounts_keyword_quote():
    amounts = extract_amounts("I charge 27750 for my services")
    assert any(a["value"] == 27750.0 for a in amounts)


def test_integrity_flags_missing_dates():
    flags = build_integrity_flags(
        [{"index": 1, "sender": "me", "text": "hi", "timestamp": "10:00"}],
        [],
        image_count=1,
    )
    assert "missing_dates" in flags


def test_integrity_flags_edited_and_backwards():
    flags = build_integrity_flags(
        [
            {"index": 1, "sender": "me", "text": "a", "timestamp": "11:00", "edited": True},
            {"index": 2, "sender": "me", "text": "b", "timestamp": "10:00"},
        ],
        [{"text": "8 Nov"}],
        image_count=1,
    )
    assert "edited_messages" in flags
    assert "time_order_backwards" in flags


def test_merge_dedupes_boundary():
    m = {"index": 1, "sender": "me", "text": "hello", "timestamp": "09:00"}
    merged = merge_ocr_results(
        [
            {"contact": {"name": "Neptune"}, "messages": [dict(m)], "system_events": []},
            {"contact": {"name": "Neptune"}, "messages": [dict(m)], "system_events": []},
        ]
    )
    assert len(merged["messages"]) == 1
    assert merged["messages"][0]["index"] == 1


def test_build_analysis_input_masks_pii():
    cleaned = {
        "contact": {"name": "Neptune Shelters", "presence": "online"},
        "messages": [
            {
                "index": 1, "sender": "Neptune Shelters", "sender_confidence": "high",
                "text": "pay +254721445100", "timestamp": "10:00", "edited": False,
                "avg_confidence": 0.9,
            }
        ],
        "system_events": [],
    }
    out = build_analysis_input(cleaned, {"user_role": "claimant", "user_account": "buyer"}, image_count=1)
    assert "2547***100" in out["messages"][0]["text"]
    assert out["messages"][0]["sender"] == "them"
    assert out["messages"][0]["sender_name"] == "Neptune Shelters"


# ---------------------------------------------------------------------------
# Rules / verification
# ---------------------------------------------------------------------------

def _result(**overrides):
    base = {
        "conversation_integrity": {"rating": "ADEQUATE", "issues": []},
        "elements": {},
        "mpesa_payments": [],
        "overall_verdict": "LIKELY VALID",
    }
    base.update(overrides)
    return AnalystResult.model_validate(base)


def test_quote_grounding_detects_unverifiable_quote():
    result = _result(
        elements={
            "C": {
                "status": "PRESENT",
                "evidence": [{"msg_index": 1, "quote": "this text does not exist"}],
                "reasoning": "x",
                "confidence": 0.9,
            }
        }
    )
    ai = {"messages": [{"index": 1, "text": "I charge 27750 for my services"}]}
    ungrounded = check_quotes_grounded(result, ai)
    assert len(ungrounded) == 1
    assert ungrounded[0]["reason"] == "quote_not_found"


def test_quote_grounding_accepts_verbatim():
    result = _result(
        elements={
            "C": {
                "status": "PRESENT",
                "evidence": [{"msg_index": 1, "quote": "I charge 27750"}],
                "reasoning": "x",
                "confidence": 0.9,
            }
        }
    )
    ai = {"messages": [{"index": 1, "text": "I charge 27750 for my services"}]}
    assert check_quotes_grounded(result, ai) == []


def test_mpesa_cross_match_flags_llm_only():
    result = _result(
        mpesa_payments=[
            {"code": "FAKE123456", "amount_kes": 100.0, "classification": "VERIFIED-LOOKING"}
        ]
    )
    ai = {"mpesa_extracted": [{"code": "QHB9MQD7R2", "amount_kes": 2500.0}]}
    cross = check_mpesa_cross_match(result, ai)
    assert cross["codes_llm_only"] == ["FAKE123456"]
    assert cross["codes_parsed_only"] == ["QHB9MQD7R2"]
    assert cross["consistent"] is False


def test_mpesa_cross_match_reclassifies_unmatched_confirmation():
    result = _result(
        mpesa_payments=[
            {
                "code": "FAKE123456",
                "amount_kes": 100.0,
                "datetime": "8/11/26 6:15 PM",
                "basis": ["confirmation"],
                "classification": "VERIFIED-LOOKING",
            }
        ]
    )
    ai = {"mpesa_extracted": [], "mentioned_amounts": [], "messages": []}
    cross = check_mpesa_cross_match(result, ai)
    assert cross["consistent"] is False
    assert cross["invented_fields"] is True
    assert result.mpesa_payments[0].classification is PaymentClassification.CLAIMED_ONLY
    assert result.mpesa_payments[0].code is None
    assert result.mpesa_payments[0].datetime is None


def test_mpesa_cross_match_keeps_verified_match():
    result = _result(
        mpesa_payments=[
            {
                "code": "QHB9MQD7R2",
                "amount_kes": 2500.0,
                "basis": ["confirmation"],
                "classification": "VERIFIED-LOOKING",
            }
        ]
    )
    ai = {
        "mpesa_extracted": [{"code": "QHB9MQD7R2", "amount_kes": 2500.0}],
        "mentioned_amounts": [],
        "messages": [],
    }
    cross = check_mpesa_cross_match(result, ai)
    assert cross["consistent"] is True
    assert result.mpesa_payments[0].classification is PaymentClassification.VERIFIED_LOOKING


def test_compute_verdict_downgrades_likely_valid_on_weak_integrity():
    result = _result(conversation_integrity={"rating": "WEAK", "issues": ["missing dates"]})
    verdict, down, reasons = compute_verdict(result, ["missing_dates"], {"no_dates": False})
    assert down is True
    assert verdict == OverallVerdict.POSSIBLY_VALID_WITH_GAPS
    assert "record_integrity_weak" in reasons


def test_compute_verdict_no_downgrade_for_non_likely_valid():
    result = _result(overall_verdict="INSUFFICIENT EVIDENCE")
    verdict, down, reasons = compute_verdict(result, ["missing_dates"], {"no_dates": True})
    assert down is False
    assert verdict == OverallVerdict.INSUFFICIENT_EVIDENCE


def test_compute_verdict_caps_likely_valid_when_no_dates():
    result = _result(overall_verdict="LIKELY VALID")
    verdict, down, reasons = compute_verdict(result, [], {"no_dates": True})
    assert down is True
    assert verdict == OverallVerdict.POSSIBLY_VALID_WITH_GAPS
    assert "missing_dates" in reasons


def test_element_status_enforcement_no_evidence_downgrade():
    from SCCVision.analyst.rules import check_element_statuses

    result = _result(
        elements={"C": {"status": "PRESENT", "evidence": [], "reasoning": "x", "confidence": 0.9}}
    )
    issues = check_element_statuses(result)
    assert result.elements["C"].status is ElementStatus.UNCLEAR
    assert any(i["issue"] == "no_evidence" for i in issues)


def test_element_status_restricts_special_statuses():
    from SCCVision.analyst.rules import check_element_statuses

    result = _result(
        elements={
            "C": {"status": "NO_ISSUE_INDICATED", "evidence": [], "reasoning": "x", "confidence": 0.9},
            "J": {"status": "NOT_REQUIRED", "evidence": [], "reasoning": "x", "confidence": 0.9},
        }
    )
    check_element_statuses(result)
    assert result.elements["C"].status is ElementStatus.UNCLEAR
    assert result.elements["J"].status is ElementStatus.NOT_REQUIRED


# ---------------------------------------------------------------------------
# Pipeline (stubbed LLM)
# ---------------------------------------------------------------------------

class _FakeLLM:
    model = "test-model"

    def complete(self, analysis_input):
        return _result(
            elements={
                "C": {
                    "status": "PARTIAL",
                    "evidence": [{"msg_index": 1, "quote": "I charge 27750"}],
                    "reasoning": "price quoted",
                    "confidence": 0.8,
                }
            },
            overall_verdict="POSSIBLY VALID WITH GAPS",
        )


def test_pipeline_returns_analysis_and_verification():
    ocr = [{
        "contact": {"name": "Neptune Shelters"},
        "messages": [
            {"index": 1, "sender": "Neptune Shelters", "sender_confidence": "high",
             "text": "I charge 27750 for my services", "timestamp": "10:01",
             "edited": True, "avg_confidence": 0.95, "incomplete": False},
        ],
        "system_events": [],
    }]
    out = analyze(ocr, {"user_role": "claimant", "user_account": "buyer"}, llm=_FakeLLM())
    assert out["result"]["overall_verdict"] == "POSSIBLY VALID WITH GAPS"
    assert out["verification"]["quotes_grounded"] is True
    assert out["prompt_version"]
    assert out["model"] == "test-model"


def test_pipeline_rejects_empty_ocr():
    import pytest
    with pytest.raises(ValueError):
        analyze([], llm=_FakeLLM())


# ---------------------------------------------------------------------------
# M-Pesa screenshot parsing (mpssa)
# ---------------------------------------------------------------------------

def test_find_codes_requires_letter_and_digit():
    codes = find_codes("QHB9MQD7R2 confirmed. 1234567890 not-a-code ABCDEFGHIJ")
    assert codes == ["QHB9MQD7R2"]


def test_classify_evidence_mpesa_screenshot():
    ocr = {
        "full_text": "QHB9MQD7R2 Confirmed. Ksh2,500.00 sent to NEPTUNE SHELTERS "
                     "+254721445100 on 8/11/26 at 6:15 PM. New M-PESA balance is Ksh4,180.50.",
        "messages": [],
        "contact": {},
    }
    assert classify_evidence(ocr) == "mpesa_screenshot"


def test_parse_mpesa_screenshot_multiple_entries():
    ocr = {
        "full_text": (
            "QHB9MQD7R2 Confirmed. Ksh2,500.00 sent to NEPTUNE SHELTERS "
            "+254721445100 on 8/11/26 at 6:15 PM. New M-PESA balance is Ksh4,180.50. "
            "RTY6UJ8K2L Confirmed. Ksh1,000.00 sent to NEPTUNE SHELTERS "
            "+254721445100 on 9/11/26 at 7:00 AM."
        ),
        "messages": [],
        "contact": {},
    }
    out = parse_mpesa_screenshot(ocr, "e-mpesa-1")
    assert out["layout"] == "supported"
    assert [e["code"] for e in out["entries"]] == ["QHB9MQD7R2", "RTY6UJ8K2L"]
    assert out["entries"][0]["amount_kes"] == 2500.0
    assert out["entries"][0]["evidence_id"] == "e-mpesa-1"


def test_parse_mpesa_screenshot_unsupported_without_codes():
    out = parse_mpesa_screenshot({"full_text": "no transaction here", "messages": [], "contact": {}})
    assert out["layout"] == "unsupported_layout"
    assert out["raw_text"] == "no transaction here"


def test_parse_mpesa_message_bank_deposit():
    text = (
        "Dear Kristein Gichuhi Mwaura, You have deposited KES27750.0 MPESA "
        "ref: UJV4PO6ABO to account XX4151 for NEPTUNE SHELTERS LIMITED "
        "through Absa Bank Kenya PLC"
    )
    p = parse_mpesa_message(text, 3)
    assert p is not None
    assert p["code"] == "UJV4PO6ABO"
    assert p["amount_kes"] == 27750.0
    assert p["recipient"] == "NEPTUNE SHELTERS LIMITED"
    assert p["account"] == "XX4151"


def test_parse_mpesa_screenshot_bank_notifications():
    ocr = {
        "full_text": (
            "Absa Bank Kenya PLC\nFriday, 2 Oct • 11:57\n"
            "Dear Kristein Gichuhi Mwaura, You have deposited KES18500.0 MPESA "
            "ref: UJ24P8JAPR to account XX4151 for NEPTUNE SHELTERS LIMITED "
            "through Absa Bank Kenya PLC\nSaturday, 24 Oct • 14:36\n"
            "EMMANUEL MAGUNGA ORINGE has transferred KES210 to your MPESA "
            "ref: UJO4PB8294. Please await MPESA notification.\n"
            "Saturday, 31 Oct • 15:23\n"
            "Dear Kristein Gichuhi Mwaura, You have deposited KES27750.0 MPESA "
            "ref: UJV4PO6ABO to account XX4151 for NEPTUNE SHELTERS LIMITED "
            "through Absa Bank Kenya PLC"
        ),
        "messages": [],
        "contact": {"name": "AbsaBank"},
    }
    out = parse_mpesa_screenshot(ocr, "e-mpesa-1")
    assert out["layout"] == "supported"
    assert [e["code"] for e in out["entries"]] == [
        "UJ24P8JAPR", "UJO4PB8294", "UJV4PO6ABO",
    ]
    assert [e["amount_kes"] for e in out["entries"]] == [18500.0, 210.0, 27750.0]
    assert out["entries"][0]["recipient"] == "NEPTUNE SHELTERS LIMITED"
    assert out["entries"][1]["code"] == "UJO4PB8294"


# ---------------------------------------------------------------------------
# Date checks (dates)
# ---------------------------------------------------------------------------

def _case_file(**overrides):
    base = {
        "case_context": {},
        "evidence": [],
        "user_provided_facts": [],
    }
    base.update(overrides)
    return base


def test_date_checks_no_dates():
    dc = compute_date_checks(_case_file(), now=_dt(2026, 11, 9, 12, 0))
    assert dc["no_dates"] is True
    assert dc["analysis_date"] == "2026-11-09"


def test_date_checks_conversation_date_and_due_date():
    case = _case_file(
        evidence=[
            {
                "evidence_id": "e-chat-1",
                "type": "chat_screenshot",
                "ocr": {
                    "contact": {"name": "Neptune Shelters"},
                    "messages": [
                        {"index": 1, "sender": "me", "text": "I will deliver by November 16th", "timestamp": "10:00"},
                    ],
                    "system_events": [{"text": "8 November 2026", "after_message_index": 0}],
                },
                "parsed": [],
            }
        ]
    )
    dc = compute_date_checks(case, now=_dt(2026, 11, 9, 12, 0))
    assert dc["no_dates"] is False
    assert any(c["date"] == "2026-11-08" for c in dc["conversation_dates"])
    assert dc["performance_due"]["status"] == "not_yet_due"
    assert dc["performance_due"]["date"] == "2026-11-16"


def test_date_checks_performance_overdue():
    case = _case_file(
        evidence=[
            {
                "evidence_id": "e-chat-1",
                "type": "chat_screenshot",
                "ocr": {
                    "contact": {"name": "Neptune Shelters"},
                    "messages": [
                        {"index": 1, "sender": "me", "text": "I will deliver by November 5th", "timestamp": "10:00"},
                    ],
                    "system_events": [{"text": "2 November 2026", "after_message_index": 0}],
                },
                "parsed": [],
            }
        ]
    )
    dc = compute_date_checks(case, now=_dt(2026, 11, 9, 12, 0))
    assert dc["performance_due"]["status"] == "overdue"


def _dt(y, m, d, hh, mm):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo("Africa/Nairobi"))


# ---------------------------------------------------------------------------
# Case-file analysis input + follow-ups
# ---------------------------------------------------------------------------

def test_build_analysis_input_from_case_masks_and_merges():
    case = {
        "case_context": {"user_role": "claimant", "user_account": "buyer"},
        "evidence": [
            {
                "evidence_id": "e-chat-1",
                "type": "chat_screenshot",
                "file_name": "shot.png",
                "ocr": {
                    "contact": {"name": "Neptune Shelters"},
                    "messages": [
                        {"index": 1, "sender": "Neptune Shelters", "text": "pay +254721445100", "timestamp": "10:00"},
                    ],
                    "system_events": [],
                },
            }
        ],
    }
    dc = compute_date_checks(case, now=_dt(2026, 11, 9, 12, 0))
    out = build_analysis_input_from_case(case, dc)
    assert out["case_context"]["user_role"] == "claimant"
    assert out["case_context"]["analysis_date"] == "2026-11-09"
    assert out["evidence"][0]["evidence_id"] == "e-chat-1"
    assert "2547***100" in out["messages"][0]["text"]
    assert out["messages"][0]["sender"] == "them"
    assert out["messages"][0]["evidence_id"] == "e-chat-1"


def test_deterministic_followups_missing_date_and_payment():
    case = {
        "case_context": {},
        "evidence": [
            {
                "evidence_id": "e-chat-1",
                "type": "chat_screenshot",
                "ocr": {
                    "contact": {"name": "Neptune Shelters"},
                    "messages": [
                        {"index": 1, "sender": "me", "text": "payment details till 578892", "timestamp": "10:00"},
                        {"index": 2, "sender": "me", "text": "done", "timestamp": "10:01"},
                    ],
                    "system_events": [],
                },
            }
        ],
    }
    dc = compute_date_checks(case, now=_dt(2026, 11, 9, 12, 0))
    ai = build_analysis_input_from_case(case, dc)
    reqs = build_deterministic_followups(ai, case)
    ids = [r.id for r in reqs]
    assert f"provide_date:E:e-chat-1" in ids
    assert "upload_mpesa_screenshot:K:payment" in ids


def test_merge_followups_dedupes_by_id():
    from SCCVision.analyst.models import FollowUpRequest

    case = {
        "case_context": {},
        "evidence": [
            {
                "evidence_id": "e-chat-1",
                "type": "chat_screenshot",
                "ocr": {"contact": {"name": "N"}, "messages": [], "system_events": []},
            }
        ],
    }
    dc = compute_date_checks(case, now=_dt(2026, 11, 9, 12, 0))
    ai = build_analysis_input_from_case(case, dc)
    result = _result(
        follow_up_requests=[
            {"id": "provide_date:E:e-chat-1", "type": "provide_date", "prompt": "dup", "blocking": True},
            {"id": "custom:question", "type": "confirm_choice", "prompt": "q", "blocking": False},
        ]
    )
    merged = merge_followups(result, ai, case)
    ids = [r.id for r in merged]
    assert ids.count("provide_date:E:e-chat-1") == 1
    assert "custom:question" in ids


# ---------------------------------------------------------------------------
# Case-file pipeline (stubbed LLM)
# ---------------------------------------------------------------------------

class _CaseFakeLLM:
    model = "test-model"

    def complete(self, analysis_input):
        return _result(
            elements={
                "C": {
                    "status": "PARTIAL",
                    "evidence": [{"msg_index": 1, "quote": "I charge 27750"}],
                    "reasoning": "price quoted",
                    "confidence": 0.8,
                }
            },
            overall_verdict="POSSIBLY VALID WITH GAPS",
        )


def test_analyze_case_returns_full_response():
    case = {
        "case_context": {"user_role": "claimant", "user_account": "buyer"},
        "evidence": [
            {
                "evidence_id": "e-chat-1",
                "type": "chat_screenshot",
                "ocr": {
                    "contact": {"name": "Neptune Shelters"},
                    "messages": [
                        {"index": 1, "sender": "Neptune Shelters", "text": "I charge 27750 for my services", "timestamp": "10:00"},
                    ],
                    "system_events": [],
                },
            }
        ],
    }
    out = analyze_case(case, llm=_CaseFakeLLM())
    assert out["result"]["overall_verdict"] == "POSSIBLY VALID WITH GAPS"
    assert out["analysis_input"]["date_checks"]["no_dates"] is True
    assert out["verification"]["verdict_after"] == "POSSIBLY VALID WITH GAPS"
    assert out["prompt_version"]
