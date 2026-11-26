"""
Unit tests for the Section 106B gate (Phase 2 — FR-04).

Run with:
    cd backend
    PYTHONPATH=. pytest tests/test_s106b_gate.py -v
"""

import os
import sys

import pytest
from pydantic import ValidationError

_BACKEND = os.path.dirname(os.path.dirname(__file__))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from SCCVision.s106b.models import DeviceFacts, ProductionMethod
from SCCVision.s106b.gate import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_facts(**overrides) -> DeviceFacts:
    """Return a DeviceFacts that satisfies every condition."""
    base = dict(
        device_make="Samsung",
        device_model="Galaxy A14",
        phone_number_last4="6789",
        production_method=ProductionMethod.SCREENSHOT_ON_DEVICE,
        owner_is_declarant=True,
        lawful_control=True,
        regular_use=True,
        regular_feed=True,
        operating_properly=True,
        malfunction_details=None,
        malfunction_affects_record=None,
        messages_on_this_device=True,
        declarant_name="Jane Doe",
        declarant_position="Owner and sole user of the device",
        declaration_confirmed=True,
    )
    base.update(overrides)
    return DeviceFacts(**base)


def _good_snapshot(**overrides) -> dict:
    """Return a minimal analysis snapshot that satisfies every gate check."""
    base = {
        "result": {
            "overall_verdict": "LIKELY VALID",
            "confidence": 0.9,
            "conversation_integrity": {"rating": "ADEQUATE", "issues": []},
            "mpesa_payments": [],
        },
        "verification": {
            "quotes_grounded": True,
            "ungrounded_quotes": [],
            "mpesa_cross_check": {
                "consistent": True,
                "invented_fields": False,
                "codes_llm_only": [],
            },
        },
        "evidence": [
            {
                "evidence_id":   "e-chat-1",
                "type":          "chat_screenshot",
                "file_name":     "chat.png",
                "sha256":        "aabbcc",
                "server_sha256": "aabbcc",
                "hash_match":    True,
                "added_at":      "2026-11-17T10:00:00+03:00",
            }
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestAllConditionsMet:
    def test_returns_certificate(self):
        result = evaluate(_good_snapshot(), _good_facts())
        assert result.document_type == "CERTIFICATE"
        assert result.outcome == "certificate_ready"
        assert result.unmet == []

    def test_all_eight_conditions_present(self):
        result = evaluate(_good_snapshot(), _good_facts())
        codes = [c.code for c in result.conditions]
        for code in ("4a", "4b", "2a", "2b", "2c", "2d", "4d", "INTEGRITY"):
            assert code in codes

    def test_contract_verdict_does_not_decide_admissibility(self):
        """LIKELY NOT VALID contract verdict + all s.106B conditions met → still CERTIFICATE."""
        snapshot = _good_snapshot()
        snapshot["result"]["overall_verdict"] = "LIKELY NOT VALID"
        result = evaluate(snapshot, _good_facts())
        assert result.document_type == "CERTIFICATE"
        assert result.outcome == "certificate_ready"


# ---------------------------------------------------------------------------
# Individual condition failures
# ---------------------------------------------------------------------------

class TestCondition4aFailures:
    def test_no_evidence_items_fails_4a(self):
        snapshot = _good_snapshot(evidence=[])
        result = evaluate(snapshot, _good_facts())
        assert "4a" in result.unmet

    def test_missing_server_sha256_fails_4a(self):
        ev = _good_snapshot()["evidence"][0].copy()
        ev["server_sha256"] = None
        snapshot = _good_snapshot(evidence=[ev])
        result = evaluate(snapshot, _good_facts())
        assert "4a" in result.unmet

    def test_hash_mismatch_fails_4a_and_outcome_weak(self):
        ev = _good_snapshot()["evidence"][0].copy()
        ev["server_sha256"] = "ddeeff"
        ev["sha256"] = "aabbcc"
        ev["hash_match"] = False
        snapshot = _good_snapshot(evidence=[ev])
        result = evaluate(snapshot, _good_facts())
        assert "4a" in result.unmet
        assert result.outcome == "weak"

    def test_production_method_other_fails_4a(self):
        result = evaluate(
            _good_snapshot(),
            _good_facts(production_method=ProductionMethod.OTHER),
        )
        assert "4a" in result.unmet
        assert result.outcome == "weak"  # 4a unmet → weak


class TestCondition4bAlwaysMet:
    def test_4b_always_met_when_make_model_given(self):
        result = evaluate(_good_snapshot(), _good_facts())
        c4b = next(c for c in result.conditions if c.code == "4b")
        assert c4b.met is True


class TestCondition4bFailure:
    """
    Condition 4b (device make/model) CAN be unmet in the gate's internal logic.
    However, in normal API flow, blank make/model is rejected by Pydantic with
    HTTP 422 before the gate is ever called — so 4b can never be unmet through
    the production endpoint.  These tests verify the gate branch directly.
    """

    def test_4b_unmet_when_make_blank(self):
        """Gate 4b logic returns met=False when device_make is blank (gate-layer test)."""
        from types import SimpleNamespace
        from SCCVision.s106b.gate import _check_4b
        blank = SimpleNamespace(device_make="", device_model="Galaxy A14", phone_number_last4=None)
        c4b = _check_4b(blank)
        assert c4b.met is False
        assert c4b.remedy is not None

    def test_4b_unmet_when_model_blank(self):
        """Gate 4b logic returns met=False when device_model is blank (gate-layer test)."""
        from types import SimpleNamespace
        from SCCVision.s106b.gate import _check_4b
        blank = SimpleNamespace(device_make="Samsung", device_model="", phone_number_last4=None)
        c4b = _check_4b(blank)
        assert c4b.met is False
        assert c4b.remedy is not None

    def test_4b_unmet_when_both_blank(self):
        """Gate 4b logic returns met=False when both make and model are blank."""
        from types import SimpleNamespace
        from SCCVision.s106b.gate import _check_4b
        blank = SimpleNamespace(device_make="", device_model="", phone_number_last4=None)
        c4b = _check_4b(blank)
        assert c4b.met is False


class TestCondition2aFailure:
    def test_no_lawful_control_fails_2a(self):
        result = evaluate(_good_snapshot(), _good_facts(lawful_control=False))
        assert "2a" in result.unmet
        assert result.outcome == "needs_strengthening"

    def test_no_regular_use_fails_2a(self):
        result = evaluate(_good_snapshot(), _good_facts(regular_use=False))
        assert "2a" in result.unmet

    def test_remedy_present(self):
        result = evaluate(_good_snapshot(), _good_facts(lawful_control=False))
        c = next(c for c in result.conditions if c.code == "2a")
        assert c.remedy is not None


class TestCondition2bFailure:
    def test_regular_feed_false_fails_2b(self):
        result = evaluate(_good_snapshot(), _good_facts(regular_feed=False))
        assert "2b" in result.unmet
        assert result.outcome == "needs_strengthening"


class TestCondition2cFailures:
    def test_malfunction_affecting_record_fails_2c(self):
        result = evaluate(
            _good_snapshot(),
            _good_facts(
                operating_properly=False,
                malfunction_details="Screen cracked, some texts missing",
                malfunction_affects_record=True,
            ),
        )
        assert "2c" in result.unmet
        assert result.outcome == "needs_strengthening"

    def test_malfunction_not_affecting_record_passes_2c(self):
        result = evaluate(
            _good_snapshot(),
            _good_facts(
                operating_properly=False,
                malfunction_details="Battery drains fast",
                malfunction_affects_record=False,
            ),
        )
        assert "2c" not in result.unmet


class TestCondition2dFailures:
    def test_forwarded_copy_fails_2d(self):
        result = evaluate(
            _good_snapshot(),
            _good_facts(
                production_method=ProductionMethod.FORWARDED_OR_COPIED,
                messages_on_this_device=False,
            ),
        )
        assert "2d" in result.unmet
        assert result.outcome in ("needs_strengthening", "weak")

    def test_messages_not_on_device_fails_2d(self):
        result = evaluate(
            _good_snapshot(),
            _good_facts(messages_on_this_device=False),
        )
        assert "2d" in result.unmet

    def test_mpesa_no_codes_fails_2d(self):
        snapshot = _good_snapshot()
        snapshot["evidence"][0]["type"] = "mpesa_screenshot"
        snapshot["result"]["mpesa_payments"] = []  # no codes
        result = evaluate(snapshot, _good_facts())
        assert "2d" in result.unmet


class TestCondition4dFailures:
    def test_not_owner_fails_4d(self):
        result = evaluate(_good_snapshot(), _good_facts(owner_is_declarant=False))
        assert "4d" in result.unmet
        assert result.outcome == "needs_strengthening"

    def test_declaration_not_confirmed_fails_4d(self):
        result = evaluate(_good_snapshot(), _good_facts(declaration_confirmed=False))
        assert "4d" in result.unmet


class TestIntegrityFailures:
    def test_ungrounded_quotes_fails_integrity(self):
        verification = {
            "quotes_grounded": False,
            "ungrounded_quotes": [{"element": "C", "reason": "quote_not_found"}],
            "mpesa_cross_check": {"consistent": True, "invented_fields": False},
        }
        snapshot = _good_snapshot(verification=verification)
        result = evaluate(snapshot, _good_facts())
        assert "INTEGRITY" in result.unmet
        assert result.outcome == "weak"

    def test_mpesa_inconsistent_fails_integrity(self):
        verification = {
            "quotes_grounded": True,
            "ungrounded_quotes": [],
            "mpesa_cross_check": {
                "consistent": False,
                "invented_fields": True,
                "codes_llm_only": ["FAKE123456"],
            },
        }
        snapshot = _good_snapshot(verification=verification)
        result = evaluate(snapshot, _good_facts())
        assert "INTEGRITY" in result.unmet
        assert result.outcome == "weak"

    def test_weak_conversation_integrity_fails(self):
        result_data = _good_snapshot()["result"].copy()
        result_data["conversation_integrity"] = {"rating": "WEAK", "issues": ["gaps"]}
        snapshot = _good_snapshot()
        snapshot["result"] = result_data
        result = evaluate(snapshot, _good_facts())
        assert "INTEGRITY" in result.unmet
        assert result.outcome == "weak"


# ---------------------------------------------------------------------------
# Outcome mapping
# ---------------------------------------------------------------------------

class TestOutcomeMapping:
    def test_only_questionnaire_unmet_gives_needs_strengthening(self):
        # 2a unmet; 4a and INTEGRITY met
        result = evaluate(
            _good_snapshot(),
            _good_facts(lawful_control=False),
        )
        assert result.outcome == "needs_strengthening"
        assert result.document_type == "DEFICIENCY_REPORT"

    def test_integrity_unmet_gives_weak(self):
        v = {
            "quotes_grounded": False,
            "ungrounded_quotes": [{"reason": "x"}],
            "mpesa_cross_check": {"consistent": True, "invented_fields": False},
        }
        result = evaluate(_good_snapshot(verification=v), _good_facts())
        assert result.outcome == "weak"

    def test_4a_unmet_gives_weak(self):
        result = evaluate(_good_snapshot(evidence=[]), _good_facts())
        assert result.outcome == "weak"


# ---------------------------------------------------------------------------
# DeviceFacts validation
# ---------------------------------------------------------------------------

class TestDeviceFactsValidation:
    def test_empty_device_make_raises(self):
        with pytest.raises(Exception):
            _good_facts(device_make="")

    def test_whitespace_device_make_raises(self):
        with pytest.raises(Exception):
            _good_facts(device_make="   ")

    def test_full_phone_requires_international_format(self):
        assert _good_facts(phone_number="+254712345678").phone_number == "+254712345678"
        with pytest.raises(ValidationError):
            _good_facts(phone_number="0712345678")

    def test_bad_last4_raises(self):
        with pytest.raises(Exception):
            _good_facts(phone_number_last4="123")  # 3 digits, not 4

    def test_bad_last4_non_digits_raises(self):
        with pytest.raises(Exception):
            _good_facts(phone_number_last4="123X")

    def test_none_last4_ok(self):
        facts = _good_facts(phone_number_last4=None)
        assert facts.phone_number_last4 is None

    def test_malfunction_without_details_raises(self):
        with pytest.raises(Exception):
            _good_facts(
                operating_properly=False,
                malfunction_details=None,
                malfunction_affects_record=True,
            )

    def test_malfunction_without_affects_raises(self):
        with pytest.raises(Exception):
            _good_facts(
                operating_properly=False,
                malfunction_details="Battery issue",
                malfunction_affects_record=None,
            )

    def test_empty_declarant_name_raises(self):
        with pytest.raises(Exception):
            _good_facts(declarant_name="")
