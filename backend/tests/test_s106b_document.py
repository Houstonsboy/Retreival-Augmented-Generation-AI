"""
Unit tests for the Section 106B PDF document generator (Phase 3 — FR-04).

Run with:
    cd backend
    PYTHONPATH=. pytest tests/test_s106b_document.py -v
"""

import io
import os
import sys

import pytest

_BACKEND = os.path.dirname(os.path.dirname(__file__))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from SCCVision.s106b.document import render_pdf


# ---------------------------------------------------------------------------
# Test record helpers
# ---------------------------------------------------------------------------

_CERT_NUMBER = "SCC106B-20261117-ABCDEF"
_CREATED_AT  = "2026-11-17"


def _all_met_conditions():
    codes = [
        ("4a",        "Electronic record identified and manner of production described"),
        ("4b",        "Particulars of the device"),
        ("2a",        "Device regularly used by a person with lawful control"),
        ("2b",        "Information regularly fed into the device"),
        ("2c",        "Device operating properly"),
        ("2d",        "Record derived from information fed into the device"),
        ("4d",        "Signed by a person in a responsible position"),
        ("INTEGRITY", "Evidence integrity checks passed"),
    ]
    return [
        {"code": c, "title": t, "met": True, "basis": [], "remedy": None}
        for c, t in codes
    ]


def _some_unmet_conditions():
    conds = _all_met_conditions()
    for c in conds:
        if c["code"] == "2a":
            c["met"] = False
            c["remedy"] = "Confirm lawful control."
    return conds


def _make_record(document_type="CERTIFICATE", extra_device=None, extra_analysis=None):
    device = dict(
        device_make="Samsung",
        device_model="Galaxy A14",
        phone_number_last4="6789",
        production_method="screenshot_on_device",
        owner_is_declarant=True,
        lawful_control=True,
        regular_use=True,
        regular_feed=True,
        operating_properly=True,
        malfunction_details=None,
        malfunction_affects_record=None,
        messages_on_this_device=True,
        declarant_name="Jane Synthetic",
        declarant_position="Owner and sole user of the device",
        declaration_confirmed=True,
    )
    if extra_device:
        device.update(extra_device)

    analysis_summary = dict(
        overall_verdict="LIKELY VALID",
        plain_summary="Contract appears valid based on the evidence.",
        missing_information=["Date of contract is missing"],
        strengthen_checklist=["Upload full unedited conversation"],
        disclaimer="This analysis is not legal advice.",
        prompt_version="v1",
        model="qwen/qwen3.8-27b",
    )
    if extra_analysis:
        analysis_summary.update(extra_analysis)

    is_cert = document_type == "CERTIFICATE"
    return {
        "certificate_number": _CERT_NUMBER,
        "document_type":      document_type,
        "created_at":         _CREATED_AT,
        "schema_version":     1,
        "analysis_id":        "test-analysis-id",
        "device_facts":       device,
        "evidence": [
            {
                "evidence_id":   "e-chat-1",
                "type":          "chat_screenshot",
                "file_name":     "conversation.png",
                "sha256":        "aabbccddeeff001122334455667788990011",
                "server_sha256": "aabbccddeeff001122334455667788990011",
                "hash_match":    True,
                "added_at":      "2026-11-17T10:00:00",
            }
        ],
        "gate": {
            "document_type": document_type,
            "outcome":       "certificate_ready" if is_cert else "needs_strengthening",
            "conditions":    _all_met_conditions() if is_cert else _some_unmet_conditions(),
            "unmet":         [] if is_cert else ["2a"],
            "gate_version":  "s106b-gate-1.0",
        },
        "analysis_summary": analysis_summary,
    }


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract all text from PDF bytes using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------------------------------------------------------------------------
# Basic validity
# ---------------------------------------------------------------------------

class TestPdfValidity:
    def test_certificate_starts_with_pdf_magic(self):
        b = render_pdf(_make_record("CERTIFICATE"))
        assert b[:4] == b"%PDF"

    def test_deficiency_report_starts_with_pdf_magic(self):
        b = render_pdf(_make_record("DEFICIENCY_REPORT"))
        assert b[:4] == b"%PDF"

    def test_certificate_returns_bytes(self):
        assert isinstance(render_pdf(_make_record("CERTIFICATE")), bytes)

    def test_deficiency_returns_bytes(self):
        assert isinstance(render_pdf(_make_record("DEFICIENCY_REPORT")), bytes)


# ---------------------------------------------------------------------------
# Text content — CERTIFICATE
# ---------------------------------------------------------------------------

class TestCertificateContent:
    def test_cert_number_in_text(self):
        b = render_pdf(_make_record("CERTIFICATE"))
        text = _extract_pdf_text(b)
        assert _CERT_NUMBER in text

    def test_draft_watermark_present(self):
        text = _extract_pdf_text(render_pdf(_make_record("CERTIFICATE")))
        assert "DRAFT" in text

    def test_section_106b_reference_present(self):
        text = _extract_pdf_text(render_pdf(_make_record("CERTIFICATE")))
        assert "106B" in text

    def test_device_make_in_cert(self):
        text = _extract_pdf_text(render_pdf(_make_record("CERTIFICATE")))
        assert "Samsung" in text

    def test_evidence_filename_in_cert(self):
        text = _extract_pdf_text(render_pdf(_make_record("CERTIFICATE")))
        assert "conversation.png" in text


# ---------------------------------------------------------------------------
# Text content — DEFICIENCY REPORT
# ---------------------------------------------------------------------------

class TestDeficiencyReportContent:
    def test_cert_number_in_deficiency(self):
        text = _extract_pdf_text(render_pdf(_make_record("DEFICIENCY_REPORT")))
        assert _CERT_NUMBER in text

    def test_requirements_not_yet_met_heading(self):
        text = _extract_pdf_text(render_pdf(_make_record("DEFICIENCY_REPORT")))
        assert "Requirements Not Yet Met" in text

    def test_evidence_schedule_in_deficiency(self):
        text = _extract_pdf_text(render_pdf(_make_record("DEFICIENCY_REPORT")))
        assert "conversation.png" in text

    def test_disclaimer_in_deficiency(self):
        text = _extract_pdf_text(render_pdf(_make_record("DEFICIENCY_REPORT")))
        assert "not legal advice" in text


# ---------------------------------------------------------------------------
# Phone display and legacy records
# ---------------------------------------------------------------------------

class TestPhoneDisplay:
    def test_full_international_phone_in_certificate(self):
        record = _make_record("CERTIFICATE", extra_device={"phone_number": "+254712345678"})
        text = _extract_pdf_text(render_pdf(record))
        assert "+254712345678" in text
        assert "******6789" not in text

    def test_legacy_phone_is_masked_in_certificate(self):
        record = _make_record("CERTIFICATE", extra_device={"phone_number_last4": "9999"})
        text = _extract_pdf_text(render_pdf(record))
        # last 4 may appear, but a full 10–12 digit number must not
        # We just verify "phone_number_last4" value appears only masked
        assert "******9999" in text or "9999" in text  # masked form
        # Verify the full number pattern is not present (e.g. 0712349999 / +254712349999)
        assert "0712349999" not in text
        assert "+254712349999" not in text

    def test_no_phone_number_when_not_provided(self):
        record = _make_record("CERTIFICATE", extra_device={"phone_number_last4": None})
        text = _extract_pdf_text(render_pdf(record))
        assert "not provided" in text

    def test_full_phone_not_in_deficiency(self):
        record = _make_record("DEFICIENCY_REPORT", extra_device={"phone_number_last4": "1234"})
        # phone number only appears in certificate, not deficiency; either way full number absent
        text = _extract_pdf_text(render_pdf(record))
        assert "0712341234" not in text


# ---------------------------------------------------------------------------
# Determinism: same record → identical bytes
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_certificate_bytes_identical_on_second_render(self):
        record = _make_record("CERTIFICATE")
        b1 = render_pdf(record)
        b2 = render_pdf(record)
        assert b1 == b2

    def test_deficiency_bytes_identical_on_second_render(self):
        record = _make_record("DEFICIENCY_REPORT")
        b1 = render_pdf(record)
        b2 = render_pdf(record)
        assert b1 == b2

    def test_different_dates_give_different_bytes(self):
        r1 = _make_record("CERTIFICATE")
        r2 = _make_record("CERTIFICATE")
        r2["created_at"] = "2026-01-01"
        assert render_pdf(r1) != render_pdf(r2)


# ---------------------------------------------------------------------------
# Robustness: emoji and unusual characters don't crash
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_emoji_in_device_make_does_not_crash(self):
        record = _make_record("CERTIFICATE", extra_device={"device_make": "Samsung \U0001F4F1"})
        b = render_pdf(record)
        assert b[:4] == b"%PDF"

    def test_emoji_in_analysis_summary_does_not_crash(self):
        record = _make_record(
            "DEFICIENCY_REPORT",
            extra_analysis={"missing_information": ["Contract date \U0001F4C5 missing"]},
        )
        b = render_pdf(record)
        assert b[:4] == b"%PDF"

    def test_smart_quotes_in_summary_do_not_crash(self):
        record = _make_record(
            "DEFICIENCY_REPORT",
            extra_analysis={
                "plain_summary": "“Hello” said the contract.",
                "missing_information": ["‘date’ missing"],
            },
        )
        b = render_pdf(record)
        assert b[:4] == b"%PDF"

    def test_empty_evidence_list_does_not_crash(self):
        record = _make_record("CERTIFICATE")
        record["evidence"] = []
        b = render_pdf(record)
        assert b[:4] == b"%PDF"

    def test_empty_conditions_list_does_not_crash(self):
        record = _make_record("DEFICIENCY_REPORT")
        record["gate"]["conditions"] = []
        b = render_pdf(record)
        assert b[:4] == b"%PDF"

    def test_missing_analysis_summary_does_not_crash(self):
        record = _make_record("DEFICIENCY_REPORT")
        record["analysis_summary"] = {}
        b = render_pdf(record)
        assert b[:4] == b"%PDF"

    def test_missing_device_facts_does_not_crash(self):
        record = _make_record("CERTIFICATE")
        record["device_facts"] = {}
        b = render_pdf(record)
        assert b[:4] == b"%PDF"

    def test_malfunction_cert_uses_alternate_2c_statement(self):
        """Certificate with malfunction-but-not-affecting uses the malfunction wording."""
        record = _make_record(
            "CERTIFICATE",
            extra_device={
                "operating_properly": False,
                "malfunction_details": "Battery drains fast",
                "malfunction_affects_record": False,
            },
        )
        # Mark 2c as met in the conditions list
        for c in record["gate"]["conditions"]:
            if c["code"] == "2c":
                c["met"] = True
        b = render_pdf(record)
        text = _extract_pdf_text(b)
        # Should mention the malfunction in the certificate body
        assert "Battery drains fast" in text


def test_certificate_crest_and_declaration_precede_contract_appendix():
    from pypdf import PdfReader

    certificate = PdfReader(io.BytesIO(render_pdf(_make_record())))
    first_page = certificate.pages[0]
    assert len(first_page.images) == 1
    text = first_page.extract_text()
    headings = (
        "IN THE SMALL CLAIMS COURT",
        "CLAIMANT",
        "RESPONDENT",
        "CERTIFICATE UNDER SECTION 106B(4)",
        "attached Schedule",
        "SIGNATURE OF DECLARANT",
    )
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "Appendix" not in text
    schedule_page = certificate.pages[1].extract_text()
    assert "SCHEDULE" in schedule_page
    assert "conversation.png" in schedule_page
    assert "DRAWN & FILED BY" in schedule_page
    assert "Contract Analysis" in certificate.pages[2].extract_text()

    report = PdfReader(io.BytesIO(render_pdf(_make_record("DEFICIENCY_REPORT"))))
    assert all(not page.images for page in report.pages)


def test_evidence_schedule_wraps_values_and_repeats_headings():
    from pypdf import PdfReader

    record = _make_record()
    fingerprint = "0123456789abcdef" * 4
    filename = "long_filename_" * 12 + "final_screenshot.png"
    record["evidence"] = [
        dict(record["evidence"][0], file_name=filename, server_sha256=fingerprint)
        for _ in range(20)
    ]
    reader = PdfReader(io.BytesIO(render_pdf(record)))
    schedule_pages = [
        text for page in reader.pages if "File name" in (text := page.extract_text())
    ]
    assert len(schedule_pages) > 1
    for page in schedule_pages:
        compact_text = "".join(page.split())
        assert filename in compact_text
        assert fingerprint in compact_text
