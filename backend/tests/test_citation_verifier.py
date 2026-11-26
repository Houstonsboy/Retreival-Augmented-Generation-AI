"""
Tests for citation_verifier.py — extraction, logging, and false-positive analysis.

Run with:
    cd backend
    PYTHONPATH=. pytest tests/test_citation_verifier.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_BACKEND = os.path.dirname(os.path.dirname(__file__))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from citation_verifier import (
    CITATION_LOG,
    HALL_LOG,
    extract_citations,
    log_citation_check,
    log_hallucination,
    verify_and_clean,
    _SEC_RE,
)


# ---------------------------------------------------------------------------
# Section-regex false positive: source aliases [S1], [S2], [S1.1]
# ---------------------------------------------------------------------------

class TestSectionRegexFalsePositives:
    """
    The current _SEC_RE matches source aliases like [S1] as 'SECTION 1'.
    These tests document the false positive so the behaviour is visible.
    A PROPOSED narrower pattern is tested in TestProposedNarrowerRegex.
    """

    def test_alias_S1_currently_matched_as_section_1(self):
        """Current regex: [S1] is extracted as section '1' — this is a false positive."""
        matches = _SEC_RE.findall("[S1] says the court held")
        # Document the existing (buggy) behaviour: '1' is extracted.
        assert "1" in matches, (
            "If this assertion fails the false-positive was fixed in _SEC_RE — "
            "update TestProposedNarrowerRegex accordingly."
        )

    def test_alias_S2_currently_matched(self):
        matches = _SEC_RE.findall("[S2] deals with jurisdiction")
        assert "2" in matches

    def test_alias_S1_1_currently_matched(self):
        matches = _SEC_RE.findall("[S1.1] is the chunk alias")
        assert "1" in matches

    def test_true_positive_section_106B_still_matched(self):
        matches = _SEC_RE.findall("Section 106B of the Evidence Act")
        assert "106B" in matches

    def test_true_positive_s78A_still_matched(self):
        matches = _SEC_RE.findall("s.78A of the Act")
        assert "78A" in matches


class TestProposedNarrowerRegex:
    """
    Proposed fix: add (?<!\\[) negative lookbehind to _SEC_RE so that
    [S1]-style source aliases are not matched.

    DO NOT merge this change without user approval (Task 4 instruction).
    """

    def setup_method(self):
        import re
        self._narrow = re.compile(
            r'(?<!\[)\b(?:section|sec|s)\.?\s*(\d+[A-Z]?(?:\([^)]+\))*)',
            re.IGNORECASE,
        )

    def test_alias_S1_no_longer_matched(self):
        matches = self._narrow.findall("[S1] says the court held")
        assert "1" not in matches

    def test_alias_S2_no_longer_matched(self):
        matches = self._narrow.findall("[S2] deals with jurisdiction")
        assert "2" not in matches

    def test_alias_S1_1_no_longer_matched(self):
        matches = self._narrow.findall("[S1.1] is the chunk alias")
        assert "1" not in matches

    def test_true_section_106B_still_matched(self):
        matches = self._narrow.findall("Section 106B of the Evidence Act")
        assert "106B" in matches

    def test_true_s78A_still_matched(self):
        matches = self._narrow.findall("s.78A of the Act")
        assert "78A" in matches

    def test_true_under_section_4(self):
        matches = self._narrow.findall("Under section 4 the limitation period applies")
        assert "4" in matches

    def test_bare_s_dot_number_still_matched(self):
        matches = self._narrow.findall("per s.5 of the Small Claims Courts Act")
        assert "5" in matches


# ---------------------------------------------------------------------------
# log_citation_check — total-count logging (Task 4)
# ---------------------------------------------------------------------------

class TestLogCitationCheck:
    def test_writes_jsonl_entry(self, tmp_path):
        log_file = tmp_path / "citation_checks.jsonl"
        with patch("citation_verifier.CITATION_LOG", log_file):
            with patch("citation_verifier.LOGS_DIR", tmp_path):
                log_citation_check("test query", total_checked=5, total_unsupported=2)

        lines = log_file.read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["query"] == "test query"
        assert entry["total_checked"] == 5
        assert entry["total_unsupported"] == 2
        assert "timestamp" in entry

    def test_zero_unsupported(self, tmp_path):
        log_file = tmp_path / "citation_checks.jsonl"
        with patch("citation_verifier.CITATION_LOG", log_file):
            with patch("citation_verifier.LOGS_DIR", tmp_path):
                log_citation_check("clean query", total_checked=3, total_unsupported=0)

        entry = json.loads(log_file.read_text().strip())
        assert entry["total_unsupported"] == 0

    def test_appends_multiple_entries(self, tmp_path):
        log_file = tmp_path / "citation_checks.jsonl"
        with patch("citation_verifier.CITATION_LOG", log_file):
            with patch("citation_verifier.LOGS_DIR", tmp_path):
                log_citation_check("q1", 4, 1)
                log_citation_check("q2", 6, 0)

        lines = log_file.read_text().splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# verify_and_clean — log_citation_check is called on every invocation
# ---------------------------------------------------------------------------

class TestVerifyAndCleanCallsCitationLog:
    def _sources_with_section(self, section_num: str) -> list:
        return [{"article_number": section_num, "chunks": [], "document_label": "Test Act"}]

    def test_citation_log_called_when_no_unsupported(self, tmp_path):
        """Even when all citations are valid, log_citation_check must be called."""
        log_file = tmp_path / "citation_checks.jsonl"
        with patch("citation_verifier.CITATION_LOG", log_file):
            with patch("citation_verifier.LOGS_DIR", tmp_path):
                verify_and_clean(
                    "Per section 106B of the Evidence Act",
                    self._sources_with_section("106B"),
                    "test query",
                    regen_fn=None,
                )
        assert log_file.exists(), "citation_checks.jsonl must be written even with zero unsupported"
        entry = json.loads(log_file.read_text().strip())
        assert entry["total_checked"] >= 1
        assert entry["total_unsupported"] == 0

    def test_citation_log_records_unsupported_count(self, tmp_path):
        log_file = tmp_path / "citation_checks.jsonl"
        hall_file = tmp_path / "hallucinations.jsonl"
        with patch("citation_verifier.CITATION_LOG", log_file):
            with patch("citation_verifier.HALL_LOG", hall_file):
                with patch("citation_verifier.LOGS_DIR", tmp_path):
                    verify_and_clean(
                        "Under section 999 the sky is blue",
                        [],  # no sources → section 999 is unsupported
                        "test query",
                        regen_fn=None,
                    )
        entry = json.loads(log_file.read_text().strip())
        assert entry["total_unsupported"] >= 1
