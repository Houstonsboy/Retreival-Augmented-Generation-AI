"""Unit tests for graph/keys.py — all deterministic, no DB needed."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from graph.keys import case_key, provision_key, para_key, normalise_name, judge_key


class TestCaseKey:
    def test_kehc_citation(self):
        assert case_key("[2024] KEHC 15763 (KLR)") == "KEHC-2024-15763"

    def test_keca_citation(self):
        assert case_key("[2019] KECA 42 (KLR)") == "KECA-2019-42"

    def test_eklr_citation_with_name(self):
        key = case_key("[2011] eKLR", name="Barisa v Matuguda")
        assert key.startswith("eKLR-2011-")
        assert len(key) > len("eKLR-2011-")

    def test_case_key_stable(self):
        k1 = case_key("[2024] KEHC 15763 (KLR)")
        k2 = case_key("[2024] KEHC 15763 (KLR)")
        assert k1 == k2

    def test_different_citations_different_keys(self):
        assert case_key("[2024] KEHC 15763 (KLR)") != case_key("[2024] KEHC 999 (KLR)")


class TestProvisionKey:
    def test_evidence_act_s106b(self):
        assert provision_key("Evidence Act", "s.106B(4)(d)") == "evidence-act/s106B/4/d"

    def test_simple_section(self):
        assert provision_key("Contract Act", "s.3") == "contract-act/s3"

    def test_stable(self):
        k1 = provision_key("Evidence Act", "s.106B(4)(d)")
        k2 = provision_key("Evidence Act", "s.106B(4)(d)")
        assert k1 == k2


class TestParaKey:
    def test_para_key(self):
        assert para_key("KEHC-2024-15763", 44) == "KEHC-2024-15763#p44"

    def test_para_key_zero(self):
        assert para_key("KEHC-2024-15763", 0) == "KEHC-2024-15763#p0"


class TestNormaliseName:
    def test_versus_variants(self):
        a = normalise_name("Barisa v Matuguda")
        b = normalise_name("Barisa vs Matuguda")
        c = normalise_name("Barisa versus Matuguda")
        assert a == b == c

    def test_mataguda_variants(self):
        # Alias resolution: all spellings should normalise to the same form once
        # the alias table is consulted; here we just test the normaliser lowercases.
        variants = ["Matuguda", "Matriguda", "Mataguda"]
        normed = [normalise_name(v) for v in variants]
        # All should be lowercase with no punctuation
        for n in normed:
            assert n == n.lower()
            assert "." not in n

    def test_strips_punctuation(self):
        assert "." not in normalise_name("Republic v. Kamau")


class TestJudgeKey:
    def test_judge_key(self):
        assert judge_key("Hon. Justice Ngugi") == "judge-hon-justice-ngugi"
