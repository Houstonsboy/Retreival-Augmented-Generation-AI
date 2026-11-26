import json
import pytest

from direct_answer import _subsection, assemble, checked_claims, respond
from direct_retrieval import resolve


@pytest.mark.parametrize("query", [
    "2011KEHC1481(KLR)", "2011 KEHC 1481 (KLR)", "[2011] KEHC 1481 (KLR)",
    "Republic v Barisa Wayu Matriguda [2011] eKLR", "Republic vs Barisa Wayu Matuguda",
])
def test_case_aliases_and_exclusion(query):
    resolved = resolve(query)
    assert resolved["case"]["case_key"] == "KEHC-2011-1481"
    assert resolved["case"]["source_available"]
    assert not resolved["case"]["eligible"]


def test_case_citation_is_not_a_statutory_reference():
    assert resolve("What was the ruling in Ogembo v Yongo, 2024 KEHC 15763?")["mode"] == "CASE_OUTCOME_LOOKUP"
    assert resolve("Compare 2024 KEHC 15763 and 2024 KEHC 17038")["ambiguous"]
    assert not resolve("[2011] eKLR")
    assert resolve("Ogembo v Yongo [2011] eKLR")["reason"] == "conflicting_citation_year"


def test_subsection_is_resolved_and_retained():
    resolution = resolve("What does section 106B(4) of the Evidence Act say?")
    assert resolution["provision"]["provision_key"] == "evidence-act-cap-80/s106B/4"
    retrieved = {"resolution": resolution, "rules": [{
        "source_document": "Evidence Act", "article_number": "106B", "chunks": [{
            "chunk_id": "s106b", "text": "(3) Earlier. (4) A certificate identifies the record and must be signed. (5) Next.",
        }],
    }]}
    [passage] = assemble(retrieved)
    assert passage.text == "(4) A certificate identifies the record and must be signed."
    assert passage.primary


def test_verification_failure_does_not_render_original_contradiction():
    retrieved = {"resolution": {"mode": "CASE_OUTCOME_LOOKUP", "case": {"case_identifier": "case"}}, "results": [
        {"parent_document": "case", "chunks": [{"chunk_id": "a", "text": "The appeal is dismissed with costs."}]},
    ]}
    passages = assemble(retrieved)
    calls = []
    def provider(messages, stage, **kwargs):
        calls.append(stage)
        if stage == "verification":
            raise TimeoutError
        return json.dumps({"claims": [{"answer": "The appeal succeeded.", "quote": passages[0].text, "alias": "C1.1"}]})
    answer, claims, status = respond("What was the ruling?", retrieved, passages, provider)
    assert calls == ["generation", "verification"]
    assert status == "extractive_fallback"
    assert not claims
    assert "appeal succeeded" not in answer
    assert "dismissed" in answer
    assert not checked_claims({"claims": [{"answer": "Allowed", "quote": "The appeal was allowed with costs.", "alias": "C1.1"}]}, passages)


def test_corrected_verified_output_is_rendered():
    retrieved = {"resolution": {"mode": "CASE_OUTCOME_LOOKUP", "case": {"case_identifier": "case"}}, "results": [
        {"parent_document": "case", "chunks": [{"chunk_id": "a", "text": "The appeal is dismissed with costs."}]},
    ]}
    passages = assemble(retrieved)
    def provider(messages, stage, **kwargs):
        answer = "The appeal succeeded." if stage == "generation" else "The court dismissed the appeal."
        return json.dumps({"claims": [{"answer": answer, "quote": passages[0].text, "alias": "C1.1"}]})
    answer, claims, status = respond("What was the ruling?", retrieved, passages, provider)
    assert answer == "The court dismissed the appeal. [C1.1]"
    assert claims and status == "semantic_verified"


def test_direct_identity_works_without_provider():
    retrieved = {"resolution": resolve("2011 KEHC 1481 (KLR)"), "authority_status": "excluded"}
    def fail(*args, **kwargs):
        pytest.fail("Provider must not be invoked")
    answer, _, status = respond("2011 KEHC 1481 (KLR)", retrieved, [], fail)
    assert "outside the approved" in answer
    assert status == "identity_lookup"


def test_nested_subsection_without_html_spacing():
    text = "(4)A certificate includes—(a)an identifier; and(d)a signature. (5)Next."
    assert _subsection(text, ["4", "d"]) == "(d)a signature."
