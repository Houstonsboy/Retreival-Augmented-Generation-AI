from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from groq_budget import MemoryLedger


@pytest.fixture
def research(monkeypatch):
    import app as module
    import auth
    import groq_client

    monkeypatch.setenv("LEGAL_RETRIEVAL_SIGNING_KEY", "test-signing-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-provider-key")
    monkeypatch.setattr(auth, "_extract_bearer", lambda: ("token", None))
    monkeypatch.setattr(auth, "_verify_and_load", lambda _: (("researcher", {"type": "advocate"}), None))
    monkeypatch.setattr(module, "save_turn", lambda *a, **kw: {
        "conversation_id": "dev", "turn_id": "turn", "is_new_conversation": True,
    })
    monkeypatch.setattr(module.fb_firestore, "client", lambda: None)
    ledger = MemoryLedger()
    monkeypatch.setattr(module, "firestore_reserve", lambda db, key, stage, tokens, timeout: ledger.reserve(key, stage))
    monkeypatch.setattr(module, "firestore_finish", lambda *a: None)
    monkeypatch.setattr(groq_client, "_admissions", groq_client.deque())
    monkeypatch.setattr(groq_client, "_blocked_until", 0)
    return module, module.app.test_client(), groq_client


def test_exact_excluded_citation_resolves_without_provider(research, monkeypatch):
    module, client, transport = research
    post = Mock(side_effect=AssertionError("unexpected outbound request"))
    monkeypatch.setattr(transport.requests, "post", post)
    query = {"query": "2011 KEHC 1481 (KLR)", "client_turn_id": "lookup"}
    lookup = client.post("/api/qretrieve", json=query).get_json()
    assert lookup["classification"]["retrieval_mode"] == "DIRECT_CASE_LOOKUP"
    payload = {**query, "classification": lookup["classification"], "retrieval": lookup["retrieval"],
               "retrieval_signature": lookup["retrieval_signature"]}
    answer = client.post("/api/generate-legal-answer", json=payload).get_json()
    assert answer["success"]
    assert "outside the approved retrieval corpus" in answer["answer"]
    assert not post.called
    payload["retrieval"]["authority_status"] = "indexed_lookup"
    assert client.post("/api/generate-legal-answer", json=payload).status_code == 400


def test_failed_classifier_uses_local_retrieval_without_retry(research, monkeypatch):
    module, client, transport = research
    post = Mock(return_value=SimpleNamespace(status_code=429, headers={"Retry-After": "5"}))
    monkeypatch.setattr(transport.requests, "post", post)
    retrieve = Mock(return_value={"results": []})
    monkeypatch.setattr(module, "retrieve_relevant_cases", retrieve)
    monkeypatch.setattr(module, "retrieve_relevant_rules", lambda *a, **kw: {"results": []})
    response = client.post("/api/qretrieve", json={"query": "What can I do about my case?", "client_turn_id": "fallback"})
    assert response.status_code == 200
    assert response.get_json()["classification"]["deterministic_fallback"]
    assert retrieve.call_args.args[0]["vector_query"] == "What can I do about my case?"
    assert post.call_count == 1


def test_signed_evidence_survives_browser_numeric_round_trip(research, monkeypatch):
    module, client, _ = research
    monkeypatch.setattr(module, "retrieve_resolved", lambda query, resolution, **kwargs: {
        "resolution": resolution, "authority_status": "excluded", "results": [],
        "rules": [], "retrieval_stats": {"scores": [1.0, -0.0, 0.875]},
    })
    query = {"query": "2011 KEHC 1481 (KLR)", "client_turn_id": "browser-round-trip"}
    lookup = client.post("/api/qretrieve", json=query).get_json()
    evidence = lookup["retrieval"]
    evidence["retrieval_stats"]["scores"] = [1, 0, 0.875]
    payload = {**query, "classification": lookup["classification"], "retrieval": evidence,
               "retrieval_signature": lookup["retrieval_signature"]}
    assert client.post("/api/generate-legal-answer", json=payload).status_code == 200
    evidence["retrieval_stats"]["scores"][2] = 0.9
    response = client.post("/api/generate-legal-answer", json=payload)
    assert response.status_code == 400
    assert response.get_json()["error"] == "Retrieval evidence failed validation"
