from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Event
from types import SimpleNamespace

import pytest

import groq_client
from groq_budget import BudgetUnavailable, MemoryLedger, research_budget


@pytest.fixture
def calls(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(groq_client, "_blocked_until", 0)
    monkeypatch.setattr(groq_client, "_admissions", groq_client.deque())
    monkeypatch.setattr(groq_client, "_concurrency", BoundedSemaphore(2))
    attempts = []
    def post(*args, **kwargs):
        attempts.append(kwargs)
        return SimpleNamespace(status_code=200, json=lambda: {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        })
    monkeypatch.setattr(groq_client.requests, "post", post)
    return attempts


def test_three_attempts_shared_across_requests(calls):
    ledger = MemoryLedger()
    for stage in ("classification", "generation", "verification"):
        with research_budget(lambda name, *_: ledger.reserve("turn", name)) as budget:
            assert groq_client.call_groq([], stage=stage) == "ok"
            assert budget.outcomes[stage]["usage"]["prompt_tokens"] == 4
    with research_budget(lambda name, *_: ledger.reserve("turn", name)):
        with pytest.raises(BudgetUnavailable):
            groq_client.call_groq([], stage="verification")
        with pytest.raises(BudgetUnavailable):
            groq_client.call_groq([])
    assert len(calls) == 3
    assert all(call["allow_redirects"] is False for call in calls)


def test_429_consumes_stage_and_blocks_followup(calls, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY_2", "second-test-key")
    def limited(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=429, headers={"Retry-After": "10"})
    monkeypatch.setattr(groq_client.requests, "post", limited)
    ledger = MemoryLedger()
    with research_budget(lambda name, *_: ledger.reserve("turn", name)) as budget:
        with pytest.raises(groq_client.GroqRateLimitError):
            groq_client.call_groq([], stage="generation")
        with pytest.raises(groq_client.GroqRateLimitError):
            groq_client.call_groq([], stage="verification")
        assert budget.outcomes["generation"]["status"] == 429
    assert len(calls) == 1


def test_concurrent_same_stage_and_capacity(calls, monkeypatch):
    entered = Event()
    release = Event()
    monkeypatch.setattr(groq_client, "_concurrency", BoundedSemaphore(1))
    def post(*args, **kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(3)
        raise TimeoutError
    monkeypatch.setattr(groq_client.requests, "post", post)
    ledger = MemoryLedger()
    def run(turn):
        with research_budget(lambda name, *_: ledger.reserve(turn, name)):
            return groq_client.call_groq([], stage="generation")
    with ThreadPoolExecutor() as pool:
        first = pool.submit(run, "turn")
        assert entered.wait(3)
        with pytest.raises(groq_client.GroqRateLimitError):
            run("other")
        release.set()
        with pytest.raises(TimeoutError):
            first.result()
    with pytest.raises(BudgetUnavailable):
        run("turn")
    assert len(calls) == 1


def test_shared_provider_lease_and_cooldown(monkeypatch):
    from google.cloud import firestore
    from groq_budget import firestore_finish, firestore_reserve

    monkeypatch.setattr(firestore, "transactional", lambda fn: fn)
    monkeypatch.setenv("LEGAL_GROQ_CONCURRENCY", "1")
    now = [1000.0]
    monkeypatch.setattr("groq_budget.time.time", lambda: now[0])

    rows = {}
    class Reference:
        def __init__(self, key):
            self.key = key
        def get(self, transaction):
            return SimpleNamespace(to_dict=lambda: rows.get(self.key))
    class Database:
        def collection(self, name):
            return SimpleNamespace(document=Reference)
        def transaction(self):
            return SimpleNamespace(set=lambda ref, value: rows.__setitem__(ref.key, value))
    db = Database()
    assert firestore_reserve(db, "turn1", "generation", 500, 4) == 1
    with pytest.raises(BudgetUnavailable):
        firestore_reserve(db, "turn2", "generation", 500, 4)
    firestore_finish(db, "turn1", "generation", retry_after=10)
    with pytest.raises(BudgetUnavailable):
        firestore_reserve(db, "turn2", "generation", 500, 4)
    now[0] += 11
    assert firestore_reserve(db, "turn2", "generation", 500, 4) == 1
    firestore_finish(db, "turn2", "generation", retry_after=0)
    with pytest.raises(BudgetUnavailable):
        firestore_reserve(db, "turn1", "generation", 500, 4)


def test_default_transport_has_no_sdk_retries():
    with groq_client.requests.Session() as session:
        assert session.get_adapter(groq_client.GROQ_URL).max_retries.total == 0
