"""One attempt per stage, at most three across both research endpoints."""
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from hashlib import sha256
from threading import Lock
import time

STAGES = frozenset({"classification", "generation", "verification"})
LIMIT = len(STAGES)

class BudgetUnavailable(RuntimeError):
    """No outbound request is authorized."""

@dataclass
class QueryBudget:
    reserve: Callable[[str, int, float], int]
    finish: Callable[[str, float], None] | None = None
    outcomes: dict = field(default_factory=dict)

    def take(self, stage, estimated_tokens, timeout):
        if stage not in STAGES:
            raise BudgetUnavailable("A research request requires an explicit Groq stage")
        try:
            count = self.reserve(stage, estimated_tokens, timeout)
        except BudgetUnavailable:
            raise
        except Exception:
            raise BudgetUnavailable("Groq budget ledger unavailable") from None
        self.outcomes[stage] = {"attempts_remaining": LIMIT - count}

class MemoryLedger:
    """Thread-safe ledger for isolated tests and single-command evaluations."""
    def __init__(self):
        self.lock = Lock()
        self.used = {}

    def reserve(self, key, stage):
        with self.lock:
            used = self.used.setdefault(key, set())
            if stage in used or len(used) >= LIMIT:
                raise BudgetUnavailable("Groq stage already consumed")
            used.add(stage)
            return len(used)

def turn_key(uid, turn_id, query):
    # Legacy clients share one short-lived turn per identical question.
    turn = str(turn_id) if turn_id else f"legacy-{int(time.time() // 3600)}"
    return sha256(f"{uid}\n{turn}\n{query}".encode()).hexdigest()

def provider_limits():
    return (
        int(os.getenv("LEGAL_GROQ_RPM", "20")),
        int(os.getenv("LEGAL_GROQ_TPM", "60000")),
        int(os.getenv("LEGAL_GROQ_CONCURRENCY", "2")),
    )


def _provider_ref(db):
    return db.collection("legal_query_budgets").document("provider-admission")


def firestore_reserve(db, key, stage, estimated_tokens, timeout):
    """Reserve the turn stage and a cross-worker provider lease atomically."""
    from google.cloud import firestore
    ref = db.collection("legal_query_budgets").document(key)
    provider = _provider_ref(db)
    rpm, tpm, concurrency = provider_limits()

    @firestore.transactional
    def reserve(transaction):
        row = ref.get(transaction=transaction).to_dict() or {}
        state = provider.get(transaction=transaction).to_dict() or {}
        used = set(row.get("stages", []))
        if stage in used or len(used) >= LIMIT:
            raise BudgetUnavailable("Groq stage already consumed")
        now = time.time()
        events = [e for e in state.get("events", []) if now - e["at"] < 60]
        leases = {k: v for k, v in state.get("leases", {}).items() if now < v}
        if now < state.get("blocked_until", 0) or len(events) >= rpm or len(leases) >= concurrency:
            raise BudgetUnavailable("Provider admission limit reached")
        if sum(e["tokens"] for e in events) + estimated_tokens > tpm:
            raise BudgetUnavailable("Provider token limit reached")
        used.add(stage)
        events.append({"at": now, "tokens": estimated_tokens})
        leases[f"{key}:{stage}"] = now + timeout + 10
        transaction.set(ref, {"stages": sorted(used), "updated_at": datetime.now(timezone.utc)})
        transaction.set(provider, {**state, "events": events, "leases": leases})
        return len(used)
    return reserve(db.transaction())


def firestore_finish(db, key, stage, retry_after):
    """Release a lease and share any provider cooldown with other workers."""
    from google.cloud import firestore
    provider = _provider_ref(db)

    @firestore.transactional
    def finish(transaction):
        state = provider.get(transaction=transaction).to_dict() or {}
        leases = state.get("leases", {})
        leases.pop(f"{key}:{stage}", None)
        blocked = max(state.get("blocked_until", 0), time.time() + retry_after) if retry_after else state.get("blocked_until", 0)
        transaction.set(provider, {**state, "leases": leases, "blocked_until": blocked})
    finish(db.transaction())


ACTIVE = ContextVar("research_groq_budget", default=None)

@contextmanager
def research_budget(reserve, finish=None):
    budget = QueryBudget(reserve, finish)
    token = ACTIVE.set(budget)
    try:
        yield budget
    finally:
        ACTIVE.reset(token)
