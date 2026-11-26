"""Shared Groq HTTP transport with bounded attempts and provider admission."""
import os
import re
import time
import requests
from collections import deque
from threading import BoundedSemaphore, Lock
from groq_budget import ACTIVE, BudgetUnavailable, provider_limits

GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class GroqRateLimitError(RuntimeError):
    """Provider or local admission state prevents another attempt."""

class GroqPayloadTooLargeError(RuntimeError):
    """Raised when the request body exceeds Groq's size limit (HTTP 413).

    Rotating keys will not help — the payload itself must be reduced.
    """


def get_groq_keys() -> list[str]:
    """Return all configured Groq API keys, sorted by env-var name."""
    return [v for k, v in sorted(os.environ.items())
            if k == "GROQ_API_KEY" or k.startswith("GROQ_API_KEY_")]


_rate_lock = Lock()
_admissions = deque()
_blocked_until = 0.0
_rpm, _tpm, _max_concurrency = provider_limits()
_concurrency = BoundedSemaphore(_max_concurrency)



def _retry_after(value):
    from email.utils import parsedate_to_datetime
    if not value:
        return 30.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return 30.0


def call_groq(
    msgs: list,
    *,
    model: str = GROQ_MODEL,
    temperature: float = 0,
    max_tokens: int = 2500,
    timeout: float = 60,
    api_key: str | None = None,
    seed: int | None = None,
    extra_body: dict | None = None,
    stage: str | None = None,
) -> str:
    """Issue at most one HTTP attempt. Suppress redirects and provider retries."""
    global _blocked_until
    keys = get_groq_keys()
    key = api_key if api_key in keys else (keys[0] if keys else None)
    if not key:
        raise BudgetUnavailable("Groq is unavailable")
    budget = ACTIVE.get()
    if stage is not None and budget is None:
        raise BudgetUnavailable("Research stage has no shared Groq budget")

    estimated_tokens = sum(len(str(m.get("content", "")).encode()) for m in msgs) + max_tokens
    body = {"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    if extra_body:
        body.update({k: v for k, v in extra_body.items() if k not in body})
    if not _concurrency.acquire(blocking=False):
        raise GroqRateLimitError("Groq concurrency limit reached")
    started = time.monotonic()
    attempted = False
    status = None
    succeeded = False
    cooldown = 0.0
    admission = None
    try:
        with _rate_lock:
            now = time.monotonic()
            while _admissions and now - _admissions[0][0] >= 60:
                _admissions.popleft()
            if now < _blocked_until or len(_admissions) >= _rpm:
                raise GroqRateLimitError("Groq request rate limit reached")
            if estimated_tokens + sum(tokens for _, tokens in _admissions) > _tpm:
                raise GroqRateLimitError("Groq token limit reached")
            admission = (now, estimated_tokens)
            _admissions.append(admission)
        if budget:
            budget.take(stage, estimated_tokens, timeout)
        attempted = True
        resp = requests.post(
            GROQ_URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body, timeout=timeout, allow_redirects=False,
        )
        status = resp.status_code
        if status == 429:
            cooldown = _retry_after(resp.headers.get("Retry-After"))
            with _rate_lock:
                _blocked_until = max(_blocked_until, time.monotonic() + cooldown)
            raise GroqRateLimitError("Groq returned HTTP 429")
        if status == 413:
            raise GroqPayloadTooLargeError("Groq rejected the request size")
        if not 200 <= status < 300:
            raise requests.HTTPError(f"Groq returned HTTP {status}")
        payload = resp.json()
        if budget:
            budget.outcomes[stage]["usage"] = {
                k: payload.get("usage", {}).get(k, 0) for k in ("prompt_tokens", "completion_tokens")
            }
        result = _THINK_RE.sub("", payload["choices"][0]["message"]["content"]).strip()
        succeeded = True
        return result
    finally:
        _concurrency.release()
        if not attempted and admission:
            with _rate_lock:
                if admission in _admissions:
                    _admissions.remove(admission)
        if budget and attempted:
            if budget.finish:
                try:
                    budget.finish(stage, cooldown)
                except Exception:
                    # An unreleased lease expires and cannot free another stage.
                    pass
            budget.outcomes[stage].update({
                "status": status,
                "attempts": 1,
                "succeeded": succeeded,
                "retry_attempts": 0,
                "latency_ms": round((time.monotonic() - started) * 1000),
            })
