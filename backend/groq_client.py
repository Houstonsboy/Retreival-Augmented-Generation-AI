"""
Shared Groq API client with shuffled key rotation on 429/403.

Keys are shuffled on every call so rate-limit pressure is spread evenly across
all configured keys.  Pass ``preferred_key`` to try a specific key first (e.g.
a dedicated per-service key); if that key is rate-limited the client falls
through to the remaining keys in shuffled order — seamlessly, without the
caller knowing.

Do NOT change Qclassifier.py's Groq logic — that module manages its own key.
"""

import os
import random
import re
import time
import requests

GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class GroqRateLimitError(RuntimeError):
    """Raised when every configured Groq key is rate-limited."""

class GroqPayloadTooLargeError(RuntimeError):
    """Raised when the request body exceeds Groq's size limit (HTTP 413).

    Rotating keys will not help — the payload itself must be reduced.
    """


def get_groq_keys() -> list[str]:
    """Return all configured Groq API keys, sorted by env-var name."""
    return [v for k, v in sorted(os.environ.items())
            if k == "GROQ_API_KEY" or k.startswith("GROQ_API_KEY_")]


def _build_key_order(preferred_key: str | None) -> list[str]:
    """
    Return the list of keys to try for one round.

    If ``preferred_key`` is given it goes first; the remaining keys are
    shuffled so each call distributes load differently.
    """
    all_keys = get_groq_keys()
    if not all_keys:
        raise RuntimeError("No GROQ_API_KEY configured")

    if preferred_key and preferred_key in all_keys:
        rest = [k for k in all_keys if k != preferred_key]
        random.shuffle(rest)
        return [preferred_key] + rest

    shuffled = list(all_keys)
    random.shuffle(shuffled)
    return shuffled


def call_groq(
    msgs: list,
    *,
    model: str = GROQ_MODEL,
    temperature: float = 0,
    max_tokens: int = 2500,
    timeout: float = 60,
    api_key: str | None = None,
) -> str:
    """
    POST to Groq with shuffled key rotation on 429/403.

    On every call the key list is shuffled so load spreads across all keys.
    ``api_key`` is tried first; if it is rate-limited the call falls through
    to the remaining keys automatically — the caller never sees the switch.

    After all keys fail in one pass, waits 30 s and retries once more.
    Raises GroqRateLimitError if every key in both rounds fails.
    Raises requests.HTTPError immediately for non-rate-limit errors.
    Returns raw text with <think>…</think> blocks stripped.
    """
    _MAX_ROUNDS = 2

    for round_num in range(_MAX_ROUNDS):
        keys = _build_key_order(api_key)

        for idx, key in enumerate(keys):
            try:
                resp = requests.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    json={"model": model, "messages": msgs,
                          "temperature": temperature, "max_tokens": max_tokens},
                    timeout=timeout,
                )
                if resp.status_code == 413:
                    body_snippet = resp.text[:600]
                    print(f"[WARN] Groq 413 key-slot {idx} (model={model}): {body_snippet}")
                    # Groq uses 413 for two distinct problems:
                    #   • rate_limit_exceeded (TPM budget exhausted) — rotating keys helps
                    #   • request_too_large   (HTTP body too big)    — no key will help
                    try:
                        err_code = resp.json().get("error", {}).get("code", "")
                    except Exception:
                        err_code = ""
                    if err_code == "rate_limit_exceeded":
                        # Treat as 429: rotate to the next key
                        continue
                    # True payload-size error — stop immediately
                    raise GroqPayloadTooLargeError(
                        f"Request payload too large for Groq: {body_snippet}"
                    )
                if resp.status_code in (429, 403):
                    print(f"[WARN] Groq {resp.status_code} key-slot {idx} "
                          f"(model={model}): {resp.text[:200]}")
                    continue
                if not resp.ok:
                    print(f"[ERROR] Groq {resp.status_code} key-slot {idx} "
                          f"(model={model}): {resp.text[:500]}")
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                return _THINK_RE.sub("", raw).strip()
            except requests.HTTPError:
                raise

        if round_num < _MAX_ROUNDS - 1:
            n = len(keys)
            print(f"[WARN] All {n} Groq keys exhausted on round {round_num + 1}, "
                  "waiting 30s before retry…")
            time.sleep(30)

    raise GroqRateLimitError(
        "Groq API rate limit exceeded on all configured keys; wait a moment and retry"
    )
