"""
Shared Groq API client with key rotation on 429.

Extracted from the _groq() nested closure in generate_legal_answer so it can
be reused by background jobs (e.g. conversation title generation) without
duplicating key-rotation logic.  The existing answer flow is behaviour-identical:
keys are rediscovered from the environment on every call (matching the closure
which was rebuilt per request), and key_idx resets to 0 each call (matching
the per-request _key_idx = [0] state).

Do NOT change Qclassifier.py's Groq logic — that module manages its own key.
"""

import os
import re
import time
import requests

# Copied verbatim from the model field in the original _groq() closure (app.py:1477)
GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def get_groq_keys() -> list[str]:
    """Return all configured Groq API keys, sorted by env var name."""
    return [v for k, v in sorted(os.environ.items())
            if k == "GROQ_API_KEY" or k.startswith("GROQ_API_KEY_")]


def call_groq(
    msgs: list,
    *,
    model: str = GROQ_MODEL,
    temperature: float = 0,
    max_tokens: int = 2500,
    timeout: float = 60,
) -> str:
    """
    POST to Groq completions with key rotation on 429.

    Discovers GROQ_API_KEY / GROQ_API_KEY_* from the environment on every call,
    which is behaviour-identical to the per-request closure it replaced: the
    closure captured keys at the start of generate_legal_answer (one request),
    and call_groq rediscovers them at the start of each call (still one request
    for the synthesis path, one job for the title path).

    Raises RuntimeError if all keys exhausted; raises requests.HTTPError for
    non-429 HTTP errors (same as the original _groq closure).
    Returns raw text with <think>…</think> blocks stripped.
    """
    groq_keys = get_groq_keys()
    if not groq_keys:
        raise RuntimeError("No GROQ_API_KEY configured")

    key_idx = [0]

    for _attempt in range(len(groq_keys) + 1):
        key = groq_keys[key_idx[0]]
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": model, "messages": msgs,
                      "temperature": temperature, "max_tokens": max_tokens},
                timeout=timeout,
            )
            if resp.status_code == 429:
                next_idx = (key_idx[0] + 1) % len(groq_keys)
                if next_idx != key_idx[0]:
                    print(f"[WARN] Groq 429 on key {key_idx[0]}, rotating to {next_idx}")
                    key_idx[0] = next_idx
                    continue
                print("[WARN] All Groq keys rate-limited, waiting 30s...")
                time.sleep(30)
                continue
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return _THINK_RE.sub("", raw).strip()
        except requests.HTTPError:
            raise
    raise RuntimeError("All Groq keys exhausted")
