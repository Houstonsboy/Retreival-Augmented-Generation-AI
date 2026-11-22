"""Analyst LLM client — 2-call budget with conditional call 2.

Call 1 (inspect): full extraction using ContractFacts schema, seed=1001.
Call 2 (verify):  conditional — only on triggers T1–T4, targeted schema, seed=2002.
  T1: case is provisionally certificate-eligible
  T2: model opinion on core element (C,D,E,G) differs from derived status
  T3: a core fact reference failed grounding
  T4: verdict is LIKELY VALID or POSSIBLY VALID WITH GAPS and a core element
      has only one supporting reference

Retries use the same call slot. A parse failure on call 1 uses call 2 as a
replacement full inspection. Both calls failing → AnalystError.

Call counter is returned so the pipeline can assert AC8.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from json_repair import loads as json_repair_loads
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

from pydantic import ValidationError

from groq_client import call_groq

from . import INSPECT_PROMPT, VERIFY_PROMPT
from .schema import ContractFacts, VerifyFacts
from .versions import MAX_CALLS, MODEL

_SCCVISION_API_KEY: str | None = os.getenv("GROQ_API_KEY_12") or None

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class AnalystError(RuntimeError):
    """Raised when the analyst response cannot be obtained or parsed."""


def _extract_json(raw: str) -> str:
    text = (raw or "").strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AnalystError("No JSON object found in analyst response")
    return text[start: end + 1]


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(_extract_json(raw))
    except (json.JSONDecodeError, AnalystError):
        if _HAS_JSON_REPAIR:
            try:
                return json_repair_loads(raw)  # type: ignore[return-value]
            except Exception:
                pass
        raise


def _groq_call(
    system: str,
    user_content: str,
    *,
    seed: int,
    max_tokens: int = 3000,
) -> str:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    return call_groq(
        msgs,
        model=MODEL,
        temperature=0,
        max_tokens=max_tokens,
        timeout=120,
        api_key=_SCCVISION_API_KEY,
        seed=seed,
        extra_body={"top_p": 1},
    )


def call_inspect(
    analysis_input: dict[str, Any],
) -> tuple[ContractFacts, int]:
    """Run call 1 (full inspection). Returns (ContractFacts, calls_used).

    Tries parse repair first. If call 1 fails to parse, calls the inspect
    prompt a second time as a fresh attempt. calls_used includes only
    successful-or-attempted API calls (not 429 retries).
    """
    user_content = _build_inspect_user(analysis_input)
    calls_used = 0
    last_error: Exception | None = None

    for attempt in range(MAX_CALLS):
        try:
            raw = _groq_call(INSPECT_PROMPT, user_content, seed=1001)
            calls_used += 1
        except Exception as exc:
            calls_used += 1
            last_error = exc
            break

        try:
            data = _parse_json(raw)
            facts = ContractFacts.model_validate(data)
            return facts, calls_used
        except (AnalystError, json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            if attempt == 0:
                # Let the second attempt try fresh (not in the retry loop of the
                # caller) — but only if we haven't burned both call slots.
                continue
            break

    raise AnalystError(
        f"Inspect call failed to parse after {calls_used} attempt(s): {last_error}"
    ) from last_error


def call_verify(
    analysis_input: dict[str, Any],
    targeted_fields: list[str],
) -> tuple[dict[str, Any] | None, int]:
    """Run call 2 (conditional verify). Returns (partial facts dict, calls_used=1).

    Sends only the targeted fields in the schema. Returns None on parse failure
    (soft failure — the pipeline uses only call 1's facts).
    """
    user_content = _build_verify_user(analysis_input, targeted_fields)
    try:
        raw = _groq_call(VERIFY_PROMPT, user_content, seed=2002, max_tokens=800)
    except Exception as exc:
        print(f"[WARN] verify call failed: {exc}")
        return None, 1

    try:
        data = _parse_json(raw)
        # Validate using VerifyFacts — extra fields are ignored
        vf = VerifyFacts.model_validate(data)
        return vf.model_dump(exclude_none=False), 1
    except (AnalystError, json.JSONDecodeError, ValidationError) as exc:
        print(f"[WARN] verify parse failed: {exc}")
        return None, 1


def _build_inspect_user(ai: dict[str, Any]) -> str:
    """Build the compact user message for the inspect call."""
    lines: list[str] = []

    # Compact message lines: idx | sender | ts_quality | text
    messages = ai.get("messages") or []
    if messages:
        lines.append("MESSAGES:")
        for m in messages:
            idx = m.get("index") or m.get("msg_index", "?")
            sender = "PARTY_1" if m.get("sender") == "me" else "PARTY_2"
            ts_q = m.get("timestamp_quality") or "NONE"
            text = (m.get("text") or "").replace("\n", " ")
            lines.append(f"{idx} | {sender} | {ts_q} | {text}")

    # Candidates block
    candidates = ai.get("candidates") or {}
    if candidates:
        lines.append(f"\ncandidates: {json.dumps(candidates, ensure_ascii=False)}")

    # M-Pesa extracted (compact)
    mpesa = ai.get("mpesa_extracted") or []
    if mpesa:
        lines.append("\nMPESA_RECORDS:")
        for p in mpesa:
            parts = []
            if p.get("code"):
                parts.append(f"code={p['code']}")
            if p.get("amount_kes") is not None:
                parts.append(f"amount={p['amount_kes']}")
            if p.get("recipient"):
                parts.append(f"recipient={p['recipient']}")
            if p.get("datetime"):
                parts.append(f"datetime={p['datetime']}")
            lines.append("  " + " ".join(parts))

    # Case context (stripped)
    ctx = ai.get("case_context") or {}
    lines.append(f"\ncase_context: role={ctx.get('user_role','unknown')} "
                 f"date={ctx.get('analysis_date','unknown')}")
    if ctx.get("user_account"):
        lines.append(f"user_account: {ctx['user_account'][:200]}")

    # Integrity flags (hints only)
    flags = ai.get("integrity_flags") or []
    if flags:
        lines.append(f"integrity_flags: {flags}")

    return "\n".join(lines)


def _build_verify_user(ai: dict[str, Any], targeted_fields: list[str]) -> str:
    """Build the targeted user message for the verify call."""
    lines: list[str] = []
    lines.append(f"Check these fields: {', '.join(targeted_fields)}\n")

    messages = ai.get("messages") or []
    if messages:
        lines.append("MESSAGES:")
        for m in messages:
            idx = m.get("index") or m.get("msg_index", "?")
            sender = "PARTY_1" if m.get("sender") == "me" else "PARTY_2"
            ts_q = m.get("timestamp_quality") or "NONE"
            text = (m.get("text") or "").replace("\n", " ")
            lines.append(f"{idx} | {sender} | {ts_q} | {text}")

    candidates = ai.get("candidates") or {}
    if candidates:
        lines.append(f"\ncandidates: {json.dumps(candidates, ensure_ascii=False)}")

    mpesa = ai.get("mpesa_extracted") or []
    if mpesa:
        lines.append("\nMPESA_RECORDS:")
        for p in mpesa:
            parts = []
            if p.get("code"):
                parts.append(f"code={p['code']}")
            if p.get("amount_kes") is not None:
                parts.append(f"amount={p['amount_kes']}")
            if p.get("datetime"):
                parts.append(f"datetime={p['datetime']}")
            lines.append("  " + " ".join(parts))

    return "\n".join(lines)
