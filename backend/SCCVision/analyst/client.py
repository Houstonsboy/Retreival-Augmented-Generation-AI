"""Analyst LLM client.

Wraps ``groq_client.call_groq`` (the shared advocate-side client) to call the
Contract Evidence Analyst system prompt with the masked ``analysis_input`` and
parse the JSON response into :class:`~SCCVision.analyst.models.AnalystResult`.

The response is expected to be a JSON object, possibly wrapped in markdown
fences. The parser extracts the outermost object and retries once with an
explicit correction instruction if validation fails.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import ValidationError

from groq_client import call_groq

from . import SYSTEM_PROMPT
from .models import AnalystResult

# The gpt-oss-* family has an 8K TPM cap per org on the on_demand tier,
# which is smaller than a single request (system prompt ~4.25K tokens alone).
# qwen/qwen3.8-27b has a higher TPM allowance and is used for the rest of
# the pipeline; test showed truncation was not an issue for typical cases.
SCCVISION_MODEL = "qwen/qwen3.8-27b"

# Dedicated key for self-litigant analysis — isolates SCCVision rate limits
# from the advocate research pipeline. Falls back to key rotation if not set.
_SCCVISION_API_KEY: str | None = os.getenv("GROQ_API_KEY_12") or None


class AnalystError(RuntimeError):
    """Raised when the analyst response cannot be obtained or parsed."""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_MAX_ATTEMPTS = 2


def _extract_json(raw: str) -> str:
    """Return the substring containing the outermost JSON object, if any."""
    text = (raw or "").strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AnalystError("No JSON object found in analyst response")
    return text[start : end + 1]


class AnalystLLM:
    """Stateless client for one analyst call. Safe to construct per request."""

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int = 8000,
        timeout: float = 120,
    ) -> None:
        self.model = model or SCCVISION_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(
        self,
        analysis_input: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> AnalystResult:
        """Send ``analysis_input`` and return a validated :class:`AnalystResult`.

        Retries once on a parse/validation failure by appending the failed
        response and a correction instruction. Raises :class:`AnalystError` if
        the response still cannot be parsed.
        """
        system = system_prompt or SYSTEM_PROMPT
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(analysis_input, ensure_ascii=False)},
        ]

        last_error: Exception | None = None
        for _ in range(_MAX_ATTEMPTS):
            raw = call_groq(
                msgs,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                api_key=_SCCVISION_API_KEY,
            )
            try:
                data = json.loads(_extract_json(raw))
                return AnalystResult.model_validate(data)
            except (AnalystError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                print(f"[DEBUG] Parse failed ({type(exc).__name__}): {str(exc)[:200]}")
                print(f"[DEBUG] Raw response (first 500 chars): {raw[:500]}")
                # Long response means truncation, not a format error. Appending it
                # to conversation history would push the retry's input over the
                # ITPM=7,000 limit, causing all keys to fail. Retry fresh instead.
                if len(raw) > 5000:
                    print(f"[DEBUG] Response likely truncated ({len(raw)} chars) — retrying fresh")
                    continue
                msgs.append({"role": "assistant", "content": raw})
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not a valid JSON object "
                            "matching the required schema. Return ONLY the JSON "
                            "object, with no markdown fences and no text outside it."
                        ),
                    }
                )

        raise AnalystError(
            f"Analyst response could not be parsed after {_MAX_ATTEMPTS} attempts"
        ) from last_error
