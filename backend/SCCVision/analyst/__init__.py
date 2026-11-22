"""SCCVision Contract Evidence Analyst package.

Loads the analyst system prompt once at import time and exposes helpers used by
the pipeline (prompt version, prompt text). The prompt is the only instruction
content; conversation data is always supplied separately in the user message.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.md"


def load_system_prompt() -> str:
    """Return the analyst system prompt, loaded fresh from disk."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def prompt_version(prompt_text: str | None = None) -> str:
    """First 8 hex chars of the sha256 of the system prompt."""
    text = prompt_text if prompt_text is not None else load_system_prompt()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# Loaded once at import; used by the pipeline and by tests.
SYSTEM_PROMPT = load_system_prompt()
PROMPT_VERSION = prompt_version(SYSTEM_PROMPT)

__all__ = ["SYSTEM_PROMPT", "PROMPT_VERSION", "load_system_prompt", "prompt_version"]
