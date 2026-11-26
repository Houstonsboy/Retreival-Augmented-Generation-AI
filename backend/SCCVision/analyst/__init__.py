"""SCCVision Contract Evidence Analyst package.

Loads the inspect and verify prompts once at import time. The prompts use the
ContractFacts schema (v2). Legacy SYSTEM_PROMPT alias kept for any code that
imports it directly.
"""

from __future__ import annotations

from pathlib import Path

from .versions import PROMPT_VERSION, VERIFY_PROMPT_VERSION  # noqa: F401

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_INSPECT_PATH = _PROMPTS_DIR / "inspect.md"
_VERIFY_PATH = _PROMPTS_DIR / "verify.md"


def load_inspect_prompt() -> str:
    return _INSPECT_PATH.read_text(encoding="utf-8")


def load_verify_prompt() -> str:
    return _VERIFY_PATH.read_text(encoding="utf-8")


INSPECT_PROMPT: str = load_inspect_prompt()
VERIFY_PROMPT: str = load_verify_prompt()

# Legacy alias — kept so older imports don't break.
SYSTEM_PROMPT: str = INSPECT_PROMPT

__all__ = [
    "INSPECT_PROMPT",
    "VERIFY_PROMPT",
    "SYSTEM_PROMPT",
    "PROMPT_VERSION",
    "VERIFY_PROMPT_VERSION",
    "load_inspect_prompt",
    "load_verify_prompt",
]
