"""Lazily-created singleton for the RapidOCR engine."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

_engine: Any = None
_lock = threading.Lock()

# Tiny blank image used to force model loading during warmup.
_WARMUP_SIZE = 32


def get_engine() -> Any:
    """Return the shared RapidOCR instance, creating it on first use.

    The instance is created once and reused across requests; model loading
    (which is the expensive part) happens on the first inference call.
    """
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                from rapidocr import RapidOCR

                _engine = RapidOCR()
    return _engine


def warmup() -> None:
    """Load the engine and run one tiny inference so the first request is fast."""
    engine = get_engine()
    blank = np.full((_WARMUP_SIZE, _WARMUP_SIZE, 3), 255, dtype=np.uint8)
    engine(blank)
