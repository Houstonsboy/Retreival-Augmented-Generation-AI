"""Stub heavy native dependencies before any test collection.

This avoids ImportError for cv2, onnxruntime, etc. that are only needed in
the OCR service layer, not in the deterministic logic under test.
"""
from __future__ import annotations

import sys
import types


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# --- cv2 / numpy / onnxruntime stubs ---
_stub("cv2")
_stub("numpy")
_stub("numpy.random")

ort_mod = _stub("onnxruntime")
ort_mod.InferenceSession = object  # type: ignore

# --- rapidocr stubs ---
ro_mod = _stub("rapidocr_onnxruntime")
ro_mod.RapidOCR = object  # type: ignore

# --- firebase_admin stub (just enough for analysis_store imports) ---
fa = _stub("firebase_admin")
fa_fs = _stub("firebase_admin.firestore")
fa_fs.SERVER_TIMESTAMP = None
fa_fs.client = lambda: None  # type: ignore

# --- groq / requests stubs (not needed for unit tests) ---
_stub("groq")
_stub("requests")
