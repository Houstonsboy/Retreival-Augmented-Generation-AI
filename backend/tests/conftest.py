"""Stub only packages that are genuinely absent from the venv.

Packages confirmed absent: rapidocr_onnxruntime
Everything else (cv2, numpy, PIL, torch, sentence_transformers, chromadb,
langchain_core, pypdf, groq, firebase_admin…) is installed — do NOT stub
those, because stubs block the real implementations that tests rely on.

firebase_admin and groq are stubbed to avoid credential/key side-effects at
import time; application tests that need their real API mock them individually.
"""
from __future__ import annotations

import os
import sys
import types

# ---------------------------------------------------------------------------
# Env-var stubs: prevent module-level raises when env vars are absent
# ---------------------------------------------------------------------------
os.environ.setdefault("GROQ_API_KEY", "test-key-stub")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _pkg(name: str, **attrs):
    mod = _stub(name, **attrs)
    mod.__path__ = []  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# rapidocr_onnxruntime — genuinely not installed
# ---------------------------------------------------------------------------
ro_mod = _stub("rapidocr_onnxruntime")
ro_mod.RapidOCR = object  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# firebase_admin — installed but requires credentials at import time;
# stub to avoid 503/credential errors in unit tests that don't hit Firestore
# ---------------------------------------------------------------------------
fa = _stub("firebase_admin")
fa._apps = {}
fa.initialize_app = lambda *a, **kw: None  # type: ignore[attr-defined]

fa_creds = _pkg("firebase_admin.credentials")

class _Certificate:
    def __init__(self, *args, **kwargs): pass

fa_creds.Certificate = _Certificate
fa.credentials = fa_creds

fa_auth_mod = _pkg("firebase_admin.auth")
fa_auth_mod.verify_id_token = lambda *a, **kw: {}  # type: ignore[attr-defined]
fa_auth_mod.ExpiredIdTokenError = Exception
fa.auth = fa_auth_mod

fa_fs = _pkg("firebase_admin.firestore")
fa_fs.SERVER_TIMESTAMP = None
fa_fs.client = lambda: None  # type: ignore[attr-defined]
fa.firestore = fa_fs

fa_storage = _pkg("firebase_admin.storage")
fa_storage.bucket = lambda *a, **kw: None  # type: ignore[attr-defined]
fa.storage = fa_storage
