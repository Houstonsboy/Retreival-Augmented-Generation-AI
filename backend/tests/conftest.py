"""Stub heavy native dependencies before any test collection.

This avoids ImportError for cv2, onnxruntime, ML libraries, etc. that are only
needed in the OCR/RAG service layer, not in the deterministic logic under test.

numpy IS available in the venv (v2.3.5) and is used directly — no stub needed.
cv2, onnxruntime, rapidocr and the ML stack (torch, sentence_transformers,
chromadb…) are NOT installed, so they are stubbed here.
"""
from __future__ import annotations

import importlib.machinery as _machinery
import os
import sys
import types
import typing as _typing

# Provide dummy env-vars so modules that read them at import time don't raise.
os.environ.setdefault("GROQ_API_KEY", "test-key-stub")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _pkg(name: str, **attrs):
    """Stub a package (sets __path__ so sub-module imports don't explode)."""
    mod = _stub(name, **attrs)
    mod.__path__ = []  # type: ignore
    return mod


# ---------------------------------------------------------------------------
# cv2 — not installed; give it a spec so importlib.util.find_spec won't raise
# ---------------------------------------------------------------------------
cv2_mod = _stub("cv2")
cv2_mod.__spec__ = _machinery.ModuleSpec("cv2", None)

# ---------------------------------------------------------------------------
# onnxruntime / rapidocr — not installed
# ---------------------------------------------------------------------------
ort_mod = _stub("onnxruntime")
ort_mod.InferenceSession = object  # type: ignore

ro_mod = _stub("rapidocr_onnxruntime")
ro_mod.RapidOCR = object  # type: ignore

# ---------------------------------------------------------------------------
# PIL / Pillow — might not be installed; stub so SCCVision.preprocess imports
# ---------------------------------------------------------------------------
try:
    import PIL as _pil_check  # noqa: F401
except ImportError:
    pil_root = _pkg("PIL")
    pil_root.UnidentifiedImageError = Exception  # type: ignore
    for _pil_sub in ("PIL.Image", "PIL.ImageOps", "PIL.ImageFont",
                     "PIL.ImageDraw", "PIL.ImageEnhance"):
        _pkg(_pil_sub)

# ---------------------------------------------------------------------------
# Heavy ML libraries imported by digester.py / pixe.py / Qclassifier.py etc.
# Stubbed at module level so the full app can be imported without torch/HF.
# ---------------------------------------------------------------------------
for _n in (
    "torch", "torch.nn", "torch.utils", "torch.utils.data",
    "torch.utils.data._utils", "torch.utils.data._utils.collate",
    "transformers", "transformers.utils", "transformers.utils.import_utils",
    "tqdm", "tqdm.auto",
    "huggingface_hub",
    "datasets",
    "accelerate",
    "tokenizers",
    "safetensors",
):
    _pkg(_n)

st_mod = _pkg("sentence_transformers")
st_mod.SentenceTransformer = object  # type: ignore
st_mod.CrossEncoder = object  # type: ignore
_pkg("sentence_transformers.util")

lc_core = _pkg("langchain_core")
lc_docs = _pkg("langchain_core.documents")
lc_docs.Document = object  # type: ignore
lc_core.documents = lc_docs  # type: ignore

lc_split = _pkg("langchain_text_splitters")
lc_split.RecursiveCharacterTextSplitter = object  # type: ignore

pypdf_mod = _pkg("pypdf")
pypdf_mod.PdfReader = object  # type: ignore

dotenv_mod = _stub("dotenv")
dotenv_mod.load_dotenv = lambda *a, **kw: None  # type: ignore

# ---------------------------------------------------------------------------
# chromadb — stub entire package tree (pulls in numpy internals we don't need)
# ---------------------------------------------------------------------------
_pkg("chromadb")
for _sub in (
    "chromadb.api", "chromadb.api.client", "chromadb.api.types",
    "chromadb.base_types", "chromadb.errors",
    "chromadb.db", "chromadb.db.base", "chromadb.db.impl",
    "chromadb.db.impl.grpc", "chromadb.db.impl.grpc.client",
    "chromadb.segment", "chromadb.segment.impl",
    "chromadb.segment.impl.metadata", "chromadb.types",
    "chromadb.utils", "chromadb.utils.embedding_functions",
    "chromadb.telemetry", "chromadb.telemetry.opentelemetry",
):
    _pkg(_sub)

# chromadb.config.Settings is used directly in constitution_digester.py
_chroma_config = _pkg("chromadb.config")
_chroma_config.Settings = object  # type: ignore

# ---------------------------------------------------------------------------
# firebase_admin — just enough for unit test imports
# ---------------------------------------------------------------------------
fa = _stub("firebase_admin")
fa._apps = {}
fa.initialize_app = lambda *a, **kw: None  # type: ignore

fa_creds = _pkg("firebase_admin.credentials")
fa_creds.Certificate = object  # type: ignore
fa.credentials = fa_creds

fa_auth_mod = _pkg("firebase_admin.auth")
fa_auth_mod.verify_id_token = lambda *a, **kw: {}  # type: ignore
fa_auth_mod.ExpiredIdTokenError = Exception
fa.auth = fa_auth_mod

fa_fs = _pkg("firebase_admin.firestore")
fa_fs.SERVER_TIMESTAMP = None
fa_fs.client = lambda: None  # type: ignore
fa.firestore = fa_fs

fa_storage = _pkg("firebase_admin.storage")
fa_storage.bucket = lambda *a, **kw: None  # type: ignore
fa.storage = fa_storage

# ---------------------------------------------------------------------------
# groq SDK (not installed; groq_client uses requests directly)
# ---------------------------------------------------------------------------
groq_mod = _stub("groq")
groq_mod.Groq = object  # type: ignore

# requests is NOT stubbed — the real package is in the venv and
# google-api-core (pulled in via history_store.py → google.cloud.firestore)
# accesses requests.exceptions at import time; an empty stub broke that import.
