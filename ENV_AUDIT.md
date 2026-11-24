# ENV_AUDIT.md — Implementation Environment Audit
## GraphRAG Legal Decision-Support System for Kenya's Small Claims Court
**Audit date:** 2026-11-17  
**Auditor note:** Every value is sourced from a specific command or file. The source is stated for each claim. Secret values (API keys, tokens, passwords) are never printed — only variable names.

---

## 5.2.1 Hardware and Operating System

### 5.2.1.1 CPU

| Field | Value | Source |
|---|---|---|
| Chip model | Intel Core i7-8665U @ 1.90 GHz (Whiskey Lake, 2018) | `lscpu` → Model name |
| Physical cores | 4 | `lscpu` → Core(s) per socket |
| Logical processors | 8 (Hyper-Threading: 2 threads per core) | `lscpu` → CPU(s) |
| Base clock | 1.90 GHz | `lscpu` → Model name |
| L1d cache | 128 KiB (4 instances, 32 KiB each) | `lscpu` → L1d cache |
| L2 cache | 1 MiB (4 instances, 256 KiB each) | `lscpu` → L2 cache |
| L3 cache | 8 MiB (shared) | `lscpu` → L3 cache |
| SIMD support | AVX2, AVX, FMA, SSE4.1, SSE4.2 | `lscpu` → Flags |
| Virtualisation | Intel VT-x | `lscpu` → Virtualization |
| Architecture | x86_64 | `lscpu` → Architecture |

The CPU includes AVX2 and FMA instructions. PyTorch and ONNX Runtime both detect and use these for CPU inference, which benefits the embedding model and cross-encoder.

### 5.2.1.2 Memory

| Field | Value | Source |
|---|---|---|
| Total RAM | 15 GiB | `free -h` → Mem: total |
| Used (at time of audit) | ~11 GiB | `free -h` → Mem: used |
| Available | ~4.4 GiB | `free -h` → Mem: available |
| Swap total | 15 GiB | `free -h` → Swap: total |
| Swap used | ~1.3 GiB | `free -h` → Swap: used |

The running system had ~4.4 GiB available at the time of measurement. Both the Flask backend (~1 GiB RSS) and the Next.js dev server (~800 MB RSS) were already running at that time (see §5.2.3 Process Memory).

### 5.2.1.3 Disk

| Field | Value | Source |
|---|---|---|
| Disk device | nvme0n1 (NVMe SSD) | `lsblk -d -o NAME,SIZE,ROTA,TRAN` |
| Total disk capacity | 476.9 GB | `lsblk` → SIZE |
| Rotational | No (ROTA=0, NVMe SSD) | `lsblk` → ROTA |
| Mounted partition | /dev/nvme0n1p2 | `df -h /home/gift/Retreival-Augmented-Generation-AI` |
| Partition size | 452 GB | `df -h` → Size |
| Partition used | 356 GB | `df -h` → Used |
| Partition available | 74 GB | `df -h` → Avail |
| Use% | 83% | `df -h` → Use% |
| Filesystem type | ext4 (inferred from Kali Linux default) | UNCLEAR — `df` does not show fstype |

**Disk space occupied by project artefacts:**

| Artefact | Size | Source |
|---|---|---|
| `backend/venv/` (Python virtualenv + all packages) | 7.6 GB | `du -sh backend/venv` |
| `frontend/node_modules/` | 1.1 GB | `du -sh frontend/node_modules` |
| `~/.cache/huggingface/hub/` (all HF models) | 5.6 GB | `du -sh ~/.cache/huggingface` |
| `~/.cache/chroma/` (ChromaDB ONNX model: all-MiniLM-L6-v2) | 167 MB | `du -sh ~/.cache/chroma` |
| `~/.cache/torch/` | NOT FOUND | `du -sh ~/.cache/torch` |
| `backend/CaseLawDB/` (ChromaDB case-law collection) | 4.7 MB | `du -sh backend/CaseLawDB` |
| `backend/StatutesDB/` (ChromaDB statutes collection) | 20 MB | `du -sh backend/StatutesDB` |
| `backend/constitutionlogic/MasterRulesDB/` (ChromaDB master-rules) | 8.3 MB | `du -sh backend/constitutionlogic/MasterRulesDB` |
| `backend/locations.db` (SQLite bbox index) | 280 KB | `du -sh backend/locations.db` |
| `backend/SCCCases/` (source case PDFs) | 1.7 MB | `du -sh backend/SCCCases` |
| `data/raw_statutes/` (14 HTML statute files) | 4.6 MB | `du -sh data/raw_statutes` |
| `backend/MasterRules/` (source PDFs: Constitution, Evidence Act) | 2.8 MB | `du -sh backend/MasterRules` |
| `backend/Repo/` (uploaded research documents) | 8.2 MB | `du -sh backend/Repo` |

### 5.2.1.4 GPU

| Field | Value | Source |
|---|---|---|
| Discrete/CUDA GPU | None | `nvidia-smi`: NOT FOUND |
| Integrated GPU | Intel UHD Graphics 620 (WhiskeyLake-U GT2) | `lspci | grep -i vga` |
| CUDA available | No | No CUDA driver or `nvidia-smi` present |

All ML inference (embedding, cross-encoder reranking, RapidOCR) runs on CPU only. PyTorch is installed with CUDA libraries bundled (see pip freeze: `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`, etc.), but these are unused stubs since no NVIDIA GPU is present. The CPU-only code path is the runtime path.

### 5.2.1.5 Operating System and Kernel

| Field | Value | Source |
|---|---|---|
| OS distribution | Kali GNU/Linux Rolling, version 2025.2 | `cat /etc/os-release` |
| OS family | Debian-based (`ID_LIKE=debian`) | `cat /etc/os-release` |
| Kernel version | 6.12.38+kali-amd64 | `uname -r` |

---

## 5.2.2 Software Versions Installed

### 5.2.2.1 Python Runtime

| Field | Value | Source |
|---|---|---|
| Python version inside `backend/venv` | **Python 3.13.7** | `venv/bin/python --version` |
| Virtualenv location | `backend/venv/` | Filesystem |

**Note:** The `.gitignore` and project documentation do not pin a Python version. Python 3.13.7 is what is installed on the development machine. This is a relatively recent Python release; a fresh clone on a machine with an older Python may need adjustment.

### 5.2.2.2 Python Package Versions (Installed vs requirements.txt)

The table below lists every package named in `backend/requirements.txt` with its installed version from `venv/bin/pip freeze`. Mismatches are marked.

| Package | requirements.txt | Installed (pip freeze) | Match? |
|---|---|---|---|
| python-dotenv | 1.2.1 | 1.2.1 | ✓ |
| huggingface-hub | 0.36.0 | 0.36.0 | ✓ |
| langchain | 1.0.7 | 1.0.7 | ✓ |
| langchain-core | 1.0.6 | 1.0.6 | ✓ |
| scikit-learn | 1.7.2 | 1.7.2 | ✓ |
| numpy | 2.3.5 | 2.3.5 | ✓ |
| **requests** | **2.32.5** | **2.34.2** | **✗ MISMATCH** |
| pypdf | 6.3.0 | 6.3.0 | ✓ |
| **PyMuPDF** | **1.24.14** | **1.26.6** | **✗ MISMATCH** |
| pytesseract | 0.3.13 | 0.3.13 | ✓ |
| **Pillow** | **11.1.0** | **12.0.0** | **✗ MISMATCH** |
| **ocrmypdf** | **16.5.0** | **17.13.0** | **✗ MISMATCH** |
| python-docx | 1.2.0 | 1.2.0 | ✓ |
| rank-bm25 | 0.2.2 | 0.2.2 | ✓ |
| torch | 2.9.1 | 2.9.1 | ✓ |
| transformers | 4.57.1 | 4.57.1 | ✓ |
| numpy | 2.3.5 | 2.3.5 | ✓ |
| sentence-transformers | 5.1.2 | 5.1.2 | ✓ |
| chromadb | 1.3.5 | 1.3.5 | ✓ |
| opentelemetry-api | 1.45.1 | 1.45.1 | ✓ |
| firebase-admin | 7.7.0 | 7.7.0 | ✓ |
| groq (SDK — installed but not used at runtime) | 0.35.0 | 0.35.0 | ✓ |
| rapidocr | 3.10.0 | 3.10.0 | ✓ |
| onnxruntime | 1.23.2 | 1.23.2 | ✓ |
| opencv-python-headless | 5.0.0.93 | 5.0.0.93 | ✓ |

**Four packages are installed at versions higher than those pinned in requirements.txt.** None of these mismatches affect runtime correctness for the documented functionality, but they mean `pip install -r requirements.txt` on a fresh machine would produce a different environment than is currently running.

The following packages are **installed in the venv but not in requirements.txt** and are relevant to note:

| Package | Installed version | Why notable |
|---|---|---|
| pydantic | 2.14.2 | Used by SCCVision `analyst/models.py` for structured LLM output |
| google-cloud-firestore | 2.34.1 | Installed as a dependency of `firebase-admin`; can be used directly |
| pytest | 9.2.0 | Test runner (dev dependency — not needed to run the app) |
| pytest-mock | 3.17.0 | Test dependency |
| langchain-text-splitters | 1.0.0 | LangChain sub-package |
| fastapi | 0.124.2 | Installed but NOT used in the application (Flask is used instead) |
| uvicorn | 0.38.0 | Installed but NOT used (no ASGI server) |
| anthropic | 0.75.0 | Installed but NOT used in this application |
| openai | 2.9.0 | Installed but NOT used (Groq is called directly via `requests`) |
| pdfminer.six | 20260107 | PDF text extraction (may be used by LangChain or ocrmypdf internally) |
| fpdf2 | 2.9.0 | PDF generation library |
| RapidFuzz | 3.14.6 | Fuzzy string matching; used in `compute_constitution_bboxes.py` |

Source: `venv/bin/pip freeze` (full output 196 lines).

### 5.2.2.3 System Python Libraries Usage at Runtime vs Ingestion-Only

The following PDF and OCR libraries are present in the venv. This table documents whether each is imported by the **running application** (runtime) or only by **offline ingestion/build scripts** (not needed to serve requests):

| Library | Runtime or Ingestion? | Importing files |
|---|---|---|
| PyMuPDF (`fitz`) | **Both** — runtime (via `app.py` top-level import of `CaseLaw.firac` and `constitutionlogic.constitutionbreaker`, which import fitz) | `app.py:30,34`, `CaseLaw/firac.py:2`, `constitutionlogic/constitutionbreaker.py:1`, `constitutionlogic/constitution.py:1`, `constitutionlogic/evidencedigester.py:6`, `build_case_chunks.py:154`, `compute_constitution_bboxes.py:19` |
| pypdf | Runtime (called by `app.py` indirectly via LangChain or ingestion routes) | `app.py` and ingestion routes |
| pytesseract | Ingestion-only | `constitutionlogic/constitutionbreaker.py:2`, `constitutionlogic/constitution.py:2`, `CaseLaw/firac.py:3`, `constitutionlogic/evidencedigester.py:7` — none of these are imported by the Flask request path |
| ocrmypdf | Ingestion-only | Not imported in any runtime module; used only in offline PDF pre-processing |
| rapidocr | Runtime (SCCVision) | `SCCVision/engine.py` — loaded at Flask startup via `sccvision_warmup()` (`app.py:25,1899`) |
| onnxruntime | Runtime (via rapidocr and sentence-transformers) | Loaded when rapidocr or cross-encoder model is used |
| opencv-python-headless | Runtime (via rapidocr preprocessing) | `SCCVision/preprocess.py` |
| Pillow | Runtime (via rapidocr and image upload handling) | `SCCVision/preprocess.py` |
| python-docx | Ingestion-only | `build_case_chunks.py` — DOCX ingestion; not imported in Flask routes |

### 5.2.2.4 Node.js and Frontend

| Field | Value | Source |
|---|---|---|
| Node.js version | 20.19.2 | `node --version` |
| npm version | 9.2.0 | `npm --version` |

**Installed frontend packages** (from `npm ls --depth=0` in `frontend/`):

| Package | Installed version |
|---|---|
| next | 16.0.3 |
| react | 19.2.0 |
| react-dom | 19.2.0 |
| firebase | 10.14.1 |
| react-pdf | 7.7.3 |
| react-markdown | 10.1.0 |
| remark-gfm | 4.0.1 |
| tailwindcss | 4.1.17 |
| @tailwindcss/postcss | 4.1.17 |
| typescript | 5.9.3 |
| lucide-react | 0.562.0 |
| @types/node | 20.19.25 |
| @types/react | 19.2.6 |
| @types/react-dom | 19.2.3 |
| eslint | 9.39.1 |
| eslint-config-next | 16.0.3 |

**Notable:** `firebase` is installed at **10.14.1**, while `package.json` specifies `^10.12.0`. This is a minor patch upgrade within the same major and is expected npm behaviour.

Source: `frontend/npm ls --depth=0`.

### 5.2.2.5 System Binaries

| Binary | Installed version | Source |
|---|---|---|
| tesseract | 5.5.0 (Leptonica 1.84.1) | `tesseract --version` |
| ocrmypdf | NOT in system PATH; available in venv at 17.13.0 | `which ocrmypdf` → not found; `venv/bin/ocrmypdf --version` → 17.13.0 |
| firebase CLI | 15.9.0 | `firebase --version` |
| java | OpenJDK 21.0.8 | `java -version` |
| git | 2.51.0 | `git --version` |

Java 21 is required by the Firebase Firestore emulator (`firebase emulators:start`). It is installed, so the emulator can be run for testing.

---

## 5.2.2.6 ML Models, Local Files, and Licences

### LLM (Groq — remote API)

| Field | Value | Source |
|---|---|---|
| Model ID | `qwen/qwen3.8-27b` | `groq_client.py:19`, `Qclassifier.py:26` (uses its own constant `MODEL_NAME`) |
| Hosting | Groq Cloud API (`https://api.groq.com/openai/v1/chat/completions`) | `groq_client.py:20`, `Qclassifier.py:27` |
| Local copy | None — inference is remote | — |

**Groq call parameters per call site:**

| Call site | temperature | max_tokens | timeout | Notes |
|---|---|---|---|---|
| `groq_client.call_groq` default | 0 | 2500 | 60 s | Used by `app.py` generate endpoint, citation verifier regeneration |
| `Qclassifier.py:460–477` | 0.1 | 1500 | 30 s | Query classification; uses own key rotation loop |
| `Retriever.py:179–183` | 0.3 | 1000 | 30 s | Optional case summary generation (not in main retrieve path) |
| `history_store.py:446–448` | 0.2 | 300 | 10 s | Conversation title generation (background thread) |
| `SCCVision/analyst/client.py:65–67` | 0 | 8000 | 120 s | Section 106B admissibility analysis |

**Key rotation and retry behaviour** (`groq_client.py:83–136`):
- Maximum rounds: 2 (`_MAX_ROUNDS = 2`)
- Between rounds: 30-second sleep (`time.sleep(30)`)
- Key order: shuffled randomly on each call, with optional `preferred_key` placed first
- 429/403: rotate to next key; 413 rate-limit (TPM exceeded): also rotate; 413 request-too-large: raise `GroqPayloadTooLargeError` immediately
- After both rounds exhausted: raise `GroqRateLimitError`
- Configured keys at audit time: 17 keys (`GROQ_API_KEY`, `GROQ_API_KEY_1` through `GROQ_API_KEY_15`) — backend `.env` key names listed; Qclassifier reads only `GROQ_API_KEY`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3`; `SCCVision/analyst/client.py:34` reads `GROQ_API_KEY_12` as a preferred key

### Embedding Model (Local CPU)

| Field | Value | Source |
|---|---|---|
| Model ID | `intfloat/e5-base-v2` | `config.py` → `EMBEDDING_MODEL_NAME` |
| Embedding dimension | 768 | E5-base architecture (standard; confirmed by HF model card) |
| Cache location | `~/.cache/huggingface/hub/models--intfloat--e5-base-v2/` | `find ~/.cache/huggingface` |
| Cache size | 419 MB | `du -sh ~/.cache/huggingface/hub/models--intfloat--e5-base-v2` |
| Licence | MIT | `README.md` in model snapshot: `license: mit` |
| Query prefix used | `"query: " + query_text` | `Retriever.py:626`, `RulesRetriever.py:547` |
| Load timing | **Loaded on every `/api/qretrieve` request** (not at startup) | `Retriever.py:601`: `SentenceTransformer(EMBEDDING_MODEL_NAME)` is inside `retrieve_relevant_cases()`. Same pattern in `RulesRetriever.py:498`. Not a module-level variable. |

**Performance implication:** Loading the 419 MB model from disk adds latency to every retrieval request. HuggingFace's `SentenceTransformer` class caches the model in memory after the first load within the same Python process, but only if the object is not garbage-collected. Since the model is a local variable inside the function, it is eligible for GC between calls. Under load, this means repeated 1–3 s model-load delays per request.

### Cross-Encoder (Local CPU)

| Field | Value | Source |
|---|---|---|
| Model ID | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `Retriever.py:84`, `RulesRetriever.py:78` |
| Cache location | `~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/` | `find ~/.cache/huggingface` |
| Cache size | 88 MB | `du -sh` |
| Licence | Apache 2.0 | `README.md` in model snapshot: `license: apache-2.0` |
| Load timing | **Loaded lazily, once per process** — stored in module-level dict `_CASES_CE` / `_STATUTES_CE` | `Retriever.py:82–86`: `_ensure_cases_ce()` checks if already loaded before calling `CrossEncoder(...)`. Equivalent in `RulesRetriever.py:76–80`. |

### RapidOCR Models (Local CPU)

RapidOCR model files are bundled **inside the Python package** in the venv (not in the HuggingFace cache). Location: `backend/venv/lib/python3.13/site-packages/rapidocr/models/`.

| Model file | Size | Purpose |
|---|---|---|
| `PP-OCRv6_det_small.onnx` | 9.5 MB | Text region detection (find where text is in the image) |
| `PP-OCRv6_rec_small.onnx` | 21 MB | Text recognition (read the detected text) |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | 572 KB | Text orientation classifier |

Source: `ls -lh venv/lib/python3.13/site-packages/rapidocr/models/`.

**Load timing:** `SCCVision/engine.py` creates a `RapidOCR()` singleton once at startup via `warmup()` which is called from `app.py:1899` in a daemon thread. The engine is **available from first request** because warmup completes during the Flask startup sequence. `sccvision/health` returns `"loaded": true` once the warmup completes.

Models are ONNX format, run via `onnxruntime==1.23.2` on CPU, using the CPU execution provider.

---

## 5.2.2.7 External Services

| Service | Endpoint called | Purpose | Auth method | Identifier |
|---|---|---|---|---|
| Groq API | `https://api.groq.com/openai/v1/chat/completions` | LLM inference (all natural language generation) | Bearer token (`Authorization: Bearer <key>`) | 17 keys configured (`GROQ_API_KEY`, `GROQ_API_KEY_1` … `GROQ_API_KEY_15`) |
| Firebase Authentication | `https://identitytoolkit.googleapis.com/...` (via firebase-admin SDK) | Server-side token verification; `fb_auth.verify_id_token()` | Firebase Admin SDK private key | Firebase project: `legalai-scc` |
| Firestore | Google Cloud Firestore (via firebase-admin SDK) | User profiles, chat history (conversations + turns), CaseLaws metadata | Firebase Admin SDK private key | Same project: `legalai-scc` |
| Firebase Storage | `legalai-scc.firebasestorage.app` | Storage bucket configured in `auth.py:31`; no Storage read/write calls found in application code | Firebase Admin SDK | `legalai-scc.firebasestorage.app` |

There are **no other outbound network calls** at runtime. The HuggingFace Hub is accessed only once at model download time (ingestion/setup), not during serving.

---

## 5.2.2.8 Configuration: Environment Variables

### Backend environment variables

All are read from `backend/.env` (loaded via `python-dotenv` in `Qclassifier.py` and `Retriever.py`; `auth.py` reads directly from `os.environ`).

| Variable name | Read by | Default if absent | Required? |
|---|---|---|---|
| `GROQ_API_KEY` | `Qclassifier.py:14`, `groq_client.py:37`, `Retriever.py:39`, `SCCVision/analyst/client.py:34` | None | Yes — Qclassifier raises `ValueError` if missing |
| `GROQ_API_KEY_1` … `GROQ_API_KEY_15` | `groq_client.py:37` (reads all env vars starting with `GROQ_API_KEY`) | None | No — additional rotation keys; more keys = less rate limiting |
| `GROQ_API_KEY_12` | `SCCVision/analyst/client.py:34` (preferred key for SCCVision) | None — falls back to `call_groq`'s standard rotation | No |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | `auth.py:20` | `""` (empty) — `init_firebase_admin()` will raise if empty | Yes |
| `CORS_ORIGIN` | `app.py:38` | `"http://localhost:3000"` | No — default is correct for local development |
| `FIRESTORE_EMULATOR_HOST` | `backend/tests/test_history.py:31` | None | No — only needed for integration tests against the emulator |

**Variables confirmed in `backend/.env` at audit time** (names only):  
`GROQ_API_KEY`, `GROQ_API_KEY_1` through `GROQ_API_KEY_8`, `GROQ_API_KEY_10` through `GROQ_API_KEY_15`, `FIREBASE_SERVICE_ACCOUNT_PATH`, `CORS_ORIGIN`.

Source: `grep -oP "^[A-Z_0-9]+(?==)" backend/.env`.

**Root `.env.example`** (at repo root) contains only: `HF_API_TOKEN`. This is outdated — it describes the original Hugging Face prototype pipeline and does not list any of the current required variables.

### Frontend environment variables

All are read from `frontend/.env.local` (Next.js standard location).

| Variable name | Read by | Default if absent | Required? |
|---|---|---|---|
| `NEXT_PUBLIC_FIREBASE_API_KEY` | `frontend/app/lib/firebase.ts:6` | `undefined` | Yes — Firebase client SDK fails to initialise |
| `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN` | `frontend/app/lib/firebase.ts:7` | `undefined` | Yes |
| `NEXT_PUBLIC_FIREBASE_PROJECT_ID` | `frontend/app/lib/firebase.ts:8` | `undefined` | Yes |
| `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET` | `frontend/app/lib/firebase.ts:9` | `undefined` | Yes |
| `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID` | `frontend/app/lib/firebase.ts:10` | `undefined` | Yes |
| `NEXT_PUBLIC_FIREBASE_APP_ID` | `frontend/app/lib/firebase.ts:11` | `undefined` | Yes |
| `NEXT_PUBLIC_API_URL` | `frontend/app/components/ChatView.tsx:50`, `frontend/app/lib/api.ts:3`, `frontend/app/repository/caselaws/page.tsx:84` | `"http://localhost:5000"` | No — defaults to localhost:5000 |

**`frontend/.env.example`** lists all 7 variables above (names only, values empty).  
Source: `cat frontend/.env.example`.

**Variables confirmed in `frontend/.env.local` at audit time** (names only):  
`NEXT_PUBLIC_FIREBASE_API_KEY`, `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN`, `NEXT_PUBLIC_FIREBASE_PROJECT_ID`, `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET`, `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID`, `NEXT_PUBLIC_FIREBASE_APP_ID`, `NEXT_PUBLIC_API_URL`.  
Source: `grep -oP "^[A-Z_0-9]+(?==)" frontend/.env.local`.

---

## 5.2.2.9 Startup Settings

### Flask backend (`app.py:1920`)

```python
app.run(debug=True, port=5000, host='0.0.0.0')
```

| Setting | Value | Source |
|---|---|---|
| Host | `0.0.0.0` (all interfaces) | `app.py:1920` |
| Port | 5000 | `app.py:1920` |
| Debug mode | True (auto-reload on code change; stack traces in responses) | `app.py:1920` |
| CORS allowed origin | `http://localhost:3000` (default) | `app.py:38–39` |
| WSGI server | Werkzeug development server | No gunicorn/uWSGI configured |

**Flask startup sequence** (measured via timed module import):

| Phase | Time | Notes |
|---|---|---|
| dotenv + stdlib imports | 0.045 s | — |
| Flask import | 0.229 s | — |
| firebase-admin import + `init_firebase_admin()` | 0.763 s | Reads service-account JSON, initialises Firestore client |
| groq_client import | 0.764 s | Trivial after firebase |
| Qclassifier import (loads sentence-transformers + transformers) | 7.74 s | Heavy — sentence-transformers imports PyTorch |
| Retriever, RulesRetriever imports | 7.74 s | Shared torch already loaded |
| SCCVision warmup (RapidOCR ONNX models) | 7.78 s | Runs in daemon thread; sccvision/health shows `"loaded": true` once complete |
| **Total until Flask binds port** | ~9–10 s | Full `app.py` module import measured at 9.01 s |

Source: timed Python import sequence (this audit session).

### Next.js frontend

**npm scripts** (`frontend/package.json`):

| Script | Command |
|---|---|
| `dev` | `next dev` |
| `build` | `next build` |
| `start` | `next start` |
| `lint` | `next lint` |

**Production build** (`npm run build`): completed in **28 seconds** on this machine. Used Turbopack (Next.js 16 default). Output: 20 routes — 19 static (○), 1 dynamic (ƒ: `/chat/[conversationId]`). Zero TypeScript errors, zero route errors.

**Build warnings**: only one type, repeated 8 times:  
`[baseline-browser-mapping] The data in this module is over two months old. To ensure accurate Baseline data, please update: npm i baseline-browser-mapping@latest -D`  
This is a cosmetic informational warning from the `baseline-browser-mapping` package used by Next.js for browser compatibility tables. It does not affect functionality.

Source: `/tmp/nextjs_build.log`.

**Next.js dev server**: already running on port 3000 (started before this audit session). `curl http://localhost:3000` returns HTTP 200 in 0.12 s.

---

## 5.2.2.10 Retrieval and Pipeline Configuration

| Parameter | Value | Location |
|---|---|---|
| BM25 candidates per query | top 50 by score | `Retriever.py:678–681` |
| Vector search candidates per query | top 50 (or collection size if smaller) | `Retriever.py:682–688` |
| RRF fusion k parameter | 60 | `Retriever.py:694` |
| RRF output (before cross-encoder) | top 30 | `Retriever.py:694` |
| Cross-encoder output (after reranking) | grouped → `TOP_N_PARENTS` | `Retriever.py:756` |
| `TOP_N_PARENTS` (returned to generate endpoint) | 3 | `Retriever.py:47` |
| `LOW_CONFIDENCE_THRESHOLD` (CE sigmoid score) | 0.4 | `Retriever.py:53` |
| Same parameters for RulesRetriever | identical (BM25 top 50, vector top 50, RRF k=60, top 30) | `RulesRetriever.py:582` |
| Constitution chunk size | 1000 characters | `constitutionlogic/constitutionbreaker.py:23` |
| Constitution chunk overlap | 300 characters | `constitutionlogic/constitutionbreaker.py:24` |
| Constitution minimum chunk | 150 characters | `constitutionlogic/constitutionbreaker.py:25` |
| Statute chunk max tokens (tokenizer-aware) | 480 tokens | `backend/build_statute_chunks.py:28` |
| Statute chunk overlap | 50 tokens | `backend/build_statute_chunks.py:29` |
| Statute chunk stride | 430 tokens | `backend/build_statute_chunks.py:30` |
| Case chunk max characters | 3000 characters | `backend/build_case_chunks.py` → `MAX_CHARS_PER_CHUNK` |
| Case chunk overlap | 1 paragraph | `backend/build_case_chunks.py` → `OVERLAP_PARAGRAPHS` |
| SCCVision max upload size | 10 MB | `SCCVision/router.py:26` |
| SCCVision allowed file types | `png`, `jpg`, `jpeg`, `webp` | `SCCVision/router.py:25` |
| Firestore turn size guard | 900 KB | `history_store.py` → `_size_guard` |
| Groq retry rounds | 2 | `groq_client.py:83` |
| Groq inter-round wait | 30 seconds | `groq_client.py:132` |
| SCCVision OCR thread pool workers | 2 | `SCCVision/router.py:30` |

---

## 5.2.2.11 Firebase and Emulator Configuration

| File | Location |
|---|---|
| `firebase.json` | Repo root |
| `firestore.rules` | Repo root |
| `firestore.indexes.json` | Repo root |

**Emulator ports** (from `firebase.json`):

| Service | Port |
|---|---|
| Firestore emulator | 8080 |
| Emulator UI | 4000 |

Source: `cat firebase.json`.

**Firebase Admin SDK initialisation** (`auth.py:31`):  
Project: `legalai-scc` (inferred from `storageBucket: 'legalai-scc.firebasestorage.app'`).  
Service account file: path read from `FIREBASE_SERVICE_ACCOUNT_PATH` env var.

---

## 5.2.3 Process Measurements at Runtime (Step 4)

Both the Flask backend and the Next.js dev server were **already running** when this audit began. The `/api/health` and `/sccvision/health` endpoints were confirmed live.

### Health Check Responses

```
$ curl -s -w "\nHTTP %{http_code} in %{time_total}s\n" http://localhost:5000/api/health
{
  "status": "ok"
}
HTTP 200 in 0.001622s

$ curl -s -w "\nHTTP %{http_code} in %{time_total}s\n" http://localhost:5000/sccvision/health
{
  "engine": "rapidocr",
  "loaded": true,
  "module": "SCCVision",
  "status": "ok"
}
HTTP 200 in 0.001904s
```

Source: `curl` commands run during this audit session.

### Process Memory at Idle

| Process | PID | RSS | VSZ | %CPU | Elapsed |
|---|---|---|---|---|---|
| `venv/bin/python app.py` (main Flask process) | 140733 | **1,074 MB** | 9,343 MB | 5.3% | ~1h 37m |
| `python app.py` (auxiliary process, see note) | 140669 | 452 MB | 7,758 MB | 0.2% | ~1h 37m |
| `next-server (v16.0.3)` | 140380 | **800 MB** | ~35,512 MB | 0.5% | ~1h 39m |

Source: `ps -o pid,rss,vsz,%cpu,etime,cmd -p <PID>`.

**Note on two Flask processes:** Both `140669` and `140733` were running. Process `140669` appears to be a shell wrapper process (`python app.py` without venv prefix), while `140733` is the full venv-activated process. Both were started at the same time and correspond to the same Flask app run from different terminal sessions.

**Virtual memory (VSZ) is misleadingly large** for all three processes due to memory-mapped torch model files, shared Python runtime libraries, and ONNX model mappings. RSS (Resident Set Size) is the physically held memory — 1,074 MB for the Flask backend reflects the loaded PyTorch runtime, sentence-transformers models, Firebase Admin SDK, and all ChromaDB clients being held in memory.

### Startup Log (New Flask Attempt)

A fresh start attempt was made on the already-occupied port 5000. The startup sequence printed before the port-conflict error was:

```
[INFO] Loaded 3 Groq API key(s)
Firebase Admin initialized from /home/gift/.../backend/legalai-scc-firebase-adminsdk-fbsvc-73bde27853.json
 * Serving Flask app 'app'
 * Debug mode: on
Address already in use
Port 5000 is in use by another program.
```

This confirms: (a) Groq key loading and Firebase Admin initialisation are the first steps after import; (b) the app reaches readiness within ~9–10 s of starting; (c) no warning messages appear in the startup sequence before the Flask bind.

Source: `/tmp/flask_startup.log`.

### Manual Steps to Run an End-to-End Test

**Step A — Advocate RAG query:**
1. Open `http://localhost:3000` in a browser.
2. Sign in with an account that has role `advocate` or `admin` (Firebase Auth).
3. Navigate to `/research`.
4. Type a legal question (e.g., "What is the monetary jurisdiction of the Small Claims Court?") and press Enter.
5. **Numbers to record:**
   - Time from pressing Enter to answer displayed (browser stopwatch or DevTools Network tab)
   - `latency_ms` stored in Firestore: Firebase Console → Firestore → `users/<uid>/conversations/<cid>/turns/<tid>` → `latency_ms` field
   - Backend RSS during query: in a terminal run `ps -o rss -p 140733` before and during the LLM call

**Step B — SCC evidence analysis:**
1. Sign in at `http://localhost:3000/litigant/sign-in` with a Google account (creates litigant role).
2. Navigate to `/litigant/evidence`.
3. Upload a WhatsApp screenshot (PNG/JPG, max 10 MB).
4. Navigate to `/litigant/analysis` and trigger the admissibility analysis.
5. **Numbers to record:**
   - Time from upload to OCR result (Chrome DevTools → Network → `sccvision/extract` response time)
   - Time from submit to analysis result (`sccvision/case/analyze` response time)
   - Backend RSS during OCR: `ps -o rss -p 140733`

---

## 5.2.4 Reproducibility, Recovery and Rollback

### 5.2.4.1 Setup Sequence from a Fresh Clone

The following is the **exact ordered setup sequence** reconstructed from code, scripts, and data directory structure. No README documents this sequence for the current stack.

```
# ── 0. Prerequisites ──────────────────────────────────────────────────────────
# Install system packages (Debian/Ubuntu/Kali):
sudo apt-get install python3.13 python3.13-venv tesseract-ocr git nodejs npm
# Java 21+ for Firebase emulator (optional, only for rule tests):
sudo apt-get install default-jdk
# Install Firebase CLI:
npm install -g firebase-tools

# ── 1. Clone ──────────────────────────────────────────────────────────────────
git clone git@github.com:Houstonsboy/Retreival-Augmented-Generation-AI.git
cd Retreival-Augmented-Generation-AI

# ── 2. Backend Python environment ─────────────────────────────────────────────
cd backend
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Note: torch 2.9.1 is large (~2 GB download). Expect 10–20 min on a slow connection.
cd ..

# ── 3. Backend environment variables ──────────────────────────────────────────
# Create backend/.env with the following names (do NOT commit this file):
#   GROQ_API_KEY=<your Groq API key>
#   GROQ_API_KEY_2=<second key, optional>
#   ...
#   GROQ_API_KEY_15=<additional keys, optional>
#   FIREBASE_SERVICE_ACCOUNT_PATH=legalai-scc-firebase-adminsdk-fbsvc-73bde27853.json
#   CORS_ORIGIN=http://localhost:3000
# Download the Firebase Admin SDK service account JSON from:
#   Firebase Console → Project Settings → Service Accounts → Generate new private key
# Place it at backend/<FIREBASE_SERVICE_ACCOUNT_PATH>

# ── 4. Statute ingestion pipeline (5 steps) ────────────────────────────────────
# Step 1: Download statute HTML files from Kenya Law (no API key needed)
python download_statutes.py
# Output: data/raw_statutes/<slug>.html (14 files) + data/raw_statutes/manifest.json

# Step 2: Parse HTML into JSONL (BeautifulSoup, no API key needed)
python parse_statutes.py
# Input:  data/raw_statutes/*.html
# Output: data/processed/statutes/sections.jsonl, structure.jsonl, etc.

# Step 3: Chunk sections into embedding-ready JSONL (tokenizer-aware, no API key)
cd backend && python build_statute_chunks.py
# Input:  data/processed/statutes/sections.jsonl
# Output: data/processed/statutes/sections_embed.jsonl

# Step 4: Embed into StatutesDB (sentence-transformers, no API key)
python embed_knowledge.py --source statutes
# Input:  data/processed/statutes/sections_embed.jsonl
# Output: backend/StatutesDB/ (ChromaDB PersistentClient, ~20 MB)

# ── 5. Case law ingestion pipeline ────────────────────────────────────────────
# Step 1: Create cases_meta.json — manually or via the FIRAC API endpoint
# Format: data/processed/cases/cases_meta.json (array of case objects with
#         case_identifier, parties, court_level, judge, year, legal_domain, etc.)

# Step 2: Chunk case PDFs + build bbox index (PyMuPDF, no API key)
cd backend && python build_case_chunks.py
# Input:  backend/SCCCases/*.pdf + data/processed/cases/cases_meta.json
# Output: data/processed/cases/cases_embed.jsonl + backend/locations.db

# Step 3: Embed into CaseLawDB (sentence-transformers, no API key)
python embed_knowledge.py --source cases
# Input:  data/processed/cases/cases_embed.jsonl
# Output: backend/CaseLawDB/ (ChromaDB PersistentClient, ~4.7 MB)

# ── 6. Constitution ingestion ─────────────────────────────────────────────────
# Method A: via admin API (requires running backend + admin token)
# POST http://localhost:5000/api/digest/constitution
# Method B: via constitutionbreaker script (TODO: verify exact script name)
# Input:  backend/MasterRules/TheConstitutionOfKenya.pdf
# Output: backend/constitutionlogic/MasterRulesDB/

# ── 7. Firebase setup ─────────────────────────────────────────────────────────
firebase login
firebase use legalai-scc
# Deploy Firestore rules and indexes:
firebase deploy --only firestore:rules,firestore:indexes

# ── 8. Admin account bootstrap ────────────────────────────────────────────────
# (from repo root with backend venv active)
cd backend
python scripts/bootstrap_admin.py <email> <password> <username>
# Or to change role of an existing account:
python scripts/set_role.py <uid> admin

# ── 9. Run backend ────────────────────────────────────────────────────────────
cd backend
source venv/bin/activate
python app.py
# Flask starts on http://0.0.0.0:5000
# Expect ~9–10 s startup time (PyTorch import heavy)

# ── 10. Frontend setup ────────────────────────────────────────────────────────
cd frontend
npm install
cp .env.example .env.local
# Edit .env.local: fill in NEXT_PUBLIC_FIREBASE_* from Firebase Console
#   Project Settings → Your apps → Web app → SDK configuration
# NEXT_PUBLIC_API_URL is already http://localhost:5000 by default

npm run dev    # development (HMR, faster startup)
# OR:
npm run build && npm start   # production build (28s build, then static serving)
```

**Ingestion scripts that call the Groq API:**

| Script | Groq call? | Purpose |
|---|---|---|
| `download_statutes.py` | No | Downloads HTML from Kenya Law |
| `parse_statutes.py` | No | Parses HTML to JSONL |
| `backend/build_statute_chunks.py` | No | Tokenizer-aware chunking |
| `backend/build_case_chunks.py` | No | PDF → chunks + bbox; pure PyMuPDF |
| `backend/embed_knowledge.py` | No | Embeds JSONL into ChromaDB |
| `backend/CaseLaw/firac.py` | **Yes** | LLM-based FIRAC extraction from PDF text (called per-case by `build_case_chunks.py` if `--firac` flag is passed, or via `/api/digest` API) |
| `backend/constitutionlogic/constitution.py` | **Yes** | Used in `constitutionbreaker.py` to call Groq for content analysis |
| `backend/CaseLaw/ingester.py` | No | Stores pre-computed FIRAC chunks in ChromaDB |

---

### 5.2.4.2 Artefacts Needed to Run Without Re-ingesting

| Artefact | Description | Git status | Size |
|---|---|---|---|
| `backend/CaseLawDB/` | ChromaDB case-law vector store | **Untracked, NOT gitignored** | 4.7 MB |
| `backend/StatutesDB/` | ChromaDB statutes vector store | **Untracked, NOT gitignored** | 20 MB |
| `backend/constitutionlogic/MasterRulesDB/` | ChromaDB master-rules (Constitution + Evidence Act) | **Tracked in git** (5 files in `git ls-files`) | 8.3 MB |
| `backend/locations.db` | SQLite bbox index for PDF viewer | **Untracked, NOT gitignored** | 280 KB |
| `~/.cache/huggingface/hub/` | HuggingFace model weights | **Not in repo** (machine-local) | 5.6 GB |
| `backend/.env` | Backend secrets | **Gitignored** | — |
| `frontend/.env.local` | Frontend Firebase config | **Gitignored** | — |
| Firebase service account JSON | Backend auth credential | **Gitignored** | — |

**Key gap:** `CaseLawDB/` and `StatutesDB/` are **neither tracked in git nor gitignored**. They are untracked (`??` in `git status`). This means they will not appear in a fresh clone and must be recreated by running the ingestion pipeline. The `.gitignore` does not explicitly exclude them either, so an accidental `git add .` could commit binary ChromaDB files.

**`MasterRulesDB/` is partially tracked:** `git ls-files` shows 5 binary files inside it are committed. This is inconsistent with the untracked status of `CaseLawDB/` and `StatutesDB/`.

**Backup options:**

| Method | What it covers | How |
|---|---|---|
| Copy ChromaDB directories | Full vector store (embeddings + metadata) | `cp -r backend/CaseLawDB backend/CaseLawDB.bak` (the `ChunkDB_backup_20261031/` and `LegalSummariesDB.bak/` directories in the repo show this pattern was already used) |
| ChromaDB export to JSON | Human-readable chunk export | `python -c "import chromadb; c = chromadb.PersistentClient('backend/CaseLawDB'); col = c.get_collection('case_law'); data = col.get(include=['documents','metadatas']); import json; json.dump(data, open('caselaw_export.json','w'))"` |
| `rsync` to external drive | Full backup including venv | `rsync -av --exclude='.git' . /mnt/external/rag-backup/` |

Backup directories already present in the repo:
- `backend/ChunkDB_backup_20261031/` — backup of the legacy `ChunkDB` before migration
- `backend/LegalSummariesDB.bak/` — backup of old `LegalSummariesDB`
- `backend/constitutionlogic/MasterRulesDB_backup_20261031/` — backup of MasterRulesDB
- `backend/ChunkDB_backup_20261031/` for the old embedding store

### 5.2.4.3 Existing Export Files

The following JSON exports exist in the repo (untracked):

| File | Contents |
|---|---|
| `backend/master_rules_db_export_20260202_153301.json` | ChromaDB master_rules export, 2026-02-02 |
| `backend/master_rules_db_export_20260130_132247.json` | ChromaDB master_rules export, 2026-01-30 |
| `backend/Database/master_rules_db_export_20260205_094943.json` | ChromaDB master_rules export, 2026-02-05 |
| `backend/Database/legal_summaries_db_export_20260205_094943.json` | ChromaDB legal_summaries (legacy) export, 2026-02-05 |
| `backend/constitutionlogic/master_rules_db_export_20260130_133016.json` | ChromaDB master_rules export, 2026-01-30 |
| `backend/Const_IntepretationLaws/blacklaw_export_for_neo4j.json` | BlackLaw Dictionary embeddings in Neo4j import format (unused — Neo4j not active) |

These exports are point-in-time snapshots of the vector store content (document texts + metadata). They do not include the embedding vectors themselves (too large), so they cannot be used to restore a ChromaDB without re-embedding.

### 5.2.4.4 Rollback Options

| Scenario | Rollback approach |
|---|---|
| **Bad code change** | `git revert <commit>` or `git checkout <prev-hash> -- <file>`. No deployment pipeline means rollback = code change only. |
| **Corrupted ChromaDB** | Delete the directory and re-run `embed_knowledge.py`. Full re-embed of statutes (~30 min on this hardware, estimate) or cases. The JSON exports (§4.3) can verify chunk content but cannot restore embeddings. |
| **Bad Firestore rules deploy** | `git show HEAD~1:firestore.rules > firestore.rules && firebase deploy --only firestore:rules` |
| **Corrupted locations.db** | Delete and re-run `python backend/build_case_chunks.py` |
| **Wrong admin role set** | `python backend/scripts/set_role.py <uid> advocate` to downgrade |
| **Frontend regression** | Re-run `npm run build` after reverting; the previous `.next/` build artefacts in the working tree serve as an implicit cache |

---

*End of ENV_AUDIT.md*
