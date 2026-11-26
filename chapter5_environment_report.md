# Chapter 5.2 — Implementation Environment Report

**Project:** GraphRAG Legal Decision-Support System (Kenya Small Claims Court)
**Report generated:** 2026-11-02
**Collector:** automated read-only audit (no services started, no files modified)

---

## 1. Summary Table

| Component | Version / Value | Source |
|---|---|---|
| **Operating System** | Kali GNU/Linux Rolling 2025.2 | `/etc/os-release` |
| **Kernel** | 6.12.38+kali-amd64 | `uname -r` |
| **CPU** | Intel Core i7-8665U @ 1.90 GHz, 4 cores / 8 threads | `lscpu` |
| **RAM** | 15 GiB total, 5.4 GiB available | `free -h` |
| **Disk (project partition)** | 452 G total, 92 G free (79% used), `/dev/nvme0n1p2` on `/` | `df -h .` |
| **GPU** | Intel UHD Graphics 620 (integrated); no discrete/NVIDIA GPU | `lspci` |
| **Python** | 3.13.7 (venv at `backend/venv/`) | `python --version` inside venv |
| **Flask** | 3.1.2 | `pip freeze` |
| **flask-cors** | 6.0.1 | `pip freeze` |
| **LangChain** | 1.0.7 (langchain-core 1.0.6, langchain-text-splitters 1.0.0) | `pip freeze` |
| **groq (SDK)** | 0.35.0 | `pip freeze` |
| **LLM model** | `qwen/qwen3.8-27b` via Groq API | `backend/groq_client.py:20` |
| **ChromaDB** | 1.3.5, embedded PersistentClient | `pip freeze`, source |
| **sentence-transformers** | 5.1.2 | `pip freeze` |
| **Embedding model** | `intfloat/e5-base-v2` (768-dim, local CPU inference) | `backend/config.py`, `pixe.py:31` |
| **PyTorch** | 2.9.1 (CPU only) | `pip freeze` |
| **pytesseract** | 0.3.13 | `pip freeze` |
| **Tesseract OCR** | 5.5.0, languages: `eng`, `osd` | `tesseract --version` |
| **PyMuPDF (fitz)** | 1.26.6 | `pip freeze` |
| **firebase-admin** | 7.7.0 | `pip freeze` |
| **google-cloud-firestore** | 2.34.1 | `pip freeze` |
| **Node.js** | v20.19.2 | `node --version` |
| **npm** | 9.2.0 | `npm --version` |
| **Next.js** | 16.0.3 | `frontend/package.json` |
| **React** | 19.2.0 | `frontend/package.json` |
| **Neo4j** | 5.18.0 Community, locally installed binary | `neo4j-community-5.18.0/packaging_info` |
| **Neo4j connection** | `neo4j://localhost:7687` (Bolt), HTTP browser: `:7474` | `neo4j-java/Neo4jWriter.java:120`, `START_NEO4J.sh` |
| **PostgreSQL (psql binary)** | 17.5 installed | `psql --version` |
| **PostgreSQL (runtime use)** | **NOT USED** by the application | source audit |
| **Docker** | 28.5.2 installed; no compose; no containers | `docker --version`; `docker ps` |
| **Git remote** | `git@github.com:Houstonsboy/Retreival-Augmented-Generation-AI.git` | `git remote -v` |
| **Total commits** | 56 | `git rev-list --count HEAD` |

---

## 2. Hardware and Operating System

**Source:** `cat /etc/os-release`, `uname -r`, `lscpu`, `free -h`, `df -h .`, `lspci`

- **OS:** Kali GNU/Linux Rolling, version 2025.2 (`kali-rolling`), Debian-based
- **Kernel:** `6.12.38+kali-amd64`
- **CPU:** Intel® Core™ i7-8665U @ 1.90 GHz
  - 4 physical cores, 2 threads per core → 8 logical CPUs
  - NUMA node 0: CPUs 0–7
- **RAM:** 15 GiB total | 10 GiB used | 348 MiB free | 6.8 GiB buff/cache | **5.4 GiB available**
  - Swap: 15 GiB total, 20 MiB used
- **Storage:** `/dev/nvme0n1p2` mounted at `/`, 452 G total, **92 G free** (79% used)
  - Project directory resides on this partition.
- **GPU:**
  - `00:02.0 VGA compatible controller: Intel Corporation WhiskeyLake-U GT2 [UHD Graphics 620] (rev 02)`
  - No NVIDIA GPU present; `nvidia-smi` not found.
  - **The project does NOT use a GPU.** PyTorch (2.9.1) is installed for CPU-based sentence-transformers inference only. No `cuda`, `torch.device("cuda")`, or GPU flags appear anywhere in the source.

---

## 3. Project Structure

**Source:** `find` with exclusions for `node_modules`, `.git`, `venv`, `__pycache__`, `.next`, data folders.

```
Retreival-Augmented-Generation-AI/
├── backend/                        ← Flask API (main backend)
│   ├── app.py                      ← Flask entry point, all routes
│   ├── config.py                   ← Central config: DB paths, embedding model name
│   ├── groq_client.py              ← Shared Groq API client with key rotation
│   ├── pixe.py                     ← Main RAG pipeline
│   ├── Retriever.py                ← Case-law hybrid retriever (dense + BM25 + cross-encoder)
│   ├── RulesRetriever.py           ← Constitution/statutes hybrid retriever
│   ├── Qclassifier.py              ← Query classifier (routes to mode A/B)
│   ├── auth.py                     ← Firebase Admin init, role-based decorator
│   ├── digester.py                 ← Repo digest helper
│   ├── history_store.py            ← Firestore chat history persistence
│   ├── history_routes.py           ← Blueprint: /api/history/* endpoints
│   ├── agentrouter.py              ← EXPERIMENTAL: HuggingFace router (standalone, not integrated)
│   ├── citation_verifier.py        ← Citation validation utility
│   ├── coverage.py                 ← Coverage reporting
│   ├── embed_knowledge.py          ← Ingestion: embed chunks → ChromaDB (statutes & cases)
│   ├── build_case_chunks.py        ← Ingestion: parse case PDFs → chunk JSONs + locations.db
│   ├── build_statute_chunks.py     ← Ingestion: parse statute PDFs → chunk JSONs
│   ├── export_blacklaw_to_json.py  ← Export Black Law Dictionary from ChromaDB → JSON for Neo4j
│   ├── compute_constitution_bboxes.py ← Compute PDF bounding boxes → locations.db
│   ├── validate_knowledge.py       ← Post-ingestion validation
│   ├── CaseLaw/
│   │   ├── firac.py                ← FIRAC extraction engine (uses Groq)
│   │   └── ingester.py             ← Ingest FIRAC data → LegalSummariesDB ChromaDB
│   ├── constitutionlogic/
│   │   ├── constitution.py         ← Constitution analysis (uses Groq)
│   │   ├── constitutionbreaker.py  ← Parse constitution articles → chunks + locations.db
│   │   └── constitution_digester.py ← ConstitutionDigester class
│   ├── Const_IntepretationLaws/    ← Black Law Dictionary source + ChromaDB
│   ├── scripts/
│   │   ├── bootstrap_admin.py      ← Create admin user in Firestore
│   │   ├── set_role.py             ← Assign user role in Firestore
│   │   └── seed_caselaws.py        ← Seed case law data
│   ├── tests/
│   │   └── test_history.py         ← pytest tests for history_store
│   ├── requirements.txt
│   ├── .env                        ← Present (gitignored)
│   └── venv/                       ← Python virtual environment
├── frontend/                       ← Next.js 16 / React 19 app
│   ├── app/                        ← Next.js App Router pages
│   ├── package.json
│   ├── .env.local                  ← Present (gitignored)
│   └── next.config.ts
├── neo4j-community-5.18.0/        ← Neo4j binary installation (local)
├── neo4j-java/                     ← Java subproject (Maven): builds Neo4j graph from JSON
│   ├── pom.xml
│   └── src/main/java/com/legal/rag/neo4j/
│       ├── DictionaryKnowledgeGraph.java
│       └── Neo4jWriter.java
├── START_NEO4J.sh                  ← Shell script: start Neo4j
├── .env.example                    ← Root-level example env
├── .env                            ← Root-level env (gitignored)
├── .gitignore
├── README.md
└── RAG1.ipynb                      ← Early Jupyter notebook prototype
```

**Key identifiers:**
- **Backend folder:** `backend/`
- **Frontend folder:** `frontend/`
- **Ingestion/seed scripts:** `backend/embed_knowledge.py`, `backend/build_case_chunks.py`, `backend/build_statute_chunks.py`, `backend/CaseLaw/ingester.py`, `backend/export_blacklaw_to_json.py`, `backend/scripts/seed_caselaws.py`
- **Evaluation/test scripts:** `backend/tests/test_history.py`, `backend/validate_knowledge.py`, `backend/coverage.py`
- **README:** `README.md` (root)
- **Dockerfile:** NOT FOUND
- **docker-compose:** NOT FOUND
- **Makefile:** NOT FOUND
- **Shell scripts:** `START_NEO4J.sh`, `neo4j-java/run_dictionary_graph.sh`
- **`.env.example`:** root and `frontend/.env.example`
- **Requirements:** `backend/requirements.txt`
- **pyproject.toml:** NOT FOUND

---

## 4. Backend Packages (Key, with Exact Versions)

**Source:** `pip freeze` inside `backend/venv/`, confirmed against `backend/requirements.txt` and source imports.

### Virtual Environment
- Path: `backend/venv/`
- Python: **3.13.7**
- Activated with: `cd backend && source venv/bin/activate`

### Key Installed Packages

| Package | Version | Purpose |
|---|---|---|
| Flask | 3.1.2 | Web framework / API server |
| flask-cors | 6.0.1 | CORS headers for frontend |
| langchain | 1.0.7 | Document processing utilities |
| langchain-core | 1.0.6 | LangChain core |
| langchain-text-splitters | 1.0.0 | Text chunking |
| groq | 0.35.0 | Groq SDK (imported in `pixe.py`; main calls use `requests` directly) |
| chromadb | 1.3.5 | Vector store (embedded mode) |
| sentence-transformers | 5.1.2 | Embedding + CrossEncoder reranking |
| torch | 2.9.1 | Backend for sentence-transformers (CPU only) |
| transformers | 4.57.1 | HuggingFace transformers |
| huggingface-hub | 0.36.0 | Model download from HuggingFace Hub |
| pytesseract | 0.3.13 | OCR wrapper |
| Pillow | 12.0.0 | Image handling |
| PyMuPDF | 1.26.6 | PDF coordinate extraction (`fitz`) |
| pypdf | 6.3.0 | PDF text extraction |
| pdfminer.six | 20260107 | PDF mining |
| ocrmypdf | 17.13.0 | Scanned PDF pre-processing |
| rank-bm25 | 0.2.2 | BM25Okapi for hybrid retrieval |
| rapidfuzz | 3.14.6 | Fuzzy string matching |
| scikit-learn | 1.7.2 | Cosine similarity, TF-IDF |
| numpy | 2.3.5 | Numerical arrays |
| pydantic | 2.14.2 | Data validation |
| python-dotenv | 1.2.1 | `.env` loading |
| requests | 2.34.2 | HTTP calls to Groq API |
| firebase-admin | 7.7.0 | Firebase Admin SDK |
| google-cloud-firestore | 2.34.1 | Firestore client |
| google-cloud-storage | 3.17.1 | Cloud Storage client |
| PyJWT | 2.15.1 | JWT decoding for auth |
| openai | 2.9.0 | Installed; **not used** in main app (Groq called via raw `requests`) |
| anthropic | 0.75.0 | Installed; **not used** in main app |
| fastapi | 0.124.2 | Installed; **not used** in main app (Flask is the server) |
| uvicorn | 0.38.0 | Installed; **not used** in main app |
| fpdf2 | 2.9.0 | PDF generation |

### Import Frequency Analysis (top 30 from source)

(Source: `grep -rhoE "^(from|import)..."` on `backend/`, excluding `venv/`)

```
32  pathlib     22  json        16  typing      16  re
15  os          15  chromadb    11  time        10  firebase_admin
10  dotenv       9  sentence_transformers   7  requests
 6  fitz         5  pytesseract  5  PIL         3  flask
 2  langchain_core.documents    2  groq         2  rank_bm25
 2  sklearn      1  rapidfuzz   1  groq_client  1  pypdf
```

### Cross-check: Imports vs Requirements

**Imported but NOT in requirements.txt:** `firebase_admin`, `google.cloud.firestore`, `PyJWT`, `fitz (PyMuPDF)`, `rapidfuzz`, `openai`, `rank_bm25` (it IS listed, confirmed installed). Notable omissions that should be added for full reproducibility: `firebase-admin`, `PyMuPDF`, `rapidfuzz`, `PyJWT`.

**In requirements.txt but never imported by main app:** `openai`, `fastapi`, `uvicorn`, `anthropic` (all installed but unused in the serving path).

### Tesseract OCR (system binary)

- **Version:** 5.5.0 (leptonica 1.84.1)
- **Language packs installed:** `eng` (English), `osd` (orientation/script detection)
- **Note:** Only English language data is present; OCR is limited to English-language documents.

---

## 5. Frontend Packages

**Source:** `frontend/package.json`, `node --version`, `npm --version`

- **Node.js:** v20.19.2
- **npm:** 9.2.0
- **Package manager:** npm (confirmed by `package-lock.json`)

### package.json

```json
{
  "name": "frontend",
  "version": "0.1.0",
  "scripts": {
    "dev":   "next dev",
    "build": "next build",
    "start": "next start",
    "lint":  "eslint"
  }
}
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| next | 16.0.3 | React framework / SSR |
| react | 19.2.0 | UI library |
| react-dom | 19.2.0 | React DOM renderer |
| firebase | ^10.12.0 | Firebase JS SDK (Auth, Firestore) |
| lucide-react | ^0.562.0 | Icon library |
| react-markdown | ^10.1.0 | Markdown rendering |
| react-pdf | ^7.7.3 | In-browser PDF viewer |
| remark-gfm | ^4.0.1 | GitHub Flavoured Markdown plugin |

### Dev Dependencies

| Package | Version |
|---|---|
| tailwindcss | ^4 |
| @tailwindcss/postcss | ^4 |
| typescript | ^5 |
| @types/node | ^20 |
| @types/react | ^19 |
| @types/react-dom | ^19 |
| eslint | ^9 |
| eslint-config-next | 16.0.3 |

### Frontend–Backend Communication

- **Backend API base URL variable:** `NEXT_PUBLIC_API_URL`
- **Value in `.env.example`:** `http://localhost:5000`
- **Frontend dev port:** 3000 (Next.js default; CORS origin set to `http://localhost:3000` in `app.py:36`)

---

## 6. Databases and Stores

### 6a. PostgreSQL

- **Installed binary:** `psql (PostgreSQL) 17.5 (Debian 17.5-1)` — source: `psql --version`
- **Runtime use by application:** **NOT USED.** No `psycopg2`, `psycopg`, `SQLAlchemy`, `alembic`, or `DATABASE_URL` references were found anywhere in the backend source. PostgreSQL is installed on the machine but the application does not connect to it.
- **Replacement:** Firestore (Google Cloud) is used for user accounts, roles, and chat history. SQLite (stdlib) is used for chunk bounding-box location metadata (`backend/locations.db`).

### 6b. SQLite (stdlib — `locations.db`)

- **Version:** Python stdlib `sqlite3` (no external driver)
- **File:** `backend/locations.db`
- **Purpose:** Stores page/bounding-box coordinates for PDF chunks, used to highlight source text in the PDF viewer.
- **Used in:** `build_case_chunks.py`, `constitutionlogic/constitutionbreaker.py`, `compute_constitution_bboxes.py`, `app.py` (at routes `/api/chunks/<chunk_id>/location` etc.)

### 6c. Neo4j

- **Version:** 5.18.0 Community Edition
- **Source:** `neo4j-community-5.18.0/packaging_info`
- **Installation:** Local binary at `neo4j-community-5.18.0/`, started via `START_NEO4J.sh`
- **Connection scheme/host:** `neo4j://localhost:7687` (Bolt). HTTP browser: `http://localhost:7474`. Source: `neo4j-java/src/…/Neo4jWriter.java:120`, `START_NEO4J.sh`
- **Edition:** Community (single-instance, no clustering)
- **Status at audit time:** Not running — no `neo4j` process found.
- **Queries:** Not queried (service not running).
- **Usage note:** Neo4j is populated by the separate Java subproject (`neo4j-java/`) from the Black Law Dictionary export (`export_blacklaw_to_json.py`). The main Flask/Python query pipeline does **not** call Neo4j at query time. The Python `neo4j` driver is **not** in `pip freeze` and not imported by any backend Python file.

### 6d. ChromaDB Collections

- **Version:** 1.3.5 (embedded `PersistentClient`, no HTTP server)
- **Source:** `pip freeze`, `chromadb.PersistentClient` in source

| Collection Name | Persist Directory | Disk Size | Document Count | Purpose |
|---|---|---|---|---|
| `case_law` | `backend/CaseLawDB/` | 4.7 M | **78** | FIRAC-processed case law chunks |
| `statutes` | `backend/StatutesDB/` | 20 M | **1,331** | Kenyan statutes/acts chunks |
| `master_rules` | `backend/constitutionlogic/MasterRulesDB/` | 8.3 M | **619** | Constitution of Kenya chunks |
| `legal_summaries` | `backend/LegalSummariesDB/` | 2.9 M | **40** | FIRAC summaries (older collection) |
| `ChunkDB` | `backend/ChunkDB/` | 12 M | **704** | General document chunks |
| `black_law_dictionary` | `backend/Const_IntepretationLaws/BlackLawDictionaryDB/` | **220 M** | **40,061** | Black's Law Dictionary entries |

*Counts obtained read-only via `python -c "chromadb.PersistentClient(...).get_collection(...).count()"` on 2026-11-02.*

### 6e. Embedding Model

- **Model ID:** `intfloat/e5-base-v2`
- **Dimension:** 768 (stated in `pixe.py:31`: "768 dimensions vs 384")
- **Inference:** Local CPU via `sentence_transformers.SentenceTransformer("intfloat/e5-base-v2")`
- **Source in code:** `backend/config.py` (`EMBEDDING_MODEL_NAME`), `pixe.py:31`, `Retriever.py:35`, `RulesRetriever.py:27`, `embed_knowledge.py:39`
- **HuggingFace cache:** `~/.cache/huggingface/` = **5.5 GiB** (contains model weights + tokenizer files)

### 6f. Firestore (Google Cloud)

- **SDK:** `google-cloud-firestore 2.34.1`, `firebase-admin 7.7.0`
- **Purpose:** User account profiles, role assignments (`advocate`/`admin`), chat history (conversation turns, titles, sources), idempotency cache for the generate endpoint.
- **Authentication:** Service account JSON file (path set via `FIREBASE_SERVICE_ACCOUNT_PATH` env var). Files `legalai-scc-firebase-adminsdk-fbsvc-73bde27853.json` and `legalai-service-account.json` exist in `backend/` but are gitignored.
- **Connection:** Cloud-hosted; no local emulator in production (emulator supported in tests via `FIRESTORE_EMULATOR_HOST`).

---

## 7. LLMs and External Services

**Source:** `grep -rnE "llama|mistral|qwen|gpt|MODEL_NAME|Groq|HuggingFace"` across `backend/`

### LLM Call Table

| File | Provider | Model ID | Purpose | Temperature | Max Tokens | Timeout |
|---|---|---|---|---|---|---|
| `groq_client.py:20` | Groq | `qwen/qwen3.8-27b` | Central answer generation; shared client with key rotation on HTTP 429 | 0 (default) | 2,500 (default) | 60 s |
| `Retriever.py:43,179` | Groq | `qwen/qwen3.8-27b` | Case-law retrieval: synthesise answer from retrieved chunks | 0.3 | 1,000 | 30 s |
| `Qclassifier.py:26,460` | Groq | `qwen/qwen3.8-27b` | Query classification / routing (Mode A: statutory lookup vs Mode B: strategic analysis) | 0.1 | 1,500 | 30 s |
| `pixe.py:26,136` | Groq | `qwen/qwen3.8-27b` | Query expansion (synonym generation) | 0.3 | 150 | — |
| `pixe.py:26,507` | Groq | `qwen/qwen3.8-27b` | RAG answer generation (final answer synthesis) | 0.3 | 800 | — |
| `CaseLaw/firac.py:30` | Groq | `qwen/qwen3.8-27b` | FIRAC extraction during ingestion (facts/issues/rules/application/conclusion); metadata extraction | 0.1 | 800–16,000 | 120 s |
| `constitutionlogic/constitution.py:26` | Groq | `qwen/qwen3.8-27b` | Constitution article analysis during ingestion | not set | not set | — |
| `agentrouter.py:13` | HuggingFace Inference Router | `meta-llama/Llama-3.1-8B-Instruct` | **EXPERIMENTAL / NOT INTEGRATED** — standalone test script only, not imported by `app.py` | — | 100 | — |

**Note:** All production LLM calls use Groq's OpenAI-compatible REST endpoint (`https://api.groq.com/openai/v1/chat/completions`) via Python `requests` directly (not the `openai` or `groq` SDK). The `groq` SDK (0.35.0) is imported only in `pixe.py`.

### Rate-Limit Handling

`groq_client.py` implements multi-key rotation on HTTP 429:
- Reads `GROQ_API_KEY` plus any `GROQ_API_KEY_2`, `GROQ_API_KEY_3`, … from environment
- On 429: rotates to the next key; if all keys exhausted, waits 30 s
- Raises `RuntimeError` if all keys exhausted after a full rotation; raises `requests.HTTPError` for non-429 errors.

### Other External Services

| Service | Endpoint / SDK | Used for |
|---|---|---|
| Groq API | `https://api.groq.com/openai/v1/chat/completions` | All production LLM inference |
| Firebase / Firestore | `google-cloud-firestore` SDK | Auth, user profiles, chat history |
| Firebase Storage | `google-cloud-storage` SDK | User-uploaded document storage |
| HuggingFace Hub | `huggingface-hub` library | Embedding model download (one-time) |
| HuggingFace Inference Router | `https://router.huggingface.co/…` | `agentrouter.py` only (experimental, not deployed) |

---

## 8. Configuration: Environment Variables and Secrets Hygiene

### Environment Variable Names (NAMES ONLY — no values)

**Backend Python (`backend/.env`, `os.getenv` in source):**

| Variable Name | Source file(s) | Purpose |
|---|---|---|
| `GROQ_API_KEY` | `backend/.env`, `groq_client.py`, `Retriever.py`, `Qclassifier.py`, `CaseLaw/firac.py` | Groq API primary key |
| `GROQ_API_KEY_2` | `groq_client.py` | Groq API secondary key (key rotation) |
| `GROQ_API_KEY_3` | `groq_client.py` | Groq API tertiary key (key rotation) |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | `backend/.env` | Path to Firebase service account JSON |
| `CORS_ORIGIN` | `backend/.env`, `app.py:36` | Allowed CORS origin (defaults to `http://localhost:3000`) |
| `FIRESTORE_EMULATOR_HOST` | `backend/tests/test_history.py` | Firestore emulator for tests only |
| `HUGGINGFACE_API_KEY` | `agentrouter.py` | HuggingFace Inference (experimental script only) |

**Frontend (`frontend/.env.local`, `process.env.*` in source):**

| Variable Name | Purpose |
|---|---|
| `NEXT_PUBLIC_FIREBASE_API_KEY` | Firebase Web SDK auth |
| `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN` | Firebase Auth domain |
| `NEXT_PUBLIC_FIREBASE_PROJECT_ID` | Firebase project |
| `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET` | Firebase Storage bucket |
| `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID` | FCM sender |
| `NEXT_PUBLIC_FIREBASE_APP_ID` | Firebase app identifier |
| `NEXT_PUBLIC_API_URL` | Backend base URL (value in `.env.example`: `http://localhost:5000`) |
| `NODE_ENV` | Set by Next.js framework automatically |

### Secrets Hygiene Findings

| Check | Result |
|---|---|
| `.env` listed in root `.gitignore` | ✅ YES — line 8 (`\n.env`) |
| `backend/.env` listed in `.gitignore` | ✅ YES — covered by pattern `.env` |
| `frontend/.env.local` listed in `.gitignore` | ✅ YES — listed in `frontend/.gitignore` |
| Firebase service account JSONs in `.gitignore` | ✅ YES — patterns `*-service-account.json`, `*firebase-adminsdk*.json`, `serviceAccount*.json` |
| `.env` ever committed to git history | ✅ NO — `git log --all -- .env '*.env'` returned no results |
| Hard-coded secrets in source files (api_key=literal, password=literal) | ✅ NONE FOUND — scan of `backend/` and `frontend/app/` returned no matches |
| `venv/` in `.gitignore` | ✅ YES |
| `__pycache__/` in `.gitignore` | ✅ YES |
| `node_modules/` in `.gitignore` | ✅ YES — in `frontend/.gitignore` |
| `.next/` in `.gitignore` | ✅ YES — in `frontend/.gitignore` |
| ChromaDB data folders in `.gitignore` | ❌ NOT listed (but all six chroma dirs appear as untracked `??` in `git status` — not yet committed) |
| `uploads/` in `.gitignore` | ❌ NOT listed |
| `neo4j-community-*/data/` in `.gitignore` | ✅ YES — but see Security Warnings (§16) |

---

## 9. Containers

**Source:** `docker --version`, `docker compose version`, `docker ps`, filesystem scan for Dockerfiles

- **Docker:** version 28.5.2+dfsg4, build `9cc6dea35e9a963f281434761c656fba4ac43aed`
- **Docker Compose:** NOT INSTALLED (`docker compose version` not found)
- **Dockerfile(s):** NONE found in the project tree
- **docker-compose / compose.yml:** NONE found
- **Running containers:** NONE (Docker daemon may or may not be running; `docker ps` returned no output)

**Conclusion:** The entire system runs natively on the host OS without containerisation. All services (Flask, Neo4j, ChromaDB) are managed as local processes.

---

## 10. Actual Startup Sequence

**Sources:** `START_NEO4J.sh`, `backend/app.py` (tail), `frontend/package.json` scripts, `README.md`

The system has no Makefile or single startup script. Steps must be performed manually in this order:

```
1. Set up environment files
   backend/.env:     GROQ_API_KEY, FIREBASE_SERVICE_ACCOUNT_PATH, CORS_ORIGIN
   frontend/.env.local: NEXT_PUBLIC_FIREBASE_*, NEXT_PUBLIC_API_URL=http://localhost:5000

2. Start Neo4j (Knowledge Graph store)
   Command:  bash START_NEO4J.sh
   Detail:   cd neo4j-community-5.18.0 && ./bin/neo4j start
   Ports:    7687 (Bolt), 7474 (HTTP browser)
   Notes:    First run requires changing the default password at http://localhost:7474

3. (One-time ingestion — only when building/updating the knowledge base)

   3a. Parse and chunk case law PDFs → locations.db + JSON
       Script:  backend/build_case_chunks.py
       Input:   PDF files in backend/SCCCases/ (10 PDFs, 1 DOCX as of audit)
       Output:  Chunk JSON files + backend/locations.db (SQLite bounding boxes)

   3b. Parse and chunk statute PDFs → JSON
       Script:  backend/build_statute_chunks.py
       Input:   Statute PDFs (path from data/processed/statutes/)
       Output:  Statute chunk JSON files

   3c. Embed all chunks → ChromaDB
       Script:  python backend/embed_knowledge.py --source both
       Input:   Chunk JSON files from steps 3a/3b
       Output:  backend/CaseLawDB/ (case_law, 78 docs) + backend/StatutesDB/ (statutes, 1331 docs)
       LLM:     No — pure embedding (sentence-transformers, local CPU)

   3d. Run FIRAC extraction on case PDFs → LegalSummariesDB
       Script:  backend/CaseLaw/ingester.py (or via API: POST /api/ingest-firac)
       Input:   Case law PDFs in backend/CaseLaw/ / backend/Caselaws/
       Output:  backend/LegalSummariesDB/ (legal_summaries, 40 docs)
       LLM:     YES — Groq qwen/qwen3.8-27b (FIRAC extraction, temperature=0.1)

   3e. Embed Constitution → MasterRulesDB
       Route:   POST /api/digest/constitution (triggers constitutionlogic pipeline)
       Input:   Constitution of Kenya PDF (MasterRules/ folder)
       Output:  backend/constitutionlogic/MasterRulesDB/ (master_rules, 619 docs)
                + backend/locations.db (constitution bounding boxes)
       LLM:     NO (embedding only at this step)

   3f. Build Black Law Dictionary graph → Neo4j
       Step i:  python backend/export_blacklaw_to_json.py
                Input:  backend/Const_IntepretationLaws/BlackLawDictionaryDB/ (ChromaDB, 40061 entries)
                Output: backend/Const_IntepretationLaws/blacklaw_export_for_neo4j.json
       Step ii: cd neo4j-java && mvn compile exec:java -Dexec.args='<path-to-json>'
                Output: Neo4j graph at bolt://localhost:7687

4. Start Flask backend
   Command:  cd backend && source venv/bin/activate && python app.py
   Port:     5000 (all interfaces: 0.0.0.0)
   Mode:     debug=True (development mode)

5. Start Next.js frontend
   Command:  cd frontend && npm run dev
   Port:     3000 (Next.js default)
   Dev URL:  http://localhost:3000
```

---

## 11. Flask Routes and Health Check

**Source:** `grep -n "@app.route\|@bp.route\|Blueprint"` on `backend/`

### Registered Blueprints

| Blueprint | Variable | Registered in |
|---|---|---|
| `history_bp` | `Blueprint("history", __name__)` | `history_routes.py` → `app.py` (`app.register_blueprint`) |

### Flask Routes (app.py)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | **Health check** — returns `{"status": "ok"}` |
| POST | `/api/chat` | RAG chat via `pixe.py` pipeline (role-gated: advocate/admin) |
| GET | `/api/files` | List uploaded files |
| POST | `/api/upload` | Upload a PDF/document |
| GET | `/api/files/<filename>` | Serve an uploaded file |
| POST | `/api/digest` | Digest a repo (code summarisation) |
| POST | `/api/firac` | Run FIRAC extraction on uploaded document |
| POST | `/api/ingest-firac` | Ingest FIRAC data into ChromaDB |
| POST | `/api/digest/constitution` | Trigger constitution embedding pipeline |
| GET | `/api/digest/constitution/status` | Constitution embedding status |
| DELETE | `/api/digest/constitution/reset` | Reset constitution embedding |
| GET | `/api/case-pdf` | Serve a case law PDF |
| GET | `/api/chunks/<chunk_id>/location` | Get PDF bounding box for a chunk |
| GET | `/api/documents/<path:doc_key>/chunks` | All chunks for a document |
| GET | `/api/cases/<path:case_identifier>/all-chunks` | All chunks for a case |
| GET | `/api/statutes/<path:source_document>/all-chunks` | All chunks for a statute |
| GET | `/api/rule-pdf` | Serve a constitution/rule PDF |
| POST | `/api/qretrieve` | Query retrieval (runs classifier + retrievers) |
| POST | `/api/generate-legal-answer` | Full pipeline: classify → retrieve → generate (main endpoint) |
| POST | `/api/case-laws/upload` | Upload a case law file |
| GET | `/api/case-laws/file/<path:filename>` | Serve a case law file from SCCCases/ |

### Blueprint Routes (history_routes.py)

| Method | Path | Purpose |
|---|---|---|
| (various) | `/api/history/…` | Chat history read/manage endpoints (exact paths in `history_routes.py`) |

### Health Check

**Simplest end-to-end probe:** `GET http://localhost:5000/api/health` → `{"status": "ok"}` (HTTP 200, no auth required)

---

## 12. Observed Resource Use

**Services were NOT running at audit time (2026-11-02).** Resource figures cannot be reported.

Current machine state (static snapshot only):
- RAM available: 5.4 GiB of 15 GiB total
- Swap used: 20 MiB of 15 GiB

**Estimated minimum RAM requirements (from knowledge of components):**
- Flask + Python workers: ~150–300 MB
- Sentence-transformers model (`e5-base-v2`, 768-dim): ~500–800 MB
- ChromaDB (embedded, all collections loaded): ~200–500 MB
- Neo4j Community (JVM): ~512 MB–1 GiB
- Next.js dev server: ~200–400 MB
- **Estimated total for full stack:** ~2–3 GiB RAM minimum

---

## 13. Version Control Summary

**Source:** `git remote -v`, `git branch -a`, `git rev-list --count HEAD`, `git log --oneline`

| Item | Value |
|---|---|
| **Remote** | `git@github.com:Houstonsboy/Retreival-Augmented-Generation-AI.git` (fetch & push) |
| **Current branch** | `master` |
| **All branches** | `master` (current), `backup`, `temp-fix`; remote: `origin/master` |
| **Total commits** | 56 |
| **First commit** | `0226564` — "this is the first successful rag script" |
| **Latest commit** | `e96979e` — "feat: Round 2 skeleton — viewport-anchored shimmer + unified Skeleton component" |
| **Tags/releases** | None |

### Recent Commit Log (last 15)

```
e96979e feat: Round 2 skeleton — viewport-anchored shimmer + unified Skeleton component
9f8a335 feat: production skeleton loading for legal research workspace
4608dcb fix: navigate to dashboard after profile setup without requiring refresh
1ffd1be feat: Phase 2 frontend — history, routing, sidebar, idempotency
8c518dc feat: idempotent generate endpoint — cache hit skips LLM, retry overwrites failed turn
fea5b20 docs: add Firestore security rules, composite indexes, and CHAT_HISTORY guide
b1a6a4c feat: add Flask Blueprint for chat history management endpoints
540f0c1 feat: wire chat history into generate-legal-answer endpoint
989fd4b feat: add history persistence layer (history_store.py + tests)
641643b refactor: extract _groq closure to shared groq_client module
57edd66 chore: ignore neo4j runtime data and binaries
211b511 feat: redesign chat workspace UI with gold/cream theme and markdown rendering
0f36d8b Fix F: end-to-end test Amani/M-Pesa query; fix precision tagging for bare [S#]
e807d85 Fix B: compute constitution bboxes via rapidfuzz word-alignment
f284295 Fix A: replace <object> PDF embed with react-pdf + bbox overlay
```

---

## 14. Differences from the Expected Stack

The following items differ from the project description provided in the brief:

| # | Expected | Actual | Evidence |
|---|---|---|---|
| 1 | PostgreSQL for accounts, uploads, compliance results | **NOT USED.** Firestore (Google Cloud) handles user accounts and chat history. SQLite (`locations.db`) stores chunk PDF bounding boxes. No `psycopg2`, `SQLAlchemy`, or `alembic` installed. | `pip freeze` (absent), source grep |
| 2 | ChromaDB collections named `master_rules` and `contract_law_cases` | `master_rules` exists (619 docs, Constitution). **No `contract_law_cases` collection.** Actual collections: `case_law` (78), `statutes` (1331), `master_rules` (619), `legal_summaries` (40), `ChunkDB` (704), `black_law_dictionary` (40,061). | `chromadb PersistentClient.list_collections()` |
| 3 | Neo4j used in query pipeline ("GraphRAG") | Neo4j is **only used during data ingestion** (Black Law Dictionary → graph, via a Java subproject). **The Python query pipeline does not call Neo4j at all.** No Python `neo4j` driver is installed. The Java client (`neo4j-java/`) connects at `neo4j://localhost:7687`. | `pip freeze` (no neo4j driver), source grep on backend Python |
| 4 | LLM "most likely via Groq API" | Confirmed Groq API. Model is **`qwen/qwen3.8-27b`** (not llama, not mistral). All production calls go to `https://api.groq.com/openai/v1/chat/completions` via raw `requests`. | `groq_client.py:20`, all MODULE_NAME constants |
| 5 | GPU expected / possible | **No GPU.** Intel UHD Graphics 620 (integrated) only. PyTorch is CPU-only; no CUDA references in code. | `lspci`, `nvidia-smi` (not found), source grep |
| 6 | Embeddings "most likely intfloat/e5-base-v2" | Confirmed: **`intfloat/e5-base-v2`**, 768-dim, loaded locally via `sentence_transformers.SentenceTransformer`. No embedding API calls. | `config.py`, `pixe.py:31` |
| 7 | Two-tier "GraphRAG" architecture | The system is better described as **multi-collection hybrid RAG** (dense + BM25 + cross-encoder reranking) with a separate, non-query-time Neo4j graph. GraphRAG-style graph traversal does not occur at query time. | Source analysis |
| 8 | Flask with Flask extensions (flask-sqlalchemy, flask-login, flask-jwt-extended) | Only **flask-cors** is installed. No flask-sqlalchemy, flask-login, or flask-jwt-extended. Auth uses Firebase Admin SDK + `PyJWT` for token verification (custom `@require_role` decorator in `auth.py`). | `pip freeze` |
| 9 | Docker / Containers expected | **No Docker usage for services.** Docker (28.5.2) is installed on the machine but no Dockerfile or compose file exists. Neo4j, ChromaDB, and Flask all run as native host processes. | filesystem scan, `docker ps` |
| 10 | `langchain-groq` expected | **Not installed.** LangChain is used only for `Document` objects and text splitting. Groq calls are made directly via `requests`. | `pip freeze` |
| 11 | README for setup | README.md exists but **describes the original prototype** (uses `HF_API_TOKEN`, references `pixe.py` as a CLI tool, no mention of Flask API, Firebase, Neo4j, or the multi-collection ChromaDB setup). It is significantly out of date with the current implementation. | `README.md` content vs actual code |

---

## 15. NOT FOUND / Needs Manual Confirmation

| Item | What Was Looked For | Where Checked | Status |
|---|---|---|---|
| Neo4j graph contents (node labels, relationship types, indexes) | `MATCH (n) RETURN labels(n), count(*)` etc. | Service not running at audit time | **NOT QUERIED** — Neo4j was not running |
| PostgreSQL database name, tables | `pg_config`, `SELECT version()`, SQLAlchemy models | Source grep, `pip freeze` | **NOT APPLICABLE** — PostgreSQL not used by this application |
| Running process resource use | `ps`, `docker stats` | Services not running | **NOT MEASURED** — no services running at audit time |
| Flask debug mode in production | `debug=True` in `app.py` tail | `app.py` last line | **WARNING:** `app.run(debug=True, port=5000)` — debug mode is ON. Acceptable for local development; must be changed for any deployment. |
| Exact `FIREBASE_SERVICE_ACCOUNT_PATH` value (to confirm file location) | `backend/.env` variable value | Value intentionally redacted per Rule 3 | Not verified |
| `langchain-text-splitters` usage in chunking scripts | Source grep | NOT imported in the main chunking scripts (`build_case_chunks.py`, `build_statute_chunks.py`); used only in `embed_knowledge.py` | Confirmed present but minor |
| `agentrouter.py` integration status | Whether it is imported anywhere | Source grep: not imported by `app.py` or any other module | Confirmed **standalone experimental script only** |
| Actual Neo4j graph statistics | Node/relationship counts | Service not running | Requires manual query after `START_NEO4J.sh` |
| `constitution.py` exact LLM call parameters | temperature, max_tokens | File was not fully read | Partial — model confirmed as `qwen/qwen3.8-27b`; exact call params not extracted |

---

## 16. Security Warnings

| Severity | Finding | Location | Detail |
|---|---|---|---|
| **MEDIUM** | Neo4j binary database files committed to git | `neo4j-community-5.18.0/data/databases/neo4j/neostore.propertystore.db.index.keys` (and `.id`), `neo4j-community-5.18.0/data/databases/system/neostore.propertystore.db.index.keys` (and `.id`) — confirmed via `git ls-files` | These 4 Neo4j internal binary index files are tracked in the repository despite the `.gitignore` entry `neo4j-community-*/data/`. They were likely committed before the gitignore rule was added. These particular files store property key names (not values), so they are unlikely to contain secrets, but binary database files should not be in version control. Recommendation: add them to `.gitignore` and remove from git history with `git rm --cached`. |
| **LOW** | Flask running in `debug=True` mode | `backend/app.py`, last line: `app.run(debug=True, port=5000, host='0.0.0.0')` | Debug mode exposes the Werkzeug interactive debugger (which allows remote code execution if reached). Acceptable for localhost development only. Must be changed to `debug=False` (or use an env variable) before any external deployment. |
| **LOW** | Firebase service account JSON files present in `backend/` | `backend/legalai-scc-firebase-adminsdk-fbsvc-73bde27853.json`, `backend/legalai-service-account.json` | These files are correctly gitignored and were not committed. However, their presence in the repository directory means they could be accidentally included in a zip archive or rsync deployment. Recommend storing them outside the project directory and referencing by absolute path. |
| **LOW** | ChromaDB data directories not in `.gitignore` | Root `.gitignore` has no entries for `CaseLawDB/`, `StatutesDB/`, `LegalSummariesDB/`, `MasterRulesDB/`, `ChunkDB/`, `BlackLawDictionaryDB/` | These directories (totalling ~267 MB) are currently untracked (`??` in `git status`), so they have not been committed. However, a future `git add .` would inadvertently include them. Recommend adding explicit ignore patterns to `.gitignore`. |
| **INFO** | No hard-coded secrets found in source | Full scan of `backend/` and `frontend/app/` Python/JS/TS files for patterns `api_key=<literal>`, `password=<literal>`, `gsk_`, `sk-` | **CLEAN** — no hard-coded credential values detected. |
| **INFO** | No `.env` files in git history | `git log --all --oneline -- .env '*.env'` | **CLEAN** — no `.env` files were ever committed. |
