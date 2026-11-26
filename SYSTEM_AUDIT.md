# SYSTEM AUDIT REPORT
## GraphRAG Legal Decision-Support System for Kenya's Small Claims Court
**Audit date:** 2026-11-17  
**Auditor note:** All claims in this document are verified directly from source files. File paths and line numbers are cited. Where information could not be verified, "NOT FOUND" or "UNCLEAR" is stated.

---

## 1. PROJECT OVERVIEW

### Folder Tree (2–3 levels, excluding node_modules, venv, .next, build artefacts)

```
Retreival-Augmented-Generation-AI/
├── backend/
│   ├── app.py                      # Flask application and all API routes
│   ├── auth.py                     # Firebase Admin token verification + role enforcement
│   ├── Qclassifier.py              # LLM-based legal query classifier
│   ├── Retriever.py                # Hybrid BM25+vector+cross-encoder case-law retrieval
│   ├── RulesRetriever.py           # Hybrid retrieval for statutory provisions
│   ├── groq_client.py              # Shared Groq LLM client with key rotation
│   ├── history_routes.py           # Flask Blueprint: /api/conversations endpoints
│   ├── history_store.py            # Firestore persistence for chat history
│   ├── coverage.py                 # Acts-in-corpus coverage checker
│   ├── citation_verifier.py        # Post-generation citation hallucination check
│   ├── config.py                   # Central DB paths and embedding model name
│   ├── digester.py                 # Legacy general RAG digester
│   ├── pixe.py                     # Legacy general RAG pipeline
│   ├── agentrouter.py              # (present, purpose unclear)
│   ├── build_case_chunks.py        # Ingestion: PDF → FIRAC chunks → CaseLawDB
│   ├── build_statute_chunks.py     # Ingestion: HTML statutes → StatutesDB
│   ├── requirements.txt            # Python dependencies (pinned)
│   ├── .env                        # Local secrets (not in git)
│   ├── CaseLawDB/                  # ChromaDB: case_law collection
│   ├── StatutesDB/                 # ChromaDB: statutes collection
│   ├── LegalSummariesDB/           # ChromaDB: legacy legal_summaries collection
│   ├── ChunkDB/                    # ChromaDB: legacy chunk store
│   ├── constitutionlogic/
│   │   ├── MasterRulesDB/          # ChromaDB: master_rules (Constitution + Evidence Act)
│   │   ├── constitutionbreaker.py  # Constitution PDF → article/chunk extractor
│   │   ├── constitution_digester.py
│   │   ├── evidencedigester.py
│   │   ├── masteruledb.py
│   │   └── constitution.py / evidence.py
│   ├── CaseLaw/
│   │   ├── firac.py                # LLM-based FIRAC extractor from PDF text
│   │   └── ingester.py             # Store FIRAC chunks in CaseLawDB
│   ├── SCCVision/
│   │   ├── router.py               # Flask Blueprint: /sccvision/* endpoints
│   │   ├── engine.py               # RapidOCR singleton
│   │   ├── service.py              # OCR text extraction + WhatsApp analysis
│   │   ├── preprocess.py           # Image decode + upscale/invert
│   │   ├── analysis.py             # WhatsApp contact/message/event parser
│   │   └── analyst/
│   │       ├── pipeline.py         # analyze() / analyze_case() entry points
│   │       ├── client.py           # AnalystLLM: Groq call + Pydantic parse
│   │       ├── models.py           # Pydantic models: AnalystResult, etc.
│   │       ├── enrich.py           # PII masking, payload builder
│   │       ├── verify.py           # Deterministic post-LLM verification rules
│   │       ├── rules.py            # check_mpesa_cross_match, compute_verdict, etc.
│   │       ├── mpssa.py            # M-Pesa screenshot classifier + parser
│   │       ├── dates.py            # Date arithmetic for 3-year SCC limitation
│   │       └── __init__.py         # PROMPT_VERSION constant
│   ├── MasterRules/                # Source PDFs: Constitution, Evidence Act, etc.
│   ├── SCCCases/                   # Uploaded SCC case PDFs/DOCXs
│   ├── Repo/                       # Legacy document upload folder
│   ├── Const_IntepretationLaws/    # Supplementary PDFs + BlackLawDictionary embeddings
│   ├── CaseLaw/ (Caselaws/)        # Source case PDFs
│   ├── scripts/
│   │   ├── bootstrap_admin.py      # Set Firebase custom claim: admin
│   │   ├── set_role.py             # Set any role via Admin SDK
│   │   └── seed_caselaws.py        # Seed Firestore CaseLaws collection
│   ├── tests/
│   │   ├── test_history.py         # pytest: history_store + history_routes
│   │   └── test_sccvision_analyst.py # pytest: SCCVision analyst unit tests
│   ├── logs/
│   │   ├── embed_*.log             # Embedding run logs
│   │   └── hallucinations.jsonl    # Citation verification failure log
│   ├── locations.db                # SQLite: chunk→page+bbox index
│   └── admissibility_notes/
│       └── s106b_source_text.md    # s.106B statutory text reference
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx / layout.tsx / loading.tsx
│   │   ├── auth/
│   │   │   ├── sign-in/page.tsx
│   │   │   ├── sign-up/page.tsx
│   │   │   └── complete-profile/page.tsx
│   │   ├── dashboard/page.tsx
│   │   ├── research/page.tsx       # Advocate chat (wraps ChatView)
│   │   ├── chat/[conversationId]/page.tsx
│   │   ├── constitution/page.tsx   # Admin: Constitution ingestion UI
│   │   ├── firacChecker/page.tsx   # Admin: FIRAC extraction UI
│   │   ├── ingester/page.tsx       # Admin: document upload + digest
│   │   ├── repository/
│   │   │   ├── page.tsx
│   │   │   ├── caselaws/page.tsx
│   │   │   └── statutes/page.tsx
│   │   ├── litigant/
│   │   │   ├── sign-in/page.tsx
│   │   │   ├── page.tsx
│   │   │   ├── evidence/page.tsx   # SCC: upload & view evidence items
│   │   │   ├── analysis/page.tsx   # SCC: trigger + view contract analysis
│   │   │   ├── certificates/page.tsx
│   │   │   └── layout.tsx
│   │   ├── components/
│   │   │   ├── ChatView.tsx        # Main Advocate RAG workspace (~1500 LOC)
│   │   │   ├── PdfViewer.tsx       # PDF + highlight overlay viewer
│   │   │   ├── Navigation.tsx
│   │   │   ├── AuthGuard.tsx
│   │   │   ├── UserMenu.tsx
│   │   │   ├── evidence/
│   │   │   │   └── EvidenceDropzone.tsx
│   │   │   ├── sccvision/
│   │   │   │   └── SCCVisionTest.tsx
│   │   │   └── skeleton/           # 10 Skeleton loading components
│   │   ├── context/
│   │   │   └── AuthContext.tsx     # Firebase Auth React context
│   │   └── lib/
│   │       ├── firebase.ts         # Firebase client SDK init
│   │       ├── firestore.ts        # Firestore user/username helpers
│   │       ├── api.ts              # Authenticated fetch wrapper
│   │       ├── historyApi.ts       # /api/conversations client
│   │       ├── evidenceApi.ts      # Evidence API client
│   │       ├── evidenceTypes.ts    # EvidenceRecord, EvidenceStatus types
│   │       ├── evidenceConfig.ts   # Evidence type config
│   │       └── access.ts           # Role-based path permissions
│   ├── package.json
│   └── next.config.ts
│
├── data/
│   ├── raw_statutes/               # 14 HTML statute files from Kenya Law
│   │   └── manifest.json           # Act slugs, titles, section counts, SHA256
│   └── processed/statutes/         # (directory present; contents not inspected)
│
├── firestore.rules                 # Firestore security rules
├── firestore.indexes.json          # Composite index definitions
├── firebase.json                   # Firebase hosting + emulators config
├── firestore-tests/                # Jest/TypeScript Firestore rules tests
├── .gitignore
├── .env.example                    # Lists: HF_API_TOKEN only
├── README.md                       # Describes early prototype only (see §6)
├── neo4j-community-5.18.0/        # Neo4j community binary (not used in app)
└── [various *.md audit/analysis docs]
```

### What the System Does

The system is a Kenyan legal research assistant for the Small Claims Court context. It has two main sides. The **Advocate side** presents a chat-style research workspace where an authenticated advocate or admin types a natural-language legal question; the backend classifies the query, retrieves relevant chunks from ChromaDB (using BM25+vector search fused by RRF and reranked by a cross-encoder), feeds the retrieved statutes and case-law passages into a Groq-hosted large language model (`qwen/qwen3.8-27b`), and returns a structured grounded answer with inline citations, follow-up questions, and PDF-linked sources — all stored as a conversation turn in Firestore. The **SCC side** (for litigants and adjudicators) accepts screenshots of WhatsApp conversations and M-Pesa payment confirmations, runs local CPU-based OCR (RapidOCR), extracts transaction and conversation data, and calls the same Groq LLM to produce a Section 106B evidence-admissibility analysis with deterministic rule verification.

### Module Attribution

The codebase does **not** use the labels "ADVOCATE" and "SCC" explicitly as top-level folder boundaries. Attribution is determined by role-gating and route structure:

| Component | Module | Basis |
|---|---|---|
| `/research`, `/chat/*`, `/dashboard`, `/repository/*` pages | ADVOCATE | `access.ts:28–29` — `ADVOCATE_ONLY_ROUTES` |
| `/api/qretrieve`, `/api/generate-legal-answer`, `/api/chat`, `/api/case-pdf`, `/api/conversations/*` | ADVOCATE | `@require_role("advocate","admin")` in `app.py` |
| `Qclassifier.py`, `Retriever.py`, `RulesRetriever.py`, `coverage.py`, `citation_verifier.py` | ADVOCATE | Called only from advocate API routes |
| `/litigant/*` pages | SCC | `access.ts:34` — all three roles |
| `SCCVision/` package, `/sccvision/*` routes | SCC | `router.py` blueprint; used by litigant evidence flow |
| `auth.py`, `groq_client.py`, `history_store.py`, Firebase, ChromaDB | Shared | Used by both modules |
| `constitutionlogic/`, `CaseLaw/`, `build_case_chunks.py`, `build_statute_chunks.py` | Shared (ingestion) | Admin-only but feeds the advocate RAG |

---

## 2. IMPLEMENTATION ENVIRONMENT

### 2a. Software and Versions

#### Backend (Python)

| Component | Exact Version | Purpose | Source file |
|---|---|---|---|
| Python | 3.13 (inferred from venv path `lib/python3.13`) | Runtime | `backend/venv/` |
| Flask | 3.1.2 | HTTP API framework | `requirements.txt:34` |
| flask-cors | 6.0.1 | CORS handling | `requirements.txt:35` |
| firebase-admin | 7.7.0 | Auth token verification, Firestore writes | `requirements.txt:38` |
| groq | 0.35.0 | (installed but NOT used; app calls Groq REST directly) | `requirements.txt:41` |
| chromadb | 1.3.5 | Vector store for all embeddings | `requirements.txt:27` |
| sentence-transformers | 5.1.2 | `intfloat/e5-base-v2` embeddings + CrossEncoder | `requirements.txt:3` |
| torch | 2.9.1 | Inference for sentence-transformers | `requirements.txt:22` |
| transformers | 4.57.1 | Transformer model loading | `requirements.txt:23` |
| rank-bm25 | 0.2.2 | BM25Okapi keyword retrieval | `requirements.txt:20` |
| scikit-learn | 1.7.2 | Utility (cosine similarity etc.) | `requirements.txt:7` |
| numpy | 2.3.5 | Numerical arrays | `requirements.txt:8` |
| requests | 2.32.5 | HTTP calls to Groq REST API | `requirements.txt:9` |
| pypdf | 6.3.0 | PDF text extraction | `requirements.txt:12` |
| PyMuPDF (fitz) | 1.24.14 | Coordinate-aware PDF extraction + rendering | `requirements.txt:13` |
| pytesseract | 0.3.13 | OCR wrapper (requires system `tesseract-ocr`) | `requirements.txt:14` |
| Pillow | 11.1.0 | Image handling | `requirements.txt:15` |
| ocrmypdf | 16.5.0 | Scanned-PDF routing (pre-ingestion) | `requirements.txt:16` |
| rapidocr | 3.10.0 | Local CPU-based OCR engine (SCCVision) | `requirements.txt:44` |
| onnxruntime | 1.23.2 | ONNX model runtime for RapidOCR | `requirements.txt:45` |
| opencv-python-headless | 5.0.0.93 | Image processing for OCR | `requirements.txt:46` |
| python-docx | 1.2.0 | DOCX case file ingestion | `requirements.txt:11` |
| langchain | 1.0.7 | Document processing utilities | `requirements.txt:5` |
| langchain-core | 1.0.6 | LangChain core | `requirements.txt:6` |
| python-dotenv | 1.2.1 | `.env` loading | `requirements.txt:2` |
| huggingface-hub | 0.36.0 | Model download | `requirements.txt:4` |
| opentelemetry-* | 1.45.1 | Required by ChromaDB | `requirements.txt:28–31` |
| werkzeug (via Flask) | UNCLEAR | `secure_filename` used in `app.py:27` | `app.py:27` |
| pytest | UNCLEAR (dev dependency) | Test runner | `backend/tests/*.py` |
| rapidfuzz | UNCLEAR | Word-alignment for constitution bbox | commit `e807d85` |

#### Frontend (Node/TypeScript)

| Component | Exact Version | Purpose | Source file |
|---|---|---|---|
| Next.js | 16.0.3 | React framework (App Router) | `package.json:14` |
| React | 19.2.0 | UI library | `package.json:15` |
| react-dom | 19.2.0 | DOM bindings | `package.json:16` |
| firebase | ^10.12.0 | Client SDK: Auth, Firestore | `package.json:12` |
| react-markdown | ^10.1.0 | Markdown rendering for answers | `package.json:17` |
| remark-gfm | ^4.0.1 | GitHub Flavored Markdown | `package.json:19` |
| react-pdf | ^7.7.3 | In-browser PDF rendering + bbox overlay | `package.json:18` |
| lucide-react | ^0.562.0 | Icon library | `package.json:13` |
| tailwindcss | ^4 | Utility CSS | `package.json` devDeps |
| TypeScript | ^5 | Static types | `package.json` devDeps |

#### Firestore Security Rules Tests

| Component | Exact Version | Purpose | Source file |
|---|---|---|---|
| @firebase/rules-unit-testing | see `firestore-tests/package.json` | Rules unit tests | `firestore-tests/firestore.test.ts` |
| jest | see `firestore-tests/package.json` | Test runner | `firestore-tests/jest.config.js` |

---

### 2b. External Services and APIs

| Service | Where called | Purpose | Model / tier |
|---|---|---|---|
| **Groq API** (`api.groq.com/openai/v1/chat/completions`) | `groq_client.py:21`, `Qclassifier.py:27`, `Retriever.py:44`, `SCCVision/analyst/client.py` | All LLM inference: query classification, answer synthesis, case summaries, title generation, admissibility analysis | `qwen/qwen3.8-27b` (all usages; `groq_client.py:19`, `Qclassifier.py:26`) |
| **Hugging Face / local** (sentence-transformers) | `Retriever.py:601`, `RulesRetriever.py` | Embedding: `intfloat/e5-base-v2` (768-dim) | Local model loaded at runtime; no external API call at inference |
| **Hugging Face / local** (cross-encoder) | `Retriever.py:84`, `RulesRetriever.py:77` | Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local model |
| **Firebase Authentication** | `auth.py:49` (`fb_auth.verify_id_token`) | Stateless token verification for every protected API call | Firebase project: `legalai-scc` (inferred from `auth.py:31` storageBucket) |
| **Firestore** | `history_store.py`, `auth.py`, `app.py:1876` | User profiles, chat history (conversations + turns), CaseLaws metadata | `legalai-scc` project; default Firestore instance |
| **Firebase Storage** | `auth.py:31` (storageBucket configured) | Bucket configured; direct read/write calls NOT found in application code | `legalai-scc.firebasestorage.app` |
| **Kenya Law / kenyalaw.org** | `download_statutes.py` (data-ingestion script, not in runtime) | Source of statute HTML files | Static download; 14 acts cached in `data/raw_statutes/` |
| **Neo4j** | NOT FOUND in runtime code or `requirements.txt` | — | Binary present at `neo4j-community-5.18.0/`; a knowledge graph was started (commit `5a2cf88 knowledge graph`, 2026-02-13) but no Neo4j driver appears in `requirements.txt` and no Cypher queries are executed by the application |

---

### 2c. Environment Variables

No `.env.example` in `backend/` exists; the root `.env.example` lists only `HF_API_TOKEN`. Variables are inferred from `backend/app.py`, `auth.py`, `groq_client.py`, and `Qclassifier.py`.

| Variable name | Component | Description |
|---|---|---|
| `GROQ_API_KEY` | `Qclassifier.py:14`, `groq_client.py:37`, `Retriever.py:39` | Primary Groq API key |
| `GROQ_API_KEY_2` | `Qclassifier.py:15` | Secondary Groq API key (rotation) |
| `GROQ_API_KEY_3` | `Qclassifier.py:16` | Tertiary Groq API key (rotation) |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | `auth.py:20` | Path to Firebase Admin SDK JSON |
| `CORS_ORIGIN` | `app.py:38` | Allowed frontend origin (default `http://localhost:3000`) |
| `HF_API_TOKEN` | `README.md`, `.env.example` | Hugging Face token (used by older `pixe.py` prototype only) |
| `FIRESTORE_EMULATOR_HOST` | `backend/tests/test_history.py:31` | Enables emulator integration tests |
| `NEXT_PUBLIC_API_URL` | `frontend/app/components/ChatView.tsx:50` | Backend base URL (default `http://localhost:5000`) |
| `NEXT_PUBLIC_FIREBASE_*` | `frontend/app/lib/firebase.ts` (inferred) | Firebase client SDK config |

Frontend Firebase config (`NEXT_PUBLIC_FIREBASE_API_KEY`, etc.) is in `frontend/.env.local` (not in git). An `.env.example` is present at `frontend/.env.example`.

A `.env.example` **does** exist at the root (`/.env.example`), listing only `HF_API_TOKEN`. It does **not** document the backend Groq or Firebase variables. `frontend/.env.example` exists but its contents were not read; it likely mirrors the Firebase client config keys.

---

### 2d. Deployment Configuration

| Item | Status |
|---|---|
| Dockerfile | NOT FOUND |
| docker-compose.yml | NOT FOUND |
| Procfile | NOT FOUND |
| railway.json / railway.toml | NOT FOUND |
| nixpacks.toml | NOT FOUND |
| vercel.json | NOT FOUND |
| `.github/workflows/` CI/CD | NOT FOUND |
| `firebase.json` | PRESENT — configures Firebase Hosting (target: `hosting-public/`) and Firestore emulator (port 8080) |

**Start commands:**
- Backend: `python backend/app.py` → Flask dev server on port 5000, host `0.0.0.0` (`app.py:1920`). No production WSGI config (gunicorn, uWSGI) is present.
- Frontend: `npm run dev` (Next.js dev server, port 3000) or `npm run build && npm start` (`package.json:5–8`).

**Conclusion:** The system currently appears to be **local-only** (development). There is no production deployment configuration verified in the repository.

---

### 2e. Setup / Run Steps

As reconstructable from the repo (README describes an older prototype; steps below are from current code):

```bash
# 1. Clone
git clone git@github.com:Houstonsboy/Retreival-Augmented-Generation-AI.git
cd Retreival-Augmented-Generation-AI

# 2. Backend Python environment
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Note: also requires system packages: tesseract-ocr (for pytesseract)

# 3. Backend environment variables
cp ../env.example .env   # (no backend-specific .env.example exists)
# Manually add:
#   GROQ_API_KEY=...
#   GROQ_API_KEY_2=...  (optional)
#   GROQ_API_KEY_3=...  (optional)
#   FIREBASE_SERVICE_ACCOUNT_PATH=legalai-scc-firebase-adminsdk-fbsvc-73bde27853.json
#   CORS_ORIGIN=http://localhost:3000

# 4. Data ingestion — statutes (must run before the application serves statute queries)
#   Download statutes: python download_statutes.py   (at repo root)
#   Build StatutesDB:  python backend/build_statute_chunks.py

# 5. Data ingestion — case law
#   Add PDF files to backend/SCCCases/
#   Build CaseLawDB:  python backend/build_case_chunks.py

# 6. Data ingestion — Constitution (can also be done via API)
#   POST /api/digest/constitution   (requires admin token)
#   Or run: python backend/constitutionlogic/constitution_digester.py

# 7. Run backend
cd backend
python app.py   # Flask dev server on :5000

# 8. Frontend
cd frontend
npm install
cp .env.example .env.local
# Edit .env.local: add NEXT_PUBLIC_FIREBASE_* and NEXT_PUBLIC_API_URL
npm run dev   # Next.js dev server on :3000

# 9. (Optional) Firestore rules tests
cd firestore-tests
npm install
# Requires Firebase emulator running:
firebase emulators:start --only firestore --project legalai-rules-test
npm test
```

No README covering the current stack exists. The root `README.md` describes an earlier prototype using Hugging Face and Mistral (see §6).

---

## 3. SYSTEM IMPLEMENTATION

---

### ADVOCATE Module

---

#### 3a. Data Layer — ADVOCATE

**Firebase (Firestore)**

| Collection path | Key fields | Description | Read/write location |
|---|---|---|---|
| `users/{uid}` | `username`, `email`, `type` (advocate/admin/litigant), `authProvider`, `advocateVerified`, `createdAt`, `updatedAt` | User profile; `type` is the role field checked by `require_role` | Read: `auth.py:56`. Write: `frontend/app/lib/firestore.ts:39–42` (atomic batch) |
| `usernames/{username}` | `uid` | Username uniqueness index | Read: open to all (`firestore.rules:70`). Write: atomic batch with `users/{uid}` |
| `users/{uid}/conversations/{cid}` | `title`, `title_source`, `pinned`, `turn_count`, `last_query_preview`, `created_at`, `updated_at`, `schema_version` | Conversation metadata | Read: client. Write: backend only (`history_store.py`) |
| `users/{uid}/conversations/{cid}/turns/{tid}` | `turn_index`, `status` (complete/failed), `query`, `answer_markdown`, `answer_structured`, `sources`, `alias_map`, `coverage`, `retrieval_summary`, `model`, `latency_ms`, `error`, `error_code`, `client_turn_id`, `truncated`, `schema_version` | Individual Q&A turns | Read: client. Write: backend only (`history_store.py`) |

**Security rules** (`firestore.rules`):
- `users/{uid}`: read own only; create enforces 7-field schema, `type` limited to `advocate`/`litigant`, atomic with username claim; update locks `type` and `advocateVerified` (privilege escalation impossible from client); delete denied.
- `usernames/{username}`: read open (pre-auth availability check); write atomic only.
- `CaseLaws/{docId}`: authenticated read; write denied (backend Admin SDK only).
- `users/{uid}/conversations/{cid}` and `/turns/{tid}`: read own; write denied (backend only).
- Catch-all `{document=**}`: deny all.

**ChromaDB (Vector Stores)**

| Collection | DB directory | Purpose |
|---|---|---|
| `case_law` | `backend/CaseLawDB/` | FIRAC chunks from Kenya Law Reports case PDFs |
| `statutes` | `backend/StatutesDB/` | Section-level chunks from 14 HTML statutes |
| `master_rules` | `backend/constitutionlogic/MasterRulesDB/` | Constitution articles + Evidence Act sections |

Chunk metadata for `case_law`: `case_identifier`, `firac_component` (FACTS/ISSUES/RULES/APPLICATION/CONCLUSION), `chunk_index`, `prev_chunk_id`, `next_chunk_id`, `file_name`, `parties`, `court_level`, `judge`, `year`, `legal_domain`, `winning_party`, `source_file_path`.

Chunk metadata for `statutes`: `source_document` (act slug), `act_title`, `article_number` (section number), `article_header`, `chunk_index`, `document_type`.

Chunk metadata for `master_rules`: `document_type` (`constitution` or `evidence`), `article_number`, `article_header`, `chunk_index`, `source_document`.

**SQLite sidecar** (`backend/locations.db`): table `chunk_locations` with fields `chunk_id`, `collection`, `source_file_path`, `served_file_path`, `page_start`, `page_end`, `boxes_json`, `page_dims_json`, `label`, `ocr`. Used by `app.py:1072–1101` to return bounding-box data for the PDF viewer overlay.

**Neo4j**: NOT USED in production. The binary `neo4j-community-5.18.0/` is present and there is a commit (`5a2cf88`, 2026-02-13) titled "knowledge graph," but the Neo4j driver (`neo4j` Python package) does not appear in `requirements.txt` and no Cypher queries exist in any application file. The Neo4j integration appears to have been started but abandoned.

---

#### 3b. Data / Ingestion Pipeline — ADVOCATE

**Source documents:**
- Kenya Law Reports case PDFs (SCC judgments + precedent cases): stored in `backend/SCCCases/` (11 files verified in tree) and `backend/Repo/Res_ipsa_loquitur/`.
- Statute HTML files: 14 acts in `data/raw_statutes/`, downloaded from `new.kenyalaw.org` via `download_statutes.py`.
- Constitution PDF: `backend/MasterRules/TheConstitutionOfKenya.pdf`.
- Evidence Act PDF: `backend/MasterRules/Evidence Act.pdf`.

**Pipeline for case law** (`build_case_chunks.py`; also via API `POST /api/digest`):
1. Load PDF → extract text with PyMuPDF or pypdf; if scanned, route via OCR (ocrmypdf + pytesseract).
2. Call Groq LLM (`qwen/qwen3.8-27b`) with the full text to extract FIRAC components (`CaseLaw/firac.py`).
3. Store each FIRAC component as one or more overlapping text chunks in `CaseLawDB/case_law` via ChromaDB, using `intfloat/e5-base-v2` embeddings (`CaseLaw/ingester.py`).
4. Record page+bbox locations in `locations.db` for the PDF overlay.

**Pipeline for statutes** (`build_statute_chunks.py`):
1. Parse HTML statute files (BeautifulSoup or similar — exact parser NOT confirmed from files read).
2. Split by section with overlap; embed with `intfloat/e5-base-v2`; store in `StatutesDB/statutes`.

**Pipeline for Constitution** (via API `POST /api/digest/constitution` or `constitutionlogic/constitutionbreaker.py`):
1. Extract articles from PDF using PyMuPDF + pattern matching (`constitutionbreaker.py`).
2. Chunk each article at 1500 characters with 200-character overlap (`app.py:604` cites `CHUNK_SIZE` and `CHUNK_OVERLAP` imported from `constitutionbreaker.py`).
3. Embed with `intfloat/e5-base-v2`; store in `MasterRulesDB/master_rules` with `document_type=constitution`.

**Counts (from manifest and directory listing):**
- 14 statute HTML files; total section count: 913 sections across all acts (`manifest.json` section counts: SCC Act 52, Law of Contract 4, Sale of Goods 59, Consumer Protection 94, Limitation of Actions 46, Evidence Act 199, Civil Procedure 116, Bills of Exchange 101, KICA 198, Stamp Duty 126, Judicature 11, SCC Rules 37, Employment 96, Landlord & Tenant 16).
- 11 SCC/precedent case PDFs in `backend/SCCCases/`.
- Exact chunk counts in ChromaDB: NOT VERIFIED (would require running the application).

---

#### 3c. GraphRAG / Retrieval and Generation (End-to-End Trace)

**Trigger:** Advocate submits query `"Are M-Pesa statements admissible without a section 106B certificate?"`.

**Step 1 — Query Classification (`POST /api/qretrieve` → `Qclassifier.py`)**
- Route: `app.py:1251`, `@require_role("advocate","admin")`.
- `Qclassifier.py:669` calls `classify_legal_query()` which POSTs to Groq with the system prompt at `Qclassifier.py:30–421` (a Senior Kenyan Legal Research Specialist prompt defining routing modes `DIRECT_STATUTORY_LOOKUP`, `STRATEGIC_LEGAL_ANALYSIS`, `HYBRID_RETRIEVAL`).
- LLM parameters: temperature 0.1, max_tokens 1500, model `qwen/qwen3.8-27b` (`Qclassifier.py:449–465`).
- Output: JSON with `retrieval_mode`, `intents`, `target_components`, `legal_domains`, `entities`, `vector_query`, `citation_analysis`, `strategy_critique`.

**Step 2 — Case Law Retrieval (`Retriever.retrieve_relevant_cases()`)**
- Embed `vector_query` with `intfloat/e5-base-v2`; prefix: `"query: " + ...` (`Retriever.py:626`).
- BM25Okapi over all case chunks → top 50 by BM25 score (`Retriever.py:676–681`).
- ChromaDB vector search → top 50 (`Retriever.py:682–688`).
- RRF fusion (k=60) → top 30 candidates (`Retriever.py:694`).
- Cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2`) → sigmoid-normalised scores (`Retriever.py:710–721`).
- Group chunks by parent document (`case_identifier`); enrich with adjacent chunks; select best chunks per FIRAC component; return top 3 parent documents (`TOP_N_PARENTS = 3`, `Retriever.py:51`).

**Step 3 — Statutory Retrieval (`RulesRetriever.retrieve_relevant_rules()`)**
- Same hybrid pipeline applied to `StatutesDB/statutes`; Constitution fallback from `MasterRulesDB/master_rules`.
- Section injection guard: if classifier identifies explicit article/section numbers, those are retrieved by metadata filter first.
- Returns up to 5 statute groups.

**Step 4 — Generate Answer (`POST /api/generate-legal-answer` → `app.py:1406`)**
- Coverage check (`coverage.py:46`): classifies which named Acts are in/out of corpus.
- Build labelled source blocks (alias map `S1.1`, `C1.2`, etc.) from retrieval results.
- **System prompt** (verbatim from `app.py:1543–1578`):
  ```
  "You are a Kenyan legal research assistant. Output ONLY a single valid JSON object.
   No text before or after it.\n\nABSOLUTE RULES:\n1. Every legal statement MUST cite
   using exact aliases from the sources. Statute chunk aliases: [S1.1], [S1.2] etc.
   ...
   OUTPUT SCHEMA (all fields required): { source_review, premise_check, short_answer,
   applicable_law, case_law_discussion, application, jurisdiction_and_procedure, gaps,
   follow_up_questions, citations_used }"
  ```
- LLM call: `call_groq(messages)` with temperature 0, max_tokens 2500 (`groq_client.py:63–69`).
- JSON parse with brace-depth fallback (`app.py:1611–1625`).
- Citation verification (`citation_verifier.verify_and_clean()`): regex-extracts section/article/case references; checks each against retrieved source texts; logs failures to `logs/hallucinations.jsonl`; triggers one regeneration call if unsupported claims found.
- Build markdown answer string from parsed JSON fields.
- Persist to Firestore inside a Firestore transaction (`history_store.save_turn()`).
- Return JSON response with `answer`, `sources`, `alias_map`, `coverage`, `conversation_id`, `turn_id`.

**Fallback / refusal behaviour:**
- If `source_blocks` is empty (no retrieval results): returns `{"success": false, "error": "No relevant legal information retrieved"}` (HTTP 404) (`app.py:1534–1540`).
- If Groq JSON parse fails, returns raw text in `short_answer` field (`app.py:1631`).
- If all Groq keys exhausted, raises `GroqRateLimitError` (`groq_client.py:134`).
- Coverage notice prepended to user prompt when named Acts are not in corpus (`app.py:1583–1586`).
- `strategy_critique` in classifier output flags jurisdictional errors, temporal risks, and procedural hurdles — these appear in the classification JSON but are NOT explicitly surfaced in the final answer template.

---

#### 3d. Section 106B Evidence Admissibility (Vision-OCR Module)

**Location:** `backend/SCCVision/` package, routes registered via `sccvision_bp` in `router.py`.

**What is implemented:**

| Sub-component | Status | Description |
|---|---|---|
| Local OCR engine (RapidOCR) | Complete | `engine.py`: singleton `RapidOCR()` with warmup on startup; `service.py:extract_text()` returns lines with confidence, bounding boxes, reading-order sort |
| WhatsApp screenshot parser | Complete | `analysis.py`: classifies lines as contact name, messages (bubble detection by x-position), system events (encryption notices etc.) |
| M-Pesa screenshot parser | Complete | `analyst/mpssa.py:parse_mpesa_screenshot()` — deterministic extraction of transaction codes, amounts, dates, recipients; handles multiple known M-Pesa SMS layouts |
| Evidence classification | Complete | `analyst/mpssa.py:classify_evidence()` auto-detects `mpesa_screenshot`, `chat_screenshot`, or `other` |
| PII masking | Complete | `analyst/enrich.py:mask_pii()` — masks phone numbers and ID numbers before LLM |
| LLM analysis (s.106B) | Complete | `analyst/client.py:AnalystLLM.complete()` — Groq call returning `AnalystResult` Pydantic model; assesses contract validity, payment classification, admissibility notes |
| Deterministic verification | Complete | `analyst/verify.py` / `analyst/rules.py` — cross-checks M-Pesa codes vs chat messages, checks quote grounding, computes overall verdict |
| Date arithmetic | Complete | `analyst/dates.py:compute_date_checks()` — 3-year SCC limitation period calculation |
| Case-file API (new) | Complete | `/sccvision/evidence/process` and `/sccvision/case/analyze` endpoints |
| Old API endpoints | Deprecated (marked with `Deprecation: true` header) | `/sccvision/analyze` and `/sccvision/analyze-messages` |
| Certificates page | Partial/Stub | `frontend/app/litigant/certificates/page.tsx` exists; full content NOT read |

**Inputs:** PNG, JPG, JPEG, or WEBP image files (max 10 MB) uploaded via multipart form.

**Checks performed:**
- Section 106B admissibility: `applicable`, `inapplicable`, `uncertain` verdict from LLM.
- M-Pesa cross-match: verifies that transaction reference codes in M-Pesa screenshots appear in conversation messages.
- Quote grounding: checks that quoted text in LLM output is grounded in the OCR result.
- Payment classification: `confirmed_payment`, `payment_request`, `disputed_payment`, etc.
- Date checks: limitation period, future-dated facts.

**Outputs:** JSON with `analysis_input`, `result` (Pydantic model dump), `verification` (deterministic checks), `prompt_version`, `model`.

**Module usage:** SCC module only (`@require_auth` without role restriction, so any authenticated user may call it; litigants sign in via Google on the `/litigant/sign-in` route).

---

### SCC Module

Same data layer as shared above. Key additional point: the litigant evidence workflow stores evidence items client-side (in browser/React state), not in Firestore. The backend processes evidence on demand and returns analysis results; no Firestore write of evidence items was found in the reviewed code.

---

### Shared Components

**Authentication:** Firebase Authentication (Google + email/password). Backend verifies Firebase ID tokens on every protected request (`auth.py:49`). Frontend `AuthContext.tsx` wraps `onAuthStateChanged`.

**Groq LLM client:** `groq_client.py` — shared across both modules. Shuffled key rotation on 429/403; 2-round retry with 30s wait; strips `<think>…</think>` reasoning blocks from Qwen3 output.

---

#### 3e. Backend API Route Table

| Method | Path | Module | Auth required? | What it does | File |
|---|---|---|---|---|---|
| GET | `/api/health` | Shared | No | Health check | `app.py:48` |
| POST | `/api/chat` | Advocate | advocate, admin | Legacy RAG via `pixe.py` | `app.py:53` |
| GET | `/api/files` | Advocate | advocate, admin | List files in `Repo/` | `app.py:118` |
| POST | `/api/upload` | Admin | admin | Upload PDF to `Repo/` | `app.py:130` |
| GET | `/api/files/<filename>` | Advocate | advocate, admin | Serve PDF from `Repo/` | `app.py:162` |
| POST | `/api/digest` | Admin | admin | PDF → FIRAC → ChromaDB | `app.py:187` |
| POST | `/api/firac` | Advocate | advocate, admin | Extract FIRAC from PDF | `app.py:282` |
| POST | `/api/ingest-firac` | Admin | admin | Ingest FIRAC data into ChromaDB | `app.py:406` |
| POST | `/api/digest/constitution` | Admin | admin | Extract + embed Constitution | `app.py:525` |
| GET | `/api/digest/constitution/status` | Admin | **No** | Constitution embed status | `app.py:850` |
| DELETE | `/api/digest/constitution/reset` | Admin | admin | Delete all constitution chunks | `app.py:902` |
| GET | `/api/case-pdf` | Advocate | **No** | Stream case PDF by `case_id` | `app.py:955` |
| GET | `/api/chunks/<chunk_id>/location` | Shared | **No** | Chunk bounding-box from SQLite | `app.py:1071` |
| GET | `/api/documents/<doc_key>/chunks` | Shared | **No** | List all chunks for a doc path | `app.py:1104` |
| GET | `/api/cases/<case_identifier>/all-chunks` | Advocate | **No** | All case chunks from ChromaDB | `app.py:1129` |
| GET | `/api/statutes/<source_document>/all-chunks` | Advocate | **No** | All statute chunks from ChromaDB | `app.py:1162` |
| GET | `/api/rule-pdf` | Advocate | **No** | Stream statutory/constitution PDF | `app.py:1196` |
| POST | `/api/qretrieve` | Advocate | advocate, admin | Query classification + retrieval | `app.py:1251` |
| POST | `/api/generate-legal-answer` | Advocate | advocate, admin | Grounded LLM synthesis + history save | `app.py:1406` |
| POST | `/api/case-laws/upload` | Shared | advocate, admin | Upload case law file; write Firestore | `app.py:1848` |
| GET | `/api/case-laws/file/<filename>` | Shared | **No** ⚠️ | Serve case law file from `SCCCases/` | `app.py:1895` |
| GET | `/api/conversations` | Advocate | advocate, admin | List conversations (paginated) | `history_routes.py:43` |
| GET | `/api/conversations/<id>` | Advocate | advocate, admin | Get conversation + turns | `history_routes.py:62` |
| PATCH | `/api/conversations/<id>` | Advocate | advocate, admin | Rename or pin conversation | `history_routes.py:93` |
| DELETE | `/api/conversations/<id>` | Advocate | advocate, admin | Delete conversation (recursive) | `history_routes.py:129` |
| POST | `/api/conversations/<id>/regenerate-title` | Advocate | advocate, admin | Re-generate conversation title | `history_routes.py:148` |
| GET | `/sccvision/health` | SCC | **No** | SCCVision health check | `SCCVision/router.py:33` |
| POST | `/sccvision/extract` | SCC | Any authenticated | OCR extract text from image | `SCCVision/router.py:47` |
| POST | `/sccvision/analyze` | SCC | Any authenticated | (Deprecated) OCR + LLM analyze | `SCCVision/router.py:86` |
| POST | `/sccvision/analyze-messages` | SCC | Any authenticated | (Deprecated) Re-analyze OCR results | `SCCVision/router.py:153` |
| POST | `/sccvision/evidence/process` | SCC | Any authenticated | Classify + normalise evidence item | `SCCVision/router.py:189` |
| POST | `/sccvision/case/analyze` | SCC | Any authenticated | Full case-file admissibility analysis | `SCCVision/router.py:266` |

---

#### 3f. Frontend Pages / Routes

| Path | Module | Purpose | Main components | Backend endpoints called |
|---|---|---|---|---|
| `/` | Shared | Landing page (home) | `page.tsx` | None |
| `/auth/sign-in` | Advocate | Advocate email/password or Google sign-in | `auth/sign-in/page.tsx` | Firebase Auth SDK |
| `/auth/sign-up` | Advocate | Advocate registration | `auth/sign-up/page.tsx` | Firebase Auth SDK, `createUserDoc()` |
| `/auth/complete-profile` | Advocate | Username + role setup after OAuth | `auth/complete-profile/page.tsx` | `createUserDoc()` or `createLitigantDoc()` |
| `/dashboard` | Advocate | Role-aware tool launcher | `dashboard/page.tsx`, `UserMenu` | None (navigation only) |
| `/research` | Advocate | Main chat / RAG workspace | `ChatView.tsx` (wraps everything) | `/api/qretrieve`, `/api/generate-legal-answer`, `/api/conversations`, `/api/case-pdf`, `/api/chunks/*`, `/api/cases/*`, `/api/rule-pdf` |
| `/chat/[conversationId]` | Advocate | Load specific conversation | `chat/[conversationId]/page.tsx` | `/api/conversations/<id>` |
| `/constitution` | Admin only | Constitution ingestion management | `constitution/page.tsx` | `/api/digest/constitution`, `/api/digest/constitution/status` |
| `/firacChecker` | Admin only | Manual FIRAC extraction + ingestion | `firacChecker/page.tsx` | `/api/firac`, `/api/ingest-firac` |
| `/ingester` | Admin only | Document upload + bulk digest | `ingester/page.tsx` | `/api/upload`, `/api/files`, `/api/digest` |
| `/repository` | Advocate, Admin | Browse documents | `repository/page.tsx` | None (sub-routes) |
| `/repository/caselaws` | Advocate, Admin | Browse + upload case law files | `repository/caselaws/page.tsx` | `/api/case-laws/upload`, Firestore `CaseLaws` collection |
| `/repository/statutes` | Advocate, Admin | Browse statute HTML files | `repository/statutes/page.tsx` | UNCLEAR (not fully read) |
| `/litigant/sign-in` | SCC | Litigant Google sign-in | `litigant/sign-in/page.tsx` | Firebase Auth SDK |
| `/litigant` | SCC | Litigant landing | `litigant/page.tsx` | None |
| `/litigant/evidence` | SCC | List and upload evidence | `litigant/evidence/page.tsx` | `/sccvision/extract`, `listEvidence()` |
| `/litigant/analysis` | SCC | Evidence admissibility analysis | `litigant/analysis/page.tsx` | `/sccvision/case/analyze`, `/sccvision/evidence/process` |
| `/litigant/certificates` | SCC | Certificates view | `litigant/certificates/page.tsx` | UNCLEAR |

---

#### 3g. Authentication and Roles

**Sign-in methods:**
- Advocates: email/password or Google (`auth/sign-in/page.tsx`).
- Litigants: Google only (`litigant/sign-in/page.tsx`).
- Admin: set via `scripts/bootstrap_admin.py` or `scripts/set_role.py` (Firebase Admin SDK custom claim NOT used; role is in Firestore `users/{uid}.type`).

**Roles:**
| Role | Set by | Access |
|---|---|---|
| `advocate` | Self-registration (enforced by Firestore rule — client may only set `advocate` or `litigant`) | `/dashboard`, `/research`, `/chat/*`, `/repository/*`, plus litigant pages |
| `litigant` | Self-registration via Google | `/litigant/*` only |
| `admin` | `scripts/set_role.py` or `scripts/bootstrap_admin.py` (backend/Admin SDK, never client) | All pages including `/ingester`, `/firacChecker`, `/constitution` |

**Frontend enforcement (`access.ts:42–58`):**
- `AuthGuard.tsx` intercepts navigation; calls `canAccess(pathname, type)`.
- Admin-only: `/ingester`, `/firacChecker`, `/constitution`.
- Advocate+admin: `/research`, `/chat/*`, `/dashboard`, `/repository/*`.
- Litigant+all: `/litigant/*`.
- Default: deny → redirect to `homeRoute()`.

**Backend enforcement (`auth.py:89–114`):**
- `@require_role(*roles)` decorator: verifies Firebase ID token, loads Firestore profile, checks `profile.type` against allowed roles.
- Sets `g.uid`, `g.profile`, `g.user_type` for route handlers.
- Missing profile → 403 with `{"code": "PROFILE_REQUIRED"}`.
- `@require_auth` (less strict) used for SCCVision routes — allows any authenticated user regardless of role.

---

#### 3h. Main User Workflows

**ADVOCATE: Legal Research Query (Primary workflow)**

1. User navigates to `/research` → `ChatView` loads; calls `GET /api/conversations` to populate sidebar.
2. User types query and presses Enter.
3. `ChatView` generates `clientTurnId = uuid()` for idempotency.
4. `POST /api/qretrieve` with `{query, pinned_docs}` → runs `Qclassifier` (LLM classification) + `retrieve_relevant_cases` (hybrid retrieval on `CaseLawDB`) + optionally `retrieve_relevant_rules` (hybrid retrieval on `StatutesDB`/`MasterRulesDB`).
5. `POST /api/generate-legal-answer` with `{query, retrieval, classification, conversation_id, client_turn_id}` → coverage check → LLM synthesis → citation verification → save to Firestore.
6. Response rendered in `ChatView` as collapsible markdown sections with source cards; clicking a source opens the PDF in an overlay with highlighted bounding boxes.

Files: `frontend/app/components/ChatView.tsx`, `backend/app.py:1251,1406`, `backend/Qclassifier.py`, `backend/Retriever.py`, `backend/RulesRetriever.py`, `backend/groq_client.py`, `backend/history_store.py`.

**ADVOCATE: Load Previous Conversation**

1. User clicks a conversation in the sidebar → router navigates to `/chat/[conversationId]`.
2. `GET /api/conversations/<id>` returns conversation metadata + last 50 turns.
3. `ChatView` renders turns from history; each turn has `fromHistory=true` flag.

Files: `frontend/app/chat/[conversationId]/page.tsx`, `backend/history_routes.py:62`.

**SCC: Evidence Submission and Admissibility Analysis**

1. Litigant navigates to `/litigant/evidence` → sees uploaded items.
2. Litigant uploads WhatsApp screenshot or M-Pesa confirmation (image file).
3. `POST /sccvision/extract` → RapidOCR → structured text with contact/messages/events.
4. `POST /sccvision/evidence/process` with OCR result → evidence item classified + parsed (M-Pesa: transaction data extracted deterministically; chat: messages stored structured).
5. Litigant navigates to `/litigant/analysis` → submits the case file.
6. `POST /sccvision/case/analyze` with full case file → LLM analysis → deterministic verification → returns `AnalystResult` with admissibility verdict, payment classification, limitations check.

Files: `backend/SCCVision/router.py`, `backend/SCCVision/service.py`, `backend/SCCVision/analyst/`.

---

#### 3i. Human-in-the-Loop

**No explicit human-review or approval step exists in the implemented workflow.** The system generates legal analysis fully automatically and returns it to the user. There is no mechanism for a supervising lawyer or adjudicator to review, approve, or override AI output before it is shown. The `strategy_critique` field in the classifier JSON identifies risks but is not surfaced to the user in the current UI. The `premise_check` field in the answer flags legal premises not found in sources — this is surfaced in the answer text.

This is a known gap for a legal decision-support tool.

---

#### 3j. Implementation Status

| Feature | Module | Status | Evidence |
|---|---|---|---|
| Firebase Auth (email + Google) | Shared | Complete | `auth.py`, `AuthContext.tsx`, `firestore.ts` |
| Role-based access (advocate/litigant/admin) | Shared | Complete | `auth.py:89`, `access.ts`, `firestore.rules` |
| Query classification (Qclassifier) | Advocate | Complete | `Qclassifier.py` — full 3-mode routing |
| Hybrid case-law retrieval (BM25+vector+CE) | Advocate | Complete | `Retriever.py` — RRF fusion + cross-encoder |
| Hybrid statute retrieval | Advocate | Complete | `RulesRetriever.py` |
| Grounded answer synthesis | Advocate | Complete | `app.py:1406` — aliased sources, temperature 0 |
| Citation verification + hallucination logging | Advocate | Complete | `citation_verifier.py`, `logs/hallucinations.jsonl` |
| Coverage check (Acts in/out of corpus) | Advocate | Complete | `coverage.py` |
| Chat history persistence (Firestore) | Advocate | Complete | `history_store.py`, `history_routes.py` |
| Idempotent generate endpoint | Advocate | Complete | `app.py:1447–1464`, `history_store.check_idempotency()` |
| AI-generated conversation titles | Advocate | Complete | `history_store.submit_title_generation()` |
| PDF viewer with bbox highlight overlay | Advocate | Complete | `PdfViewer.tsx`, `app.py:1071`, `locations.db` |
| Skeleton loading UI | Advocate | Complete | `components/skeleton/` (10 components) |
| WhatsApp screenshot OCR | SCC | Complete | `SCCVision/service.py` |
| M-Pesa screenshot parser | SCC | Complete | `SCCVision/analyst/mpssa.py` |
| s.106B admissibility analysis | SCC | Complete | `SCCVision/analyst/pipeline.py` |
| Deterministic verification rules | SCC | Complete | `SCCVision/analyst/verify.py`, `rules.py` |
| Litigant evidence list page | SCC | Complete (UI) | `litigant/evidence/page.tsx` — renders status, type, file |
| Litigant analysis page | SCC | Partial | `litigant/analysis/page.tsx` exists but not fully read |
| Litigant certificates page | SCC | Unclear/Stub | `litigant/certificates/page.tsx` exists; contents not read |
| Constitution ingestion + embedding | Admin | Complete | `app.py:525`, `constitutionlogic/` |
| Neo4j knowledge graph | Planned/Abandoned | Not started | Binary present, driver not in requirements, no Cypher queries |
| FIRAC checker UI | Admin | Complete | `firacChecker/page.tsx` |
| Document ingester UI | Admin | Complete | `ingester/page.tsx` |
| Deployment configuration | Shared | Not started | No Dockerfile, no Procfile, no Railway/Vercel config |
| Human-in-the-loop review step | Shared | Not started | No review workflow found in code |
| Strategy critique display in UI | Advocate | Stub | `strategy_critique` in classifier JSON; not rendered in `ChatView` |

**TODOs / notable comments found:**
- `app.py:1897`: comment `"Serve a case law file from SCCCases/ (local, no auth — internal tool only)"` on `serve_case_law_file` — this route has no authentication.
- `SCCVision/router.py:148–149`: `Deprecation` and `Warning` headers on `/sccvision/analyze` and `/sccvision/analyze-messages` — these deprecated endpoints remain active.

---

## 4. TESTING AND EVALUATION EVIDENCE

### 4a. Automated Tests

**Backend (pytest)**

| File | Framework | What it covers | How to run |
|---|---|---|---|
| `backend/tests/test_history.py` | pytest + unittest.mock | `make_fallback_title`, `clean_generated_title`, `build_sources_for_history`, `save_turn` (new conv, follow-up, idempotency, client_turn_id), `get_or_create_conversation` (found/not-found/foreign), title generation lifecycle, `list_conversations` (order, pagination), `get_conversation` (turns in order, missing conv), rename/pin, delete, Flask endpoint tests via test client (200/400/404), `_size_guard` (oversize truncation), `check_idempotency` (complete/failed/missing), `save_turn` overwrite; Firestore-emulator integration tests (skipped without emulator) | `cd backend && PYTHONPATH=. pytest tests/test_history.py -v` |
| `backend/tests/test_sccvision_analyst.py` | pytest | `mask_phone`, `mask_id_number`, `mask_pii`, `extract_amounts`, `parse_mpesa_message`, `merge_ocr_results`, `build_analysis_input`, `build_integrity_flags`, `classify_evidence`, `find_codes`, `parse_mpesa_screenshot`, `check_mpesa_cross_match`, `check_quotes_grounded`, `compute_verdict`, `merge_followups`, `build_deterministic_followups`, `verify`, `analyze` and `analyze_case` with LLM stubbed | `cd backend && PYTHONPATH=. pytest tests/test_sccvision_analyst.py -v` |

**Frontend — Firestore Rules (Jest + Firebase Rules Testing)**

| File | Framework | What it covers | How to run |
|---|---|---|---|
| `firestore-tests/firestore.test.ts` | Jest + `@firebase/rules-unit-testing` | Firestore security rules: users create (schema enforcement, type restriction, atomic batch, `advocateVerified` must be false), users update (frozen fields), conversations read-own-only, write denied, public username reads | `firebase emulators:start --only firestore --project legalai-rules-test && cd firestore-tests && npm test` |

Tests have NOT been run as part of this audit (read-only).

**Other test files:**
- `backend/test_scc_pipeline.py` — present (not read; likely integration/regression test for the full pipeline).
- `backend/testdb.py`, `backend/validate_knowledge.py`, `backend/debug_firac.py` — utility/debug scripts, not automated tests.
- `backend/Const_IntepretationLaws/testBlackLaw.py` — test script for BlackLaw dictionary embedding (not in `tests/`).

---

### 4b. Evaluation Scripts and Metrics

| Item | Status |
|---|---|
| Hallucination log | Present: `backend/logs/hallucinations.jsonl` — records post-generation citations rejected by `citation_verifier.py`. Exact counts NOT read. |
| Retrieval accuracy metrics (F1, recall, MRR) | NOT FOUND |
| Gold question-answer evaluation set | NOT FOUND |
| Benchmark results (accuracy numbers) | NOT FOUND |
| Embedding run logs | Present: `backend/logs/embed_*.log` (5 files). Content not read. |
| Cross-encoder confidence threshold warning | Present in code: `LOW_CONFIDENCE_THRESHOLD = 0.4` in `Retriever.py:53`; a `warning` field is added to output when top result is below this threshold. |
| Latency tracking | Present: `latency_ms` stored in every Firestore turn document (`history_store.py:turn_data`). |
| Probe mode (pinned documents) | Present: `app.py:1270`, `Retriever.py:635–663` — restricts retrieval to specified documents, useful for targeted evaluation. |

---

### 4c. Baseline Comparison

No formal baseline comparison (plain LLM vs. vector-only RAG vs. GraphRAG) was found in the repository. The `PROBE_REPORT_admissibility.md`, `AUDIT_CHUNKING.md`, and `SYSTEM_ANALYSIS.md` files exist at the repo root (untracked) but their contents were not read during this audit.

---

### 4d. Input Validation and Error Handling

**Backend:**
- File uploads: `secure_filename` (werkzeug) used on all upload routes (`app.py:148`, `1863`); PDF/image extension validation; size limit check for SCCVision (10 MB, `router.py:65`); allowed extensions validated.
- API key: `get_groq_keys()` returns empty list → `generate_legal_answer` returns 500 with descriptive error (`app.py:1597`).
- Path traversal: `app.py:1046–1054` (case-pdf), `app.py:1230–1234` (rule-pdf), `app.py:1899–1901` (case-law file) — all resolve and validate against allowed root directories.
- Firestore document size guard: `_size_guard()` in `history_store.py:388–413` truncates oversized documents before write.
- Groq JSON parse failure: falls back to raw text (`app.py:1631`).
- Failed turn persistence: `_try_save_failed_turn()` saves `status="failed"` turns on any error path; retry/overwrite logic implemented.
- Rate-limit handling: `GroqRateLimitError` and `GroqPayloadTooLargeError` raised and caught.

**Logging:**
- `citation_verifier.py` appends rejected citations to `logs/hallucinations.jsonl`.
- `Retriever.py` logs CE-score statistics to stdout.
- `history_store.py` logs title generation failures.
- No structured application-level logging framework (e.g., Python `logging` module with handlers) is configured; most logging uses `print()`.

---

### 4e. Testing Gaps

| Gap | Module |
|---|---|
| No unit tests for `Qclassifier.py` (classification JSON output, key rotation, retry logic) | Advocate |
| No unit tests for `Retriever.py` or `RulesRetriever.py` (hybrid search, RRF fusion, cross-encoder) | Advocate |
| No unit tests for `citation_verifier.py` | Advocate |
| No unit tests for `coverage.py` | Advocate |
| No end-to-end integration tests for the full `/api/qretrieve` → `/api/generate-legal-answer` pipeline | Advocate |
| No load/stress tests for Groq API key rotation under concurrent requests | Shared |
| No tests for `app.py` route handlers (upload, digest, case-pdf, path-security) | Shared |
| `test_scc_pipeline.py` — not read; may cover some gaps | SCC/Advocate |
| No formal evaluation against labelled Kenyan legal Q&A gold set | Both |
| No tests for litigant sign-in flow or certificates page | SCC |
| Firestore emulator integration tests (`TestEmulatorIntegration`) skipped unless emulator is running | Advocate |

---

## 5. VERSION CONTROL

### Remote URL and Branch Information

| Item | Value |
|---|---|
| Remote URL | `git@github.com:Houstonsboy/Retreival-Augmented-Generation-AI.git` |
| Current branch | `master` |
| All branches | `backup`, `master` (local), `temp-fix` (local), `remotes/origin/master` |
| Tags / releases | None |

### Commit Statistics

| Item | Value |
|---|---|
| Total commits | 56 |
| First commit date | 2025-05-18 |
| Latest commit date | 2026-11-01 |

### 30 Most Recent Commits

| Date | Hash | Message |
|---|---|---|
| 2026-11-01 | `e96979e` | feat: Round 2 skeleton — viewport-anchored shimmer + unified Skeleton component |
| 2026-11-01 | `9f8a335` | feat: production skeleton loading for legal research workspace |
| 2026-11-01 | `4608dcb` | fix: navigate to dashboard after profile setup without requiring refresh |
| 2026-11-01 | `1ffd1be` | feat: Phase 2 frontend — history, routing, sidebar, idempotency |
| 2026-11-01 | `8c518dc` | feat: idempotent generate endpoint — cache hit skips LLM, retry overwrites failed turn |
| 2026-11-01 | `fea5b20` | docs: add Firestore security rules, composite indexes, and CHAT_HISTORY guide |
| 2026-11-01 | `b1a6a4c` | feat: add Flask Blueprint for chat history management endpoints |
| 2026-11-01 | `540f0c1` | feat: wire chat history into generate-legal-answer endpoint |
| 2026-11-01 | `989fd4b` | feat: add history persistence layer (history_store.py + tests) |
| 2026-11-01 | `641643b` | refactor: extract _groq closure to shared groq_client module |
| 2026-11-01 | `57edd66` | chore: ignore neo4j runtime data and binaries |
| 2026-11-01 | `211b511` | feat: redesign chat workspace UI with gold/cream theme and markdown rendering |
| 2026-10-31 | `0f36d8b` | Fix F: end-to-end test Amani/M-Pesa query; fix precision tagging for bare [S#] |
| 2026-10-31 | `e807d85` | Fix B: compute constitution bboxes via rapidfuzz word-alignment |
| 2026-10-31 | `f284295` | Fix A: replace <object> PDF embed with react-pdf + bbox overlay |
| 2026-10-31 | `0fc40a7` | Fix constitution chunk ID mismatch + bbox script |
| 2026-10-31 | `020581a` | Fix C/D/E: statute chunk aliases, citation precision, cluster widget |
| 2026-10-26 | `7892723` | Add Firebase Auth + Firestore RBAC |
| 2026-10-26 | `9401c38` | wire legalai-scc project config |
| 2026-10-26 | `2f2c677` | add Firebase Auth + Firestore with role-based access control |
| 2026-10-26 | `e737d54` | Hybrid retrieval: BM25+RRF+cross-encoder, StatutesDB knowledge switch, section injection guard |
| 2026-02-13 | `5a2cf88` | knowledge graph |
| 2026-02-12 | `578cf9b` | knowledge tester 1 |
| 2026-02-02 | `629cd79` | reconfigured the master rules to not have noise from small chunks that keep being retrieved |
| 2026-01-30 | `3aba7e4` | mit |
| 2026-01-30 | `d4fae83` | worked on the pdf rendering on the browser |
| 2026-01-30 | `a311fec` | we cooking, now Evidence Act is in DB |
| 2026-01-26 | `dfed1bf` | extracting the evidence act |
| 2026-01-21 | `3cd68b1` | output is on frontend |
| 2026-01-19 | `fb7afed` | improved accuracy by reducing base similarity and increasing [threshold] |

### Major Feature Milestones (by commits)

| Feature | Commit(s) |
|---|---|
| Initial RAG prototype | `0226564`–`cc0dc21` (2025-05-18) |
| Flask API + ChromaDB + FIRAC | `20062a3`, `354a0d7` (early 2026) |
| Hybrid retrieval (BM25+RRF+CE) + StatutesDB | `e737d54` (2026-10-26) |
| Firebase Auth + RBAC | `2f2c677`, `7892723` (2026-10-26) |
| ChatView UI redesign + PDF viewer + citation fixes | `020581a`–`0f36d8b` (2026-10-31) |
| Groq client shared module | `641643b` (2026-11-01) |
| Chat history + Firestore persistence + idempotency | `989fd4b`–`8c518dc` (2026-11-01) |
| Phase 2 frontend: sidebar, routing, history | `1ffd1be` (2026-11-01) |
| Skeleton loading UI | `9f8a335`, `e96979e` (2026-11-01) |

### .gitignore

A `.gitignore` is present. It excludes `.env`, `.env.local`, `*-service-account.json`, `*firebase-adminsdk*.json`, `serviceAccount*.json`, `venv/`, `__pycache__/`, `node_modules/`, `.next/`, `neo4j-community-*/data/`, and Firebase build artefacts.

**Verification — tracked sensitive files:** Running `git ls-files --cached | grep -E "adminsdk|service.account"` returned **no output** — the two Firebase service account JSON files present in `backend/` (`legalai-scc-firebase-adminsdk-fbsvc-73bde27853.json`, `legalai-service-account.json`) are **not** tracked in git. This is correct.

**Files currently tracked that may require scrutiny:**
- `backend/Database/legal_summaries_db_export_20260205_094943.json` — data export (tracked).
- `backend/Database/master_rules_db_export_20260205_094943.json` — data export (tracked).
- Several `*db_export_*.json` files in `backend/` and `backend/constitutionlogic/` (tracked).
- The ChromaDB binary files (`*.sqlite3`, `length.bin`) are tracked (modified in git status) — these are large binary artifacts that should ideally be in `.gitignore`.

### CI/CD

No `.github/workflows/` directory found. **No CI/CD pipeline exists.**

### README Quality

The root `README.md` describes the **early prototype** (Hugging Face + Mistral-7B, `pixe.py`-based). It does **not** describe the current system (Flask API, Firebase, Groq, Next.js frontend, ChromaDB knowledge bases, role-based access). The setup instructions in the README will not reproduce the current application. The frontend `README.md` is the default Next.js template.

---

## 6. DEVIATIONS AND RISKS

### Stack Corrections

| Your stated stack item | Actual finding |
|---|---|
| "Neo4j knowledge graph built from Kenya Law Reports" | **INCORRECT.** Neo4j 5.18.0 binary is present but the application does NOT use Neo4j at runtime. No Neo4j Python driver in `requirements.txt`. No Cypher queries in any application file. A commit titled "knowledge graph" (2026-02-13) exists but was not integrated. The system uses ChromaDB (vector store) + BM25 (keyword) hybrid retrieval — there is no graph traversal component. |
| "vision-OCR pipeline" | **PARTIALLY CORRECT — but technology differs.** No cloud-based vision API (Google Vision, Azure OCR) is used. The system uses **RapidOCR** (local CPU-based, ONNX runtime), not a cloud OCR API. |
| "Flask backend" | Confirmed: Flask 3.1.2. |
| "Next.js frontend" | **Confirmed but version differs from typical:** Next.js 16.0.3, React 19.2.0 — these are cutting-edge/pre-release versions. |
| "Firebase data layer" | Confirmed: Firebase Auth + Firestore. Firebase Storage bucket is configured but no Storage reads/writes found in application code. |
| "planned deployment on Railway" | **NOT CONFIRMED.** No `railway.json`, `railway.toml`, or Nixpacks configuration found. The repo has a `.firebase/` cache (Firebase Hosting cache), suggesting Firebase Hosting was tried. No deployment configuration is present for any platform. |

### Hardcoded Values / Unsafe Patterns

1. **`app.py:1897` — Unauthenticated file serving:** `GET /api/case-laws/file/<filename>` has no `@require_role` or `@require_auth` decorator. The route comment states "internal tool only." If the server is exposed externally, any HTTP client can download files from `backend/SCCCases/` without authentication. Path-traversal protection is implemented, but authentication is absent.

2. **Multiple unauthenticated read routes:** `/api/case-pdf`, `/api/chunks/<id>/location`, `/api/documents/<doc_key>/chunks`, `/api/cases/<case_identifier>/all-chunks`, `/api/statutes/<source_document>/all-chunks`, `/api/rule-pdf`, `/api/digest/constitution/status` have no auth. These expose chunk text and PDF files from the knowledge base to any caller.

3. **`Qclassifier.py:507` — partial key printed in logs:** `f"[DEBUG] API Key (first 10 chars): {GROQ_API_KEY[:10]}..."` — 10 characters of the API key are printed to stdout on 401 error. Combined with server log access, this reduces key security.

4. **ChromaDB `.sqlite3` and `length.bin` binary files tracked in git:** These are large binary artefacts and should be excluded. The git status shows them as modified.

5. **Firebase service account JSON files in backend directory:** `legalai-scc-firebase-adminsdk-fbsvc-73bde27853.json` and `legalai-service-account.json` exist on disk at `backend/` and are referenced by `FIREBASE_SERVICE_ACCOUNT_PATH`. They are correctly excluded by `.gitignore` but their presence on-disk and name being visible in directory listings is a housekeeping risk.

6. **`app.py:38` — CORS_ORIGIN defaults to `http://localhost:3000`:** Acceptable for development; must be overridden for any production deployment.

### Known Bugs / Fragile Areas

1. **BM25 lazy initialisation is not thread-safe under concurrent requests:** `Retriever.py:65–79` and `RulesRetriever.py:59–73` use a module-level dict and check `if _CASES_BM25["index"] is not None` without a lock. Under Flask's default threaded server, two concurrent first requests could both attempt to load the index simultaneously.

2. **`Qclassifier.py` has its own key rotation logic (lines 487–502) that duplicates `groq_client.py`:** The refactor to `groq_client.py` (`641643b`) was applied to `app.py` and `history_store.py` but `Qclassifier.py` still uses its own inline rotation. The comment in `groq_client.py:11` explicitly notes "Do NOT change Qclassifier.py's Groq logic."

3. **ChromaDB dimension mismatch risk:** If the embedding model is changed without rebuilding the stores, queries will fail with a dimension error. No schema migration guard is present.

4. **`history_store.py:519` — cursor-based pagination:** `q.start_after([cursor_dt])` is used for Firestore pagination; this requires a composite index on `(updated_at DESC)`. If the index is missing (e.g., after a fresh project setup), reads will fail.

5. **`app.py:70–109` `/api/chat` (legacy pixe.py endpoint):** Captures and discards stdout; uses an older retrieval path that is not hybrid. This route is `@require_role("advocate","admin")` but its underlying `run_rag_pipeline()` from `pixe.py` does not use the same ChromaDB knowledge bases as the main pipeline.

6. **`Retriever.py:626` query prefix:** Embedding is done with prefix `"query: " + query_params['vector_query']`; corpus documents must have been embedded with `"passage: " + text` for the asymmetric E5 model to work correctly. The ingestion scripts must enforce this — this was NOT verified from the ingestion scripts in this audit.

---

## 7. QUESTIONS FOR ME

The following items could not be determined from the repository alone:

1. **Neo4j status:** Was the Neo4j knowledge graph (commit `5a2cf88`, 2026-02-13) intentionally abandoned, or is it in a separate branch/repository? If it is part of the final submission, clarify where the Neo4j integration code is.

2. **"GraphRAG" label:** The project is described as a "GraphRAG" system, but no graph traversal (graph database queries, node/relationship-based reasoning) is present in the implemented code. The retrieval is a hybrid vector+BM25 approach over ChromaDB. Can you clarify what the "graph" component refers to in the submitted system?

3. **Dataset sizes:** How many total case law PDFs were actually ingested into `CaseLawDB`? How many statute chunks are in `StatutesDB`? The ChromaDB `.sqlite3` files are binary and their document counts were not read.

4. **Deployment:** Is the application currently deployed anywhere (Firebase Hosting, Railway, Render, localhost)? Who tested it and in what environment?

5. **Litigant certificates page:** The `/litigant/certificates/page.tsx` file exists but was not fully read. What functionality does it expose?

6. **SCCVision — Analyst LLM prompt:** The `AnalystLLM` in `SCCVision/analyst/client.py` was read to show its structure but the full system prompt for the s.106B analysis was not read. For the report, include the analyst's system prompt text and the Pydantic output schema.

7. **BlackLaw Dictionary DB:** `backend/Const_IntepretationLaws/BlackLawDictionaryDB/` and `blacklaw_export_for_neo4j.json` suggest a legal dictionary embedding. Is this used in production retrieval, or was it an experiment?

8. **`backend/data/processed/statutes/`:** This directory is referenced in `config.py:9` and `coverage.py:14`. What files are in it? Is there a processed manifest separate from `data/raw_statutes/manifest.json`?

9. **`agentrouter.py`:** This file is present in `backend/` but was not read. What is its purpose?

10. **Testing performed by humans:** Was any manual user-testing or formal evaluation (e.g., with real SCC cases or legal professionals) conducted? If so, any results or observations should be documented separately for Chapter 5.

11. **`PROBE_REPORT_admissibility.md`, `AUDIT_CHUNKING.md`, `SYSTEM_ANALYSIS.md`:** These untracked files at the repo root may contain important evaluation data. Their contents should be reviewed for inclusion in the Chapter 5 evaluation section.

12. **`Const_IntepretationLaws/testBlackLaw.py` and `metadata_field_report.json`:** Are these part of the submitted system or development-only utilities?

---

*End of SYSTEM_AUDIT.md*
