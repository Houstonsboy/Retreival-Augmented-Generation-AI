# PROBE REPORT — Self-Admissibility Evidence Module
*Generated: 2026-11-02 | Read-only investigation of the LegalAI repository*

---

## Summary (≤ 10 lines)

The existing Flask/Next.js system is built for trained legal advocates, not citizens: the only roles are `"advocate"` and `"admin"` — a `"litigant"` role does not exist anywhere. The core OCR pipeline (PyMuPDF + pytesseract, Tesseract 5.5 installed, `eng` only) is already proven in `CaseLaw/firac.py` and can be lifted almost verbatim for evidence extraction. **Evidence Act s.106B (including subsections 106A–106I) is already chunked in both the `statutes` collection (213 chunks) and `master_rules` (245 Evidence Act chunks)** — so the RAG knowledge base needs no new ingestion for the compliance gate. The `law_of_contract_act` is only 6 chunks and covers Cap 23's thin scope (incorporates Indian Contract Act 1872); contract elements must be identified by LLM from the submitted document text, not retrieved. The shared `call_groq()` wrapper in `groq_client.py` already accepts an arbitrary `model=` parameter — switching to `llama-3.3-70b-versatile` requires one argument. **Seven backend routes expose database content with no authentication decorator** — this is the most urgent risk. No PDF-generation library (reportlab, weasyprint, docxtpl) is installed; the certificate output will need to be added. A file-size limit (`MAX_CONTENT_LENGTH`) is absent from Flask. Of the 7 wireframe screens, only screen 1 (login) fully exists; screen 2 (litigant dashboard) is partial; screens 3–7 are missing.

---

## Part A — Backend Findings

### A1. Project Tree and Entry Point

```
backend/
├── app.py                        ← Flask entry point (python app.py, debug=True, 0.0.0.0:5000)
├── auth.py                       ← Firebase token verification + @require_role decorator
├── groq_client.py                ← Shared Groq API wrapper (call_groq, get_groq_keys, GROQ_MODEL)
├── config.py                     ← Path constants + collection names
├── history_store.py              ← Firestore conversation/turn persistence
├── history_routes.py             ← Blueprint: /api/conversations/* (registered in app.py:39)
├── Qclassifier.py                ← Query classification (STRATEGIC / DIRECT / HYBRID)
├── Retriever.py                  ← Case-law retrieval (BM25 + dense + RRF + cross-encoder)
├── RulesRetriever.py             ← Statute/Constitution retrieval (same hybrid)
├── citation_verifier.py          ← Post-generation hallucination check
├── coverage.py                   ← Coverage check (which acts are/aren't in knowledge base)
├── CaseLaw/
│   ├── firac.py                  ← PDF→text (PyMuPDF + pytesseract OCR) + FIRAC LLM extraction
│   └── ingester.py               ← ChromaDB ingestion for case-law chunks
├── constitutionlogic/
│   ├── constitutionbreaker.py    ← Constitution PDF→JSON article extractor
│   ├── constitution_digester.py  ← Constitution JSON→ChromaDB (master_rules)
│   └── evidencedigester.py       ← Evidence Act PDF→ChromaDB (master_rules)
├── digester.py                   ← Langchain-based generic PDF→ChromaDB digester
├── pixe.py                       ← Legacy RAG pipeline (called by /api/chat)
├── build_case_chunks.py          ← One-off: case PDF→JSONL + locations.db
├── build_statute_chunks.py       ← One-off: statute JSONL chunking
├── embed_knowledge.py            ← One-off: JSONL→ChromaDB embedding
├── requirements.txt
├── .env                          ← 3 variables (names only — see A3)
├── legalai-scc-firebase-adminsdk-fbsvc-73bde27853.json   ← Firebase service account
├── legalai-service-account.json                           ← Firebase service account (duplicate?)
├── locations.db                  ← SQLite: chunk→page+bbox mapping
├── logs/hallucinations.jsonl     ← Citation verification failures
└── [ChromaDB dirs, Repo/, MasterRules/, SCCCases/ — omitted per request]
```

**Entry point:** `backend/app.py`, started with `python app.py` (last line: `app.run(host='0.0.0.0', port=5000, debug=True)`). No WSGI config or factory pattern — single global `app` object.

---

### A2. All Registered Routes

**Blueprint `history_bp`** registered at `app.py:39`; all routes `@require_role("advocate","admin")`.

| # | Method | Path | Handler file:approx_line | Auth |
|---|--------|------|--------------------------|------|
| 1 | GET | `/api/health` | app.py:45 | **NONE** |
| 2 | POST | `/api/chat` | app.py:50 | `@require_role("advocate","admin")` |
| 3 | GET | `/api/files` | app.py:115 | `@require_role("advocate","admin")` |
| 4 | POST | `/api/upload` | app.py:127 | `@require_role("admin")` |
| 5 | GET | `/api/files/<filename>` | app.py:159 | `@require_role("advocate","admin")` |
| 6 | POST | `/api/digest` | app.py:184 | `@require_role("admin")` |
| 7 | POST | `/api/firac` | app.py:279 | `@require_role("advocate","admin")` |
| 8 | POST | `/api/ingest-firac` | app.py:403 | `@require_role("admin")` |
| 9 | POST | `/api/digest/constitution` | app.py:522 | `@require_role("admin")` |
| 10 | GET | `/api/digest/constitution/status` | app.py:847 | **NONE** ⚠️ |
| 11 | DELETE | `/api/digest/constitution/reset` | app.py:899 | `@require_role("admin")` |
| 12 | GET | `/api/case-pdf` | app.py:952 | **NONE** ⚠️ |
| 13 | GET | `/api/chunks/<chunk_id>/location` | app.py:1068 | **NONE** ⚠️ |
| 14 | GET | `/api/documents/<doc_key>/chunks` | app.py:1101 | **NONE** ⚠️ |
| 15 | GET | `/api/cases/<case_identifier>/all-chunks` | app.py:1126 | **NONE** ⚠️ |
| 16 | GET | `/api/statutes/<source_document>/all-chunks` | app.py:1159 | **NONE** ⚠️ |
| 17 | GET | `/api/rule-pdf` | app.py:1193 | **NONE** ⚠️ |
| 18 | POST | `/api/qretrieve` | app.py:1248 | `@require_role("advocate","admin")` |
| 19 | POST | `/api/generate-legal-answer` | app.py:1393 | `@require_role("advocate","admin")` |
| 20 | POST | `/api/case-laws/upload` | app.py:1835 | `@require_role("advocate","admin")` |
| 21 | GET | `/api/case-laws/file/<filename>` | app.py:1882 | **NONE** ⚠️ |
| 22 | GET | `/api/conversations` | history_routes.py:43 | `@require_role("advocate","admin")` |
| 23 | GET | `/api/conversations/<conv_id>` | history_routes.py:62 | `@require_role("advocate","admin")` |
| 24 | PATCH | `/api/conversations/<conv_id>` | history_routes.py:93 | `@require_role("advocate","admin")` |
| 25 | DELETE | `/api/conversations/<conv_id>` | history_routes.py:129 | `@require_role("advocate","admin")` |
| 26 | POST | `/api/conversations/<conv_id>/regenerate-title` | history_routes.py:148 | `@require_role("advocate","admin")` |

**Auth mechanism:** `auth.py` `require_role(*roles)` decorator. Extracts `Authorization: Bearer <token>`, calls `firebase_admin.auth.verify_id_token()`, reads `users/{uid}` from Firestore, checks `profile["type"] in roles`. Sets `g.uid`, `g.profile`, `g.user_type` for handler use.

---

### A3. Config & Environment Variables

**Backend `.env` — variable NAMES only:**
```
GROQ_API_KEY
FIREBASE_SERVICE_ACCOUNT_PATH
CORS_ORIGIN
```

**Where each is read:**

| Variable | Read in | Usage |
|----------|---------|-------|
| `GROQ_API_KEY` | `groq_client.py:28` (and any `GROQ_API_KEY_*` variants) | Groq completions API key; rotated on 429 |
| `GROQ_API_KEY` | `Qclassifier.py:14` (hardcoded list: `GROQ_API_KEY`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3`) | Classifier has its own key list |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | `auth.py:20` | Path to service account JSON for Admin SDK init |
| `CORS_ORIGIN` | `app.py:36` | Single origin string passed to `flask_cors.CORS(app, origins=...)` |

**Frontend `.env.local` — variable NAMES only:**
```
NEXT_PUBLIC_FIREBASE_API_KEY
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN
NEXT_PUBLIC_FIREBASE_PROJECT_ID
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID
NEXT_PUBLIC_FIREBASE_APP_ID
NEXT_PUBLIC_API_URL
```
Read in: `frontend/app/lib/firebase.ts` (Firebase config) and `frontend/app/lib/api.ts:3` (`API_BASE`).

---

### A4. Firebase

**Initialisation (`auth.py:14–33`):**
```python
firebase_admin.initialize_app(
    credentials.Certificate(sa_path),
    {'storageBucket': 'legalai-scc.firebasestorage.app'}
)
```
Reads `FIREBASE_SERVICE_ACCOUNT_PATH` env var. Two service account JSON files exist in `backend/`:
- `legalai-scc-firebase-adminsdk-fbsvc-73bde27853.json`
- `legalai-service-account.json`

**Services imported and actually used:**

| Service | Import | Actually called? |
|---------|--------|-----------------|
| Firebase Auth | `firebase_admin.auth` (via `auth.py:5`) | YES — `verify_id_token()` on every protected request |
| Firestore | `firebase_admin.firestore` (via `auth.py:5`, `app.py:13`, `history_store.py:21`) | YES |
| Firebase Storage | `firebase_admin.storage` (via `app.py:13`: `from firebase_admin import ... storage as fb_storage`) | **NO** — imported but `fb_storage` is never called anywhere in `app.py` |

**Firestore collections (all reads/writes in LIVE_RUNTIME):**

| Collection path | Shape | Written by | Read by |
|----------------|-------|------------|---------|
| `users/{uid}` | `{username, email, type, authProvider, advocateVerified, createdAt, updatedAt}` | frontend `firestore.ts` | `auth.py:56` (profile check), `history_store.py`, frontend `AuthContext` |
| `usernames/{username}` | `{uid}` | frontend `firestore.ts` | frontend uniqueness check |
| `users/{uid}/conversations/{cid}` | `{title, title_source, title_generated_at, created_at, updated_at, turn_count, last_query_preview, pinned, schema_version}` | `history_store.py` | `history_routes.py` |
| `users/{uid}/conversations/{cid}/turns/{tid}` | `{status, query, answer_markdown, answer_structured, sources, alias_map, coverage, retrieval_summary, unverified_claims_removed_count, model, latency_ms, error, error_code, turn_index, created_at}` | `history_store.py` | `history_routes.py` |
| `CaseLaws` (top-level) | `{title, fileName, fileType, localPath, uploadedBy, uploadedByUid, uploadedAt, fileSizeBytes}` | `app.py:1863` (upload_case_law) | frontend `repository/caselaws/page.tsx` (Firestore listener) |

---

### A5. LLM Layer

**Shared wrapper: `backend/groq_client.py`**
- `GROQ_MODEL = "qwen/qwen3.8-27b"` (line 20)
- `call_groq(msgs, *, model=GROQ_MODEL, temperature=0, max_tokens=2500, timeout=60)` — accepts model override
- `get_groq_keys()` — discovers all `GROQ_API_KEY` + `GROQ_API_KEY_*` vars from environment
- Key rotation on HTTP 429; sleeps 30s if all keys exhausted

**Call sites in LIVE_RUNTIME:**

| Site | File:line | Model | temp | max_tokens | Notes |
|------|-----------|-------|------|------------|-------|
| Query classification | `Qclassifier.py:449` | `qwen/qwen3.8-27b` | 0.1 | 1500 | Does NOT use `call_groq()`; own `requests.post` + own key list |
| Legal answer synthesis | `app.py:1595` | `qwen/qwen3.8-27b` | 0 | 2500 | `call_groq(messages)` |
| Citation regen | `app.py:1650` | `qwen/qwen3.8-27b` | 0 | 2500 | closure `_regen()` → `call_groq(regen_msgs)` |
| Background title | `history_store.py:441` | `qwen/qwen3.8-27b` | 0.2 | 300 | `call_groq(...)`, ThreadPoolExecutor |

**`CaseLaw/firac.py` — anomaly:** imports `openai` and calls `requests.post` to `https://api.groq.com/openai/v1/chat/completions` directly (not via `call_groq()`). Uses `GROQ_API_KEY` from env directly (line 24). This is not in the shared rotation pool.

**Prompt templates:** Inline multi-line strings in each handler — no separate templates file. The synthesis system prompt lives at `app.py:1530–1565` (10-field JSON schema). The FIRAC extraction prompt is in `CaseLaw/firac.py` (~line 120–200).

**Structured output handling:** `call_groq()` returns raw text with `<think>…</think>` stripped. JSON is extracted by `json.loads()` on the response, with retry on `json.JSONDecodeError`.

---

### A6. Existing Retrieval

**ChromaDB collections (chromadb 1.3.5, PersistentClient):**

| Collection | Path | Count | Embedding model | Metric | Relevant to new module? |
|------------|------|-------|----------------|--------|------------------------|
| `case_law` | `backend/CaseLawDB/` | 78 | intfloat/e5-base-v2 | cosine | Indirectly (M-Pesa / WhatsApp case precedent) |
| `statutes` | `backend/StatutesDB/` | 1 331 | intfloat/e5-base-v2 | cosine | **YES — s.106B chunks present** |
| `master_rules` | `backend/constitutionlogic/MasterRulesDB/` | 619 | intfloat/e5-base-v2 | cosine | **YES — s.106B chunks present** |
| `legal_summaries` | `backend/LegalSummariesDB/` | 40 | — | — | INACTIVE |

**Evidence Act s.106B — already in knowledge base (verified by live query):**
- `statutes` collection: 213 `evidence_act` chunks; articles `106`, `106A`, `106B`, `106C`, `106D`, `106E`, `106F`, `106G`, `106H`, `106I` all present.
- `master_rules` collection: 245 Evidence Act chunks including s.106B subsections (also confirmed by live query).
- **No new ingestion is needed for the 106B compliance gate.**

**Law of Contract Act (Cap 23) — limited coverage:**
- `statutes` collection: only **6 chunks** for `law_of_contract_act`.
- The Act is a shell statute (s.2 applies English contract law to Kenya; s.4 applies the Indian Contract Act 1872). The actual contract doctrine (offer, acceptance, consideration, intention) lives in common law, not the Act text.
- **Contract element extraction must be done by LLM over the uploaded document, not by RAG retrieval of Cap 23.**

**Neo4j:** No live connection anywhere in LIVE_RUNTIME or LIVE_BUILD. `export_blacklaw_to_json.py` references a `neo4j-java` Maven project for a graph of Black's Law Dictionary terms, but that Java project is not present in this repository and no Python neo4j driver is installed.

---

### A7. File Handling

**Existing upload endpoints:**

| Endpoint | Field name | Allowed ext | Saved to | Max size | Auth |
|----------|------------|------------|---------|---------|------|
| `POST /api/upload` | `file` | `.pdf` only (app.py:142) | `backend/Repo/` | **None set** ⚠️ | admin |
| `POST /api/case-laws/upload` | `file` | `.pdf`, `.docx` (app.py:1832) | `backend/SCCCases/` | **None set** ⚠️ | advocate/admin |

**No `MAX_CONTENT_LENGTH` is set on the Flask app anywhere in `app.py`.** Without this, Flask accepts unlimited upload sizes, constrained only by OS/reverse-proxy.

**Upload pattern:** `request.files['file']` → `file.read()` into memory → `open(path, 'wb').write(data)`. No streaming; the full file is buffered in RAM.

**Temp files:** No `tempfile.mkstemp()`/`NamedTemporaryFile` usage in existing upload routes. Files are written directly to destination directories. `CaseLaw/firac.py` uses `tempfile.mkdtemp()` for intermediate images during OCR.

**Security:** Path traversal protection via `werkzeug.utils.secure_filename` (app.py:1855) and explicit root-check (`in_allowed`, app.py:1044–1049). New upload endpoints should follow the same pattern.

---

### A8. Existing Evidence/OCR/106B Code

**OCR pipeline — proven and reusable:**

`CaseLaw/firac.py:45–95` — `extract_text_from_pdf(file_path, use_ocr_threshold=50)`:
- Opens with `fitz.open()` (PyMuPDF)
- Per page: tries `page.get_text()`; if `< 50 chars`, renders at 2× zoom → PIL Image → `pytesseract.image_to_string(img, lang='eng', config='--oem 3 --psm 6')`
- Falls back to empty string if OCR fails

Same function is duplicated in `constitutionlogic/evidencedigester.py:32–62` and `constitutionlogic/constitution.py:80–119`. All three are identical implementations.

**106B references found in codebase:**
- `backend/history_store.py:47` — example title string `"M-Pesa Records as Evidence (s.106B)"` in `_TITLE_SYSTEM_PROMPT`
- `backend/validate_knowledge.py:74–88` — test query `"WhatsApp messages electronic evidence admissibility certificate Evidence Act 106B"` with expected retrieval of `("evidence_act", "106B")`
- ChromaDB (live): s.106B text confirmed present in both `statutes` and `master_rules` collections

**Nothing started for the new module:** no `admissibility.py`, no `evidence_upload.py`, no certificate template, no supplementary questionnaire logic.

---

### A9. Dependencies

**Python packages — installed vs. needed:**

| Package | Installed version | In requirements.txt? | Needed for module? | Notes |
|---------|------------------|---------------------|-------------------|-------|
| pytesseract | 0.3.13 | YES | YES | OCR wrapper |
| PyMuPDF (fitz) | **1.26.6** | YES (pinned 1.24.14 — mismatch) | YES | PDF text + image extraction |
| pypdf | 6.3.0 | YES | Optional | Alternative PDF reader |
| Pillow | **12.0.0** | YES (pinned 11.1.0 — mismatch) | YES | Image handling for OCR |
| python-docx | 1.2.0 | YES | Maybe | DOCX evidence files |
| groq | 0.35.0 | YES | Indirectly | SDK installed; live code uses `requests` directly |
| firebase-admin | 7.7.0 | YES | YES | Auth + Firestore persistence |
| langchain | 1.0.7 | YES | No | Not needed for new module |
| chromadb | 1.3.5 | YES | YES (read only) | Query s.106B and Cap 23 chunks |
| sentence-transformers | 5.1.2 | YES | YES | Embedding for 106B retrieval |
| flask | 3.1.2 | YES | YES | |
| flask-cors | 6.0.1 | YES | YES | |
| openai | **2.9.0** | **NOT in requirements.txt** ⚠️ | No (use call_groq) | Used only by firac.py |
| rapidfuzz | 3.14.6 | **NOT in requirements.txt** ⚠️ | No | Used by compute_constitution_bboxes.py |
| pdfplumber | — | NOT installed | Optional | Not needed; PyMuPDF covers the use case |
| pdf2image | — | NOT installed | NO | Would need poppler; use fitz instead |
| opencv-python | — | NOT installed | NO | Not needed |
| reportlab | — | **NOT installed** | **NEEDED** | For PDF certificate generation |
| weasyprint | — | NOT installed | Alternative | Heavy HTML→PDF; prefer reportlab |
| docxtpl | — | NOT installed | Alternative | DOCX template approach |

**System binaries:**

| Binary | Installed? | Version | Notes |
|--------|-----------|---------|-------|
| tesseract | **YES** | 5.5.0 | Languages: `eng`, `osd` only — no `swa` (Swahili) |
| pdftoppm (poppler) | **NOT installed** | — | `poppler-utils` package absent; use `fitz` rendering instead |
| ocrmypdf | YES (in venv) | 16.5.0 | Used by build_case_chunks.py only (LIVE_BUILD) |

---

### A10. CORS

```python
# app.py:36–37
_CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "http://localhost:3000")
CORS(app, origins=_CORS_ORIGIN)
```

- Single-value string (not a list). If `CORS_ORIGIN` contains a comma-separated list, flask-cors will treat it as a single malformed origin — only tested one origin.
- Default fallback `http://localhost:3000` means the backend works in development without any env var.
- No `supports_credentials`, `allow_headers`, or `expose_headers` overrides — flask-cors defaults apply.
- There is no per-blueprint CORS override. A `litigant` origin (e.g. a citizen-facing subdomain) must be added to `CORS_ORIGIN`.

---

## Part B — Frontend Findings

### B1. Framework and Tech Stack

| Property | Value |
|----------|-------|
| Next.js version | **16.0.3** |
| Router | **App Router** (`app/` directory, all pages are Server or `"use client"` components) |
| TypeScript | YES (`tsconfig.json` present, strict mode) |
| Styling | **Tailwind CSS v4** (`@tailwindcss/postcss`) — utility-first, no component library |
| UI components | **None** (no shadcn/ui, no Radix, no Headless UI) — all components are hand-written |
| Icons | **lucide-react 0.562.0** |
| PDF rendering | **react-pdf 7.7.3** (via `PdfViewer.tsx`) |
| Markdown | **react-markdown 10.1.0** + `remark-gfm` |
| State management | React `useState`/`useContext` only — no Zustand, Redux, Jotai |
| Data fetching | Native `fetch` via `apiFetch` wrapper — no SWR, React Query, Axios |
| Forms | Uncontrolled / manual `useState` — no react-hook-form, no Zod |
| Auth | Firebase JS SDK 10.12.0 |

---

### B2. Pages / Routes

| Route | File | Purpose | Role access |
|-------|------|---------|-------------|
| `/auth/sign-in` | `app/auth/sign-in/page.tsx` | Email/password + Google sign-in | Public |
| `/auth/sign-up` | `app/auth/sign-up/page.tsx` | Email/password + Google registration | Public |
| `/auth/complete-profile` | `app/auth/complete-profile/page.tsx` | Username selection after OAuth | Public (required after sign-up) |
| `/dashboard` | `app/dashboard/page.tsx` | Hub: Chat + Repository tiles | advocate + admin |
| `/` (root) | `app/page.tsx` | ChatView — main legal research workspace | advocate + admin |
| `/chat/[conversationId]` | `app/chat/[conversationId]/page.tsx` | ChatView with loaded history | advocate + admin |
| `/repository` | `app/repository/page.tsx` | File browser hub | advocate + admin |
| `/repository/caselaws` | `app/repository/caselaws/page.tsx` | Upload + browse case law PDFs/DOCX | advocate + admin |
| `/repository/statutes` | `app/repository/statutes/page.tsx` | Browse statute chunks | advocate + admin |
| `/ingester` | `app/ingester/page.tsx` | Document upload + digest trigger | **admin only** |
| `/firacChecker` | `app/firacChecker/page.tsx` | FIRAC extraction test tool | **admin only** |
| `/constitution` | `app/constitution/page.tsx` | Constitution ingestion admin | **admin only** |

**No "litigant" pages exist.** No pages are gated to a `litigant` role because the role does not exist.

**Role gating:** `AuthGuard.tsx` reads `profile.type` from the `AuthContext` (sourced from Firestore `users/{uid}.type`). `ADMIN_ONLY_PATHS = ["/ingester", "/firacChecker", "/constitution"]` — non-admins are redirected to `/dashboard`. There are no Firebase Custom Claims involved — roles live entirely in Firestore.

---

### B3. Auth

**Firebase client init:** `frontend/app/lib/firebase.ts` — reads `NEXT_PUBLIC_FIREBASE_*` env vars, calls `initializeApp()` once (guarded by `getApps().length`). Exports `auth` (Firebase Auth) and `db` (Firestore client).

**Auth context:** `frontend/app/context/AuthContext.tsx`
- `onAuthStateChanged` → fetches Firestore profile → sets `user`, `profile`, `loading`
- Exports `useAuth()` hook
- Provides `signOut()` and `refreshProfile()`

**Token attachment to backend requests:** `frontend/app/lib/api.ts`
```typescript
const token = await auth.currentUser?.getIdToken();
headers["Authorization"] = `Bearer ${token}`;
```
Every protected call goes through `apiFetch(path, options)` which auto-attaches the live ID token.

**Roles are NOT stored in Firebase Custom Claims** — they are read from the Firestore document `users/{uid}.type`. The backend also reads from Firestore (not decoded token claims). This means role changes in Firestore take effect on the next request without requiring token refresh.

---

### B4. API Layer

**Base URL config:** `NEXT_PUBLIC_API_URL` env var (default `http://localhost:5000`) — read in `frontend/app/lib/api.ts:3`.

**Fetch wrapper:** `apiFetch(path, options)` in `frontend/app/lib/api.ts` — auto-injects `Authorization: Bearer <token>` and `Content-Type: application/json` (skipped for `FormData`). Also exports `fetchProtectedFile(path)` for binary (PDF) responses → blob URL with 60s auto-revoke.

**All backend endpoints called by the frontend:**

| Endpoint | Called from |
|----------|------------|
| `POST /api/qretrieve` | `ChatView.tsx:464` |
| `POST /api/generate-legal-answer` | `ChatView.tsx:481` |
| `GET /api/chunks/<id>/location` | `ChatView.tsx:1769` |
| `GET /api/documents/<doc_key>/chunks` | `ChatView.tsx:1588` |
| `GET /api/case-pdf` | `ChatView.tsx` (via `fetchProtectedFile`) |
| `GET /api/rule-pdf` | `ChatView.tsx` (via `fetchProtectedFile`) |
| `GET /api/conversations` | `historyApi.ts:59` |
| `GET /api/conversations/<id>` | `historyApi.ts:67` |
| `PATCH /api/conversations/<id>` | `historyApi.ts:76` |
| `DELETE /api/conversations/<id>` | `historyApi.ts:84` |
| `POST /api/conversations/<id>/regenerate-title` | `historyApi.ts:89` |
| `GET /api/files` | `ingester/page.tsx:29` |
| `POST /api/upload` | `ingester/page.tsx:60` |
| `POST /api/digest` | `ingester/page.tsx:132`, `firacChecker/page.tsx:111` |
| `POST /api/firac` | `firacChecker/page.tsx:43` |
| `POST /api/digest/constitution` | `constitution/page.tsx:24` |
| `GET /api/digest/constitution/status` | `constitution/page.tsx:53` |
| `DELETE /api/digest/constitution/reset` | `constitution/page.tsx:83` |
| `POST /api/case-laws/upload` | `repository/caselaws/page.tsx:123` |

---

### B5. Reusable Components Relevant to the New Module

| Component | File | Relevant capability |
|-----------|------|---------------------|
| **Drag-drop upload** | `app/ingester/page.tsx:165–200` | Inline `onDrop`/`onDragOver` zone + hidden `<input type="file">` + click-to-upload. **Not extracted to a component** — inline in the page. Accepts `.pdf` only. Pattern to copy. |
| **File input (modal)** | `app/repository/caselaws/page.tsx:330–345` | `<input type="file" accept=".pdf,.docx">` inside an inline modal. Pattern to copy for multi-type uploads. |
| **Skeleton loading system** | `app/components/loading/Skeleton.tsx`, `app/components/skeleton/*` | `SkeletonBar`, `SkeletonText`, `SkeletonPill`, `SkeletonCircle`, `Reveal`, `DevReplay` — composable shimmer system. Use for the processing screen. |
| **AuthGuard** | `app/components/AuthGuard.tsx` | Wraps pages, redirects unauthenticated users. Extend `ADMIN_ONLY_PATHS` / add `LITIGANT_PATHS` here. |
| **PdfViewer** | `app/components/PdfViewer.tsx` | react-pdf wrapper. Reusable for showing the uploaded evidence document. |
| **UserMenu** | `app/components/UserMenu.tsx` | Sign-out dropdown, shown in header. Reuse in new pages. |
| **Navigation** | `app/components/Navigation.tsx` | Top nav bar. Reuse in new pages. |
| **Loader2 spinner** | lucide-react `Loader2` icon with `animate-spin` className | Existing processing indicator pattern — no dedicated component, but used consistently. |

No modal component, no toast/notification component, no form validation library — all are inline implementations per page.

---

### B6. Wireframe Screen Mapping

| Screen | Purpose | Status | Existing file(s) |
|--------|---------|--------|-----------------|
| 1. Shared login | Sign-in/sign-up for all roles | **EXISTS** | `app/auth/sign-in/page.tsx`, `app/auth/sign-up/page.tsx` |
| 2. SCC litigant dashboard with past submissions | Entry point + submission history list | **PARTIAL** | `/dashboard` page exists but no litigant role, no submission history tiles. Layout pattern reusable. |
| 3. Upload evidence | Multi-type file upload (M-Pesa PDF/PNG, WhatsApp .txt/PNG, .eml, receipt) | **MISSING** | No page. Upload patterns exist in `ingester/page.tsx` (drag-drop) and `repository/caselaws/page.tsx` (modal). |
| 4. Processing | Show progress while OCR + LLM run | **MISSING** | No page. Skeleton system (`app/components/skeleton/`) and `Loader2` spinner are available building blocks. |
| 5. Contract elements identified | Display offer/acceptance/consideration/intention extracted from document | **MISSING** | No page. ChatView's source card layout is a stylistic reference. |
| 6. Supplementary 106B questionnaire | Collect missing facts for the compliance gate | **MISSING** | No page. No form component exists to reuse directly. |
| 7. Compliance result | Certificate download or plain-language deficiency report | **MISSING** | No page. No PDF generation or download component exists. |

---

## Installed vs. Missing Dependencies Table

### Python Packages

| Package | Status | Version | Action needed |
|---------|--------|---------|--------------|
| pytesseract | ✅ Installed | 0.3.13 | None |
| PyMuPDF (fitz) | ✅ Installed | 1.26.6 | Unpin in requirements.txt (currently pinned to older 1.24.14) |
| pypdf | ✅ Installed | 6.3.0 | None |
| Pillow | ✅ Installed | 12.0.0 | Unpin in requirements.txt (currently pinned to 11.1.0) |
| python-docx | ✅ Installed | 1.2.0 | None |
| firebase-admin | ✅ Installed | 7.7.0 | None |
| chromadb | ✅ Installed | 1.3.5 | None |
| sentence-transformers | ✅ Installed | 5.1.2 | None |
| groq | ✅ Installed | 0.35.0 | None (use `call_groq()`, not SDK) |
| flask, flask-cors | ✅ Installed | 3.1.2 / 6.0.1 | None |
| langchain | ✅ Installed | 1.0.7 | Not needed for new module |
| openai | ✅ Installed | 2.9.0 | NOT in requirements.txt; not needed for new module |
| rapidfuzz | ✅ Installed | 3.14.6 | NOT in requirements.txt; not needed for new module |
| pdfplumber | ❌ Not installed | — | Not needed — use fitz |
| pdf2image | ❌ Not installed | — | Not needed — use fitz page rendering |
| opencv-python | ❌ Not installed | — | Not needed |
| **reportlab** | ❌ **Not installed** | — | **INSTALL for PDF certificate generation** (`pip install reportlab`) |
| weasyprint | ❌ Not installed | — | Alternative to reportlab (heavier; needs system libs) |
| docxtpl | ❌ Not installed | — | Alternative (DOCX output) |

### System Binaries

| Binary | Status | Version | Notes |
|--------|--------|---------|-------|
| tesseract | ✅ Installed | 5.5.0 | `eng` + `osd` only — no Swahili (`swa`) lang pack |
| pdftoppm (poppler) | ❌ Not installed | — | Not needed — use `fitz` for rendering |
| ocrmypdf | ✅ In venv | 16.5.0 | LIVE_BUILD only; not needed at request-time |

---

## Integration Points

### Where to plug in the new backend blueprint

**File:** Create `backend/admissibility/` package:
```
backend/admissibility/
├── __init__.py
├── routes.py          ← Flask Blueprint "admissibility_bp"
├── extractor.py       ← text extraction (lift from CaseLaw/firac.py:45–95)
├── contract_llm.py    ← contract element identification (new)
├── s106b_gate.py      ← 106B compliance gate (new, uses existing RulesRetriever)
├── certificate.py     ← PDF certificate generation (new, needs reportlab)
└── persistence.py     ← Firestore: users/{uid}/evidence_submissions/{subId}
```

**Registration in `app.py`:**
```python
# After line 39: app.register_blueprint(history_bp)
from admissibility.routes import admissibility_bp
app.register_blueprint(admissibility_bp)
```

**Proposed new routes (follow existing pattern):**
```
POST /api/evidence/upload          @require_role("litigant","advocate","admin")
POST /api/evidence/analyse/<sub_id>   @require_role("litigant","advocate","admin")
POST /api/evidence/questionnaire/<sub_id>  @require_role("litigant","advocate","admin")
GET  /api/evidence/result/<sub_id>   @require_role("litigant","advocate","admin")
GET  /api/evidence/submissions       @require_role("litigant","advocate","admin")
GET  /api/evidence/certificate/<sub_id>  @require_role("litigant","advocate","admin")
```

**New `"litigant"` role:** Add to `auth.py` — no code change needed, `require_role()` is variadic. Add `"litigant"` to allowed roles on each new route. Update `AuthGuard.tsx` to allow litigants on new pages.

**Groq model override:** `call_groq(msgs, model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=3000)` — the wrapper already supports this.

**106B retrieval:** Call `retrieve_relevant_rules(query_params)` from `RulesRetriever.py` with a query targeting Evidence Act; the existing `statutes` collection already contains all s.106B subsections.

**File size limit (add immediately):**
```python
# app.py, after Flask app creation
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB
```

**New Firestore collection for submissions:**
```
users/{uid}/evidence_submissions/{sub_id}
  status: "uploaded" | "extracting" | "analysing" | "questionnaire" | "complete" | "failed"
  file_name: str
  file_type: str
  uploaded_at: Timestamp
  extracted_text: str (truncated if large)
  contract_elements: { offer, acceptance, consideration, intention } (nullable)
  s106b_gate: { passed: bool, missing_facts: [], sufficient_facts: [] } (nullable)
  questionnaire_answers: {} (nullable)
  result: { type: "certificate" | "deficiency", content: str } (nullable)
  certificate_url: str (nullable)
  schema_version: 1
```

### Where to plug in new frontend pages

**New directory:** `frontend/app/evidence/`
```
frontend/app/evidence/
├── page.tsx                    ← Screen 2: Litigant dashboard (past submissions)
├── upload/page.tsx             ← Screen 3: Upload evidence
├── processing/[subId]/page.tsx ← Screen 4: Processing status
├── elements/[subId]/page.tsx   ← Screen 5: Contract elements review
├── questionnaire/[subId]/page.tsx ← Screen 6: 106B supplementary questions
└── result/[subId]/page.tsx     ← Screen 7: Certificate / deficiency report
```

**AuthGuard update (`AuthGuard.tsx`):**
```typescript
// Add to PUBLIC_PATHS check logic:
const LITIGANT_ALLOWED_PATHS = ["/evidence"];
// Litigants should NOT access /dashboard (or redirect to /evidence)
```

**API calls to add in `frontend/app/lib/evidenceApi.ts`** (follow `historyApi.ts` pattern):
```typescript
apiFetch("/api/evidence/upload", { method: "POST", body: formData })
apiFetch(`/api/evidence/analyse/${subId}`, { method: "POST" })
apiFetch(`/api/evidence/questionnaire/${subId}`, { method: "POST", body: JSON.stringify(answers) })
apiFetch(`/api/evidence/result/${subId}`)
apiFetch("/api/evidence/submissions")
```

---

## Risks and Inconsistencies

1. **⚠️ Seven routes unprotected** — `GET /api/case-pdf`, `GET /api/rule-pdf`, `GET /api/chunks/*`, `GET /api/documents/*`, `GET /api/cases/*`, `GET /api/statutes/*`, `GET /api/case-laws/file/*`, and `GET /api/digest/constitution/status` have no `@require_role` decorator. Anyone who discovers the API URL can read ChromaDB chunk content and download PDFs without authentication. This is likely intentional for the PDF viewer (which uses blob URLs client-side), but should be audited before the citizen-facing tier is exposed publicly.

2. **⚠️ No `MAX_CONTENT_LENGTH`** — Flask will accept arbitrarily large uploads. A citizen could POST a multi-GB file and exhaust server RAM (the upload is fully buffered: `file.read()` at app.py:1857).

3. **⚠️ CORS origin is a single string** — if the citizen-facing frontend is on a separate subdomain, you must update `CORS_ORIGIN`. flask-cors cannot handle a comma-separated string as multiple origins; use a list: `CORS(app, origins=["https://scc.legalai.ke", "https://app.legalai.ke"])`.

4. **`CaseLaw/firac.py` bypasses `call_groq()`** — it reads `GROQ_API_KEY` directly and posts to Groq with its own `requests.post`. Key rotation, `<think>` stripping, and retry logic in `groq_client.py` do not apply to FIRAC calls. Not a bug for the new module (the new module should use `call_groq()`), but worth noting.

5. **`fb_storage` imported but never used** — `app.py:13` imports `storage as fb_storage` but no line in the file calls `fb_storage`. Firebase Storage is configured (bucket `legalai-scc.firebasestorage.app`) but not used. This is relevant for the module: **evidence files are stored locally**, not in Cloud Storage. On Railway/Render with ephemeral storage, uploaded files would be lost between deploys unless a volume is mounted.

6. **Two service account files in `backend/`** — `legalai-scc-firebase-adminsdk-fbsvc-73bde27853.json` and `legalai-service-account.json`. It is unclear which `FIREBASE_SERVICE_ACCOUNT_PATH` points to. Both should not be committed to git.

7. **`openai` and `rapidfuzz` installed but not in `requirements.txt`** — anyone doing a fresh `pip install -r requirements.txt` will have a broken `CaseLaw/firac.py` and `compute_constitution_bboxes.py`.

8. **`law_of_contract_act` has only 6 chunks** — Cap 23 is essentially a referencing shell. The actual contract formation doctrine (offer, acceptance, consideration, intention) must be identified by the LLM reading the uploaded document, not by retrieving Cap 23 text. Design the contract element extraction as a pure LLM task with a carefully grounded prompt.

9. **Tesseract language — `eng` only** — no Swahili (`swa`) pack installed. M-Pesa statements and some WhatsApp exports may contain Swahili text. Install `tesseract-ocr-swa` on the server if needed.

10. **No PDF-generation library installed** — reportlab (or weasyprint) must be added before the certificate output can be built.

11. **`history_store.py` 900 KB Firestore document guard (`_MAX_DOC_BYTES = 900_000`)** — this limit is enforced for conversation turns. Evidence submission documents in Firestore should respect the same 1 MB ceiling; extracted text from large PDFs must be truncated or stored separately (e.g. Cloud Storage or a separate Firestore subcollection).

12. **`CORS_ORIGIN` does not support multiple origins as a comma-separated string** — flask-cors `origins=` expects a list or single string. If the citizen portal lives on a different origin than the advocate portal, a list must be passed.

---

## Open Questions for the Developer

1. **Litigant role lifecycle** — how does a citizen become a `litigant`? Self-registration (like advocate) or admin-issued? Does `advocateVerified` have an analogue (`litigantVerified`)? What stops an advocate from accessing citizen-only submissions?

2. **Evidence file storage at deployment** — uploaded M-Pesa PDFs/images go to `backend/SCCCases/` (local disk). On Railway with a mounted volume this works. Are you planning to use Firebase Storage for evidence files, or keep them local? The import is already there but unused.

3. **Async processing** — OCR + LLM for a multi-page M-Pesa PDF will take 10–30 seconds. Does the module use a polling pattern (submit → status endpoint → result), a WebSocket, or Server-Sent Events? Flask dev server doesn't support SSE well; if you use polling, the `/api/evidence/analyse/<sub_id>` endpoint should be idempotent (check if analysis is already done before re-running, following the existing `generate-legal-answer` idempotency pattern at `app.py:1332`).

4. **WhatsApp `.txt` exports** — these are plain-text chat logs. No OCR needed (read directly). WhatsApp screenshot PNG files do need OCR. Will the module accept both, and how will it distinguish?

5. **Certificate format** — does the 106B certificate need to look like an official court form, or is a clean A4 PDF sufficient? This determines whether to use reportlab (programmatic layout), weasyprint (HTML→PDF), or docxtpl (Word template).

6. **M-Pesa statement parsing** — M-Pesa PDFs from Safaricom have a consistent tabular layout. A structured parser (pdfplumber or fitz table extraction) would give cleaner data than raw OCR. Is structured extraction in scope, or is raw OCR text sufficient for the LLM to identify transaction elements?

7. **Citizen data retention and privacy** — uploaded evidence files contain personal financial/communication data. What is the retention policy? Should files be deleted after the certificate is issued? This needs a `DELETE /api/evidence/<sub_id>` route and a cleanup schedule.

8. **Section 78A vs 106B** — the module description mentions s.106B (electronic records). `validate_knowledge.py` references `s.78A` in comments (`# exp_statute: ("evidence_act", "78A")`). Are both sections in scope, or only 106B?
