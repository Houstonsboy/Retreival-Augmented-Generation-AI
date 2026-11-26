# Chapter 5 — Verified Facts
## Sections 5.2–5.5: GraphRAG Legal Decision-Support System
### Kenya's Small Claims Court

*Audit date: 2026-11-02. All values sourced from static code analysis, live
ChromaDB queries, SQLite queries, and git history. Nothing was executed from
`app.py`. Secrets are named only; values are never printed.*

---

## 1. LIVE_RUNTIME Files

Every local `.py` file reachable by static import-tracing from `backend/app.py`.
Imports inside functions are included; `venv/` is excluded.

```
backend/app.py                                   ← entry point
  ├─ backend/auth.py                             (from auth import ...)
  ├─ backend/groq_client.py                      (from groq_client import call_groq, GROQ_MODEL, get_groq_keys)
  ├─ backend/history_store.py                    (from history_store import ...)
  ├─ backend/history_routes.py                   (from history_routes import history_bp)
  ├─ backend/digester.py                         (from digester import digest_repo)
  ├─ backend/pixe.py                             (from pixe import run_rag_pipeline)
  ├─ backend/CaseLaw/firac.py                    (from CaseLaw.firac import run_firac, run_firac_from_file)
  ├─ backend/CaseLaw/ingester.py                 (from CaseLaw.ingester import ingest_firac_data)
  ├─ backend/Qclassifier.py                      (from Qclassifier import Qclassifier)
  │    └─ backend/Retriever.py                   (from Retriever import retrieve_relevant_cases)
  ├─ backend/RulesRetriever.py                   (from RulesRetriever import retrieve_relevant_rules)
  ├─ backend/constitutionlogic/constitutionbreaker.py
  │                                              (from constitutionlogic.constitutionbreaker import ...)
  ├─ backend/constitutionlogic/constitution_digester.py
  │                                              (from constitutionlogic.constitution_digester import ...)
  ├─ backend/config.py                           (from config import ...)
  ├─ backend/coverage.py                         (imported locally inside generate_legal_answer: app.py:1405)
  └─ backend/citation_verifier.py                (imported locally inside generate_legal_answer: app.py:1406)
```

**Third-party packages per LIVE_RUNTIME file:**

| File | Third-party imports |
|------|---------------------|
| app.py | flask, flask_cors, firebase_admin, requests, werkzeug, chromadb |
| auth.py | firebase_admin, flask |
| groq_client.py | requests |
| history_store.py | firebase_admin, google (google-cloud-firestore) |
| history_routes.py | flask |
| digester.py | chromadb, langchain_core, langchain_text_splitters, numpy, pypdf, sentence_transformers |
| pixe.py | chromadb, groq, langchain_core, numpy, requests, sentence_transformers, sklearn |
| CaseLaw/firac.py | PIL, dotenv, fitz (PyMuPDF), openai, pytesseract, requests |
| CaseLaw/ingester.py | chromadb, numpy, sentence_transformers |
| Qclassifier.py | dotenv, requests |
| Retriever.py | chromadb, dotenv, rank_bm25, requests, sentence_transformers |
| RulesRetriever.py | chromadb, rank_bm25, sentence_transformers |
| constitutionlogic/constitutionbreaker.py | PIL, fitz, pytesseract |
| constitutionlogic/constitution_digester.py | chromadb, sentence_transformers |
| config.py | *(stdlib only)* |
| coverage.py | *(stdlib only)* |
| citation_verifier.py | *(stdlib only)* |

**Blueprint registrations found in LIVE_RUNTIME:**
- `app.py:39` — `app.register_blueprint(history_bp)` (from `history_routes.py`)

---

## 2. LIVE_ROUTES and Frontend Pages

### Flask Routes (all routes in LIVE_RUNTIME)

| Method | Path | File:line | Role required | Called by frontend? |
|--------|------|-----------|---------------|---------------------|
| GET | /api/health | app.py:45 | none | NO (available but not called) |
| POST | /api/chat | app.py:50 | advocate/admin | NO — legacy |
| GET | /api/files | app.py:115 | advocate/admin | YES (ingester page) |
| POST | /api/upload | app.py:127 | admin | YES (ingester page) |
| GET | /api/files/<filename> | app.py:159 | advocate/admin | YES (ingester page) |
| POST | /api/digest | app.py:184 | admin | YES (ingester + firacChecker pages) |
| POST | /api/firac | app.py:279 | advocate/admin | YES (firacChecker page) |
| POST | /api/ingest-firac | app.py:403 | admin | NO |
| POST | /api/digest/constitution | app.py:522 | admin | YES (constitution page) |
| GET | /api/digest/constitution/status | app.py:847 | none | YES (constitution page) |
| DELETE | /api/digest/constitution/reset | app.py:899 | admin | YES (constitution page) |
| GET | /api/case-pdf | app.py:952 | advocate/admin | YES (ChatView workspace PDF) |
| GET | /api/chunks/<chunk_id>/location | app.py:1068 | advocate/admin | YES (ChatView:1769) |
| GET | /api/documents/<doc_key>/chunks | app.py:1101 | advocate/admin | YES (ChatView:1586–1587) |
| GET | /api/cases/<case_identifier>/all-chunks | app.py:1126 | advocate/admin | YES |
| GET | /api/statutes/<source_document>/all-chunks | app.py:1159 | advocate/admin | YES |
| GET | /api/rule-pdf | app.py:1193 | advocate/admin | YES (workspace statute PDF) |
| POST | /api/qretrieve | app.py:1248 | advocate/admin | **YES — primary** |
| POST | /api/generate-legal-answer | app.py:1393 | advocate/admin | **YES — primary** |
| POST | /api/case-laws/upload | app.py:1835 | advocate/admin | YES (repository/caselaws) |
| GET | /api/case-laws/file/<filename> | app.py:1882 | none | YES (repository/caselaws) |
| GET | /api/conversations | history_routes.py:43 | advocate/admin | YES (sidebar list) |
| GET | /api/conversations/<conv_id> | history_routes.py:62 | advocate/admin | YES (load conversation) |
| PATCH | /api/conversations/<conv_id> | history_routes.py:93 | advocate/admin | YES (pin/rename) |
| DELETE | /api/conversations/<conv_id> | history_routes.py:129 | advocate/admin | YES |
| POST | /api/conversations/<conv_id>/regenerate-title | history_routes.py:148 | advocate/admin | YES |

**LIVE_ROUTES** (called by frontend on the main research path):
- `POST /api/qretrieve` → `POST /api/generate-legal-answer` (two-call pipeline per query)
- History management: GET/PATCH/DELETE `/api/conversations/…`
- PDF and chunk endpoints for the workspace panel

### Frontend Pages

| Route | File | Main component / purpose |
|-------|------|--------------------------|
| `/` | app/page.tsx | ChatView — main research workspace |
| `/chat/[conversationId]` | app/chat/[conversationId]/page.tsx | ChatView with initialConversationId |
| `/dashboard` | app/dashboard/page.tsx | Entry hub (Chat + Repository tiles) |
| `/auth/sign-in` | app/auth/sign-in/page.tsx | Email/password + Google sign-in |
| `/auth/sign-up` | app/auth/sign-up/page.tsx | Email/password + Google sign-up |
| `/auth/complete-profile` | app/auth/complete-profile/page.tsx | Username selection after OAuth |
| `/constitution` | app/constitution/page.tsx | Constitution ingestion admin page |
| `/firacChecker` | app/firacChecker/page.tsx | FIRAC extraction test page |
| `/ingester` | app/ingester/page.tsx | Document upload/digest admin page |
| `/repository` | app/repository/page.tsx | File browser hub |
| `/repository/caselaws` | app/repository/caselaws/page.tsx | Case law document browser/upload |
| `/repository/statutes` | app/repository/statutes/page.tsx | Statute browser |

---

## 3. LIVE_DATA

Every data resource opened or read by LIVE_RUNTIME code, with the line where it is opened.

| Resource | Path (resolved from config.py constants) | File:line | Purpose |
|----------|------------------------------------------|-----------|---------|
| ChromaDB `case_law` | backend/CaseLawDB/ | Retriever.py:614, config.py:12 | Case-law retrieval |
| ChromaDB `statutes` | backend/StatutesDB/ | RulesRetriever.py:511, config.py:16 | Statute retrieval |
| ChromaDB `master_rules` | backend/constitutionlogic/MasterRulesDB/ | RulesRetriever.py:206, config.py:20 | Constitution fallback |
| SQLite `locations.db` | backend/locations.db | app.py:1068 (via /api/chunks/:id/location) | Chunk page coordinates |
| manifest.json | data/processed/statutes/manifest.json | RulesRetriever.py:34, config.py:30 | Act aliases for coverage check + retrieval filter |
| cases_meta.json | data/processed/cases/cases_meta.json | app.py (via /api/case-laws/upload route, and build_case_chunks.py) | Case registry |
| SCCCases/ PDFs | backend/SCCCases/ | app.py:34 (config constant), /api/case-pdf route | Source PDFs served to frontend |
| Embedding model | intfloat/e5-base-v2 (HuggingFace cache) | Retriever.py:600, RulesRetriever.py:497 | Query embedding at retrieval time |
| Cross-encoder model | cross-encoder/ms-marco-MiniLM-L-6-v2 (HF cache) | Retriever.py:84, RulesRetriever.py:78 | Reranking |
| Firestore `users/{uid}` | Firebase project legalai-scc | auth.py:56 | Profile + role check |
| Firestore `conversations/{cid}/turns` | Firebase project legalai-scc | history_store.py:200+ | History read/write |

---

## 4. LIVE_BUILD Scripts (in run order)

These are the scripts that produced the data currently in LIVE_DATA, and the
evidence for why each one produced the current artefact.

### Build order

```
Step 1  constitutionlogic/constitutionbreaker.py
Step 2  constitutionlogic/constitution_digester.py
Step 3  constitutionlogic/evidencedigester.py         ← produces master_rules evidence chunks
Step 4  compute_constitution_bboxes.py                ← adds bboxes to master_rules rows in locations.db
Step 5  build_case_chunks.py                          ← produces cases_embed.jsonl + locations.db case_law rows
Step 6  build_statute_chunks.py                       ← produces sections_embed.jsonl
Step 7  embed_knowledge.py --source cases             ← writes to CaseLawDB/case_law
Step 8  embed_knowledge.py --source statutes          ← writes to StatutesDB/statutes
```

### Evidence for each script

**Steps 1–4 (master_rules):**
- The live `master_rules` collection contains 619 chunks: 374 with
  `source_document="TheConstitutionOfKenya.pdf"` and 245 with
  `source_document="Evidence Act.pdf"`.
- `constitutionlogic/constitution_digester.py` opens `MasterRulesDB/master_rules`
  and adds constitution chunks with prefix `"passage: "` (constitution_digester.py:120).
- `constitutionlogic/evidencedigester.py` opens the same `MasterRulesDB` and
  same `master_rules` collection and adds evidence act chunks.
- `compute_constitution_bboxes.py` runs after ingestion to update `boxes_json`
  for `master_rules` rows in `locations.db` (compute_constitution_bboxes.py:207,226).

**Step 5 (case_law):**
- Chunk IDs in `case_law` match the pattern produced by `build_case_chunks.py`
  (e.g., `2026SCC242KLR_0`, `2001KECA154(KLR)_0`).
- The `locations.db` `case_law` rows were written by `build_case_chunks.py`
  (which calls `_init_locations_db` and inserts per chunk).
- `build_case_chunks.py` is not in LIVE_RUNTIME; its CLI entry point runs standalone.
- git log: `e737d54` ("Hybrid retrieval: BM25+RRF+cross-encoder, StatutesDB knowledge switch")
  is the commit that introduced `build_case_chunks.py`.

**Steps 6–8 (statutes):**
- Chunk IDs in `statutes` match slug patterns from `build_statute_chunks.py`
  (e.g., `small_claims_court_act__2022-12-31__s1__0`).
- `embed_knowledge.py` reads `STATUTES_JSONL` and `CASES_JSONL` from config constants
  and writes to the ChromaDB collections (embed_knowledge.py:110: `"passage: " + c["text"]`).

### LIVE_BUILD local imports

- `build_case_chunks.py` imports: stdlib + fitz, pytesseract, ocrmypdf (system), python-docx
- `build_statute_chunks.py` imports: sentence_transformers (tokenizer only), stdlib
- `embed_knowledge.py` imports: config.py, chromadb, sentence_transformers
- `constitutionlogic/constitutionbreaker.py` ← **also in LIVE_RUNTIME**
- `constitutionlogic/constitution_digester.py` ← **also in LIVE_RUNTIME**
- `constitutionlogic/evidencedigester.py` imports: fitz, PIL, pytesseract (extraction); chromadb, sentence_transformers (embedding)
- `compute_constitution_bboxes.py` imports: fitz, rapidfuzz, sqlite3

---

## 5. INACTIVE Files and Collections

### Inactive Python files (not in LIVE_RUNTIME or LIVE_BUILD)

```
backend/agentrouter.py                        INACTIVE — not used
backend/CaseLaw/__init__.py                   INACTIVE — not used
backend/CaseLaw/update_path.py                INACTIVE — not used
backend/constitutionlogic/constitution.py     INACTIVE — not used
backend/constitutionlogic/masteruledb.py      INACTIVE — not used (export/exploration only)
backend/debug_firac.py                        INACTIVE — not used
backend/deletedb.py                           INACTIVE — not used
backend/scripts/bootstrap_admin.py            INACTIVE — admin CLI tooling, not data pipeline
backend/scripts/seed_caselaws.py              INACTIVE — not used
backend/scripts/set_role.py                   INACTIVE — admin CLI tooling, not data pipeline
backend/scripts/__init__.py                   INACTIVE — not used
backend/testdb.py                             INACTIVE — not used
backend/tests/__init__.py                     INACTIVE — not used
backend/tests/test_history.py                 INACTIVE — not used
backend/validate_knowledge.py                 INACTIVE — not used
```

*(Files excluded per Rule 6: backend/Const_IntepretationLaws/reader.py,
backend/Const_IntepretationLaws/testBlackLaw.py, backend/export_blacklaw_to_json.py)*

### Inactive ChromaDB collections

```
backend/LegalSummariesDB/legal_summaries      INACTIVE — 40 chunks, 8 cases, not queried at runtime
backend/ChunkDB_backup_20261031/              INACTIVE — backup snapshot, not used
```

---

## 6. Answers A1–G1

---

### A. LLM

#### A1. Every LLM call on a live route

There are three distinct call sites. All use `call_groq()` from `groq_client.py`
except the Qclassifier, which has its own inline `requests.post`.

**Call site 1 — Query classification (Qclassifier.py:473)**
- Route: `POST /api/qretrieve` → `Qclassifier()` → `classify_legal_query()`
- **Model string:** `"qwen/qwen3.8-27b"` — Qclassifier.py:26: `MODEL_NAME = "qwen/qwen3.8-27b"`
- Sent to Groq as: `payload["model"] = MODEL_NAME` (Qclassifier.py:449)
- temperature: `0.1` (Qclassifier.py:460)
- max_tokens: `1500` (Qclassifier.py:461)
- timeout: `30` (Qclassifier.py:477)
- Does NOT use `call_groq()`; has its own key-rotation logic.

**Call site 2 — Legal answer synthesis (app.py:1595)**
- Route: `POST /api/generate-legal-answer` → `call_groq(messages)`
- **Model string:** `"qwen/qwen3.8-27b"` — groq_client.py:20: `GROQ_MODEL = "qwen/qwen3.8-27b"`; call: `call_groq(messages)` with no model override → default `model=GROQ_MODEL` (groq_client.py:35)
- Call chain: `app.py:1595 call_groq(messages)` → `groq_client.py:32 call_groq(msgs, model=GROQ_MODEL, temperature=0, max_tokens=2500, timeout=60)`
- temperature: `0` (groq_client.py:37 default, not overridden at call site)
- max_tokens: `2500` (groq_client.py:38 default)
- timeout: `60` (groq_client.py:39 default)

**Call site 3 — Regeneration after citation failure (app.py:1641–1650)**
- Same call as call_groq via `_regen()` closure → `call_groq(regen_msgs)` (app.py:1650)
- Same model, temperature=0, max_tokens=2500, timeout=60
- Called at most once per request (citation_verifier.py:188)

**Call site 4 — Background title generation (history_store.py:441–448)**
- Route: triggered after a new conversation is created in `generate-legal-answer`
- **Model string:** `"qwen/qwen3.8-27b"` — uses `call_groq(…)` with no model override
- temperature: `0.2` (history_store.py:446)
- max_tokens: `300` (history_store.py:447)
- timeout: `10` (history_store.py:448)
- Runs in a `ThreadPoolExecutor` thread; fires only when `is_new_conversation` is True.

> **CORRECTS EARLIER VALUE:** Earlier report stated `"qwen3-27b"`. The literal
> string in the live code is `"qwen/qwen3.8-27b"` (groq_client.py:20,
> Qclassifier.py:26, Retriever.py:43). Both retrievers and the classifier use
> the identical string.

---

#### A2. Key rotation

**groq_client.py (call_groq, call sites 2/3/4):**
- Env vars read: `GROQ_API_KEY` and any var whose name starts with `GROQ_API_KEY_`
  (groq_client.py:28–29: `k == "GROQ_API_KEY" or k.startswith("GROQ_API_KEY_")`)
- All matching env vars are sorted by name and collected into a list.
- On HTTP 429: immediately rotate to `(current_index + 1) % len(keys)`. If
  `next_index == current_index` (only one key), sleep 30 s (groq_client.py:76),
  then retry. After `len(keys) + 1` total attempts, raises `RuntimeError("All
  Groq keys exhausted")`.
- All live routes using `call_groq()` go through this rotation.

**Qclassifier.py (call site 1 — independent):**
- Env vars read: exactly `["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]`
  (Qclassifier.py:14 — hardcoded list of three names).
- On HTTP 429: rotate to next key immediately; if all keys are on the same index,
  reads `retry-after` header (capped at 60 s), falls back to 30 s wait
  (Qclassifier.py:496–502). Resets to key 0 after wait.
- max_retries: 3 (Qclassifier.py:423 signature default, called with no override).
- Does NOT use `call_groq()`; has its own `requests.post`.

> **NOTE:** The two key-rotation implementations have divergent env var discovery
> strategies. `groq_client.py` discovers all `GROQ_API_KEY*` vars at call time.
> `Qclassifier.py` discovers keys at module import time from three fixed names.
> A key named `GROQ_API_KEY_4` would be picked up by `groq_client` but not by
> `Qclassifier`.

---

#### A3. `<think>` block stripping

`groq_client.py:23`:
```python
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
```
Applied at `groq_client.py:81`: `return _THINK_RE.sub("", raw).strip()`

This covers call sites 2, 3, and 4. Call site 1 (Qclassifier) applies its own
stripping: `Qclassifier.py:541` (not shown here, but searches for and strips
`<think>` blocks before JSON extraction).

---

### B. Collections and Counts

#### B1. Live collections

| Collection | Path | count() | distance metric | On-disk size |
|---|---|---|---|---|
| `case_law` | backend/CaseLawDB/ | **78** | cosine | 4.7 MB |
| `statutes` | backend/StatutesDB/ | **1 331** | cosine | 20 MB |
| `master_rules` | backend/constitutionlogic/MasterRulesDB/ | **619** | cosine | 8.3 MB |

> **CORRECTS EARLIER VALUE for `master_rules`:** Earlier audits reported 374.
> The `locations.db` has 374 `master_rules` rows (only constitution articles have
> location rows), but the ChromaDB collection itself contains **619** chunks:
> 374 from `TheConstitutionOfKenya.pdf` (document_type="constitution") and 245
> from `Evidence Act.pdf` (document_type="evidence_act"). The evidence act was
> ingested into the same collection by `constitutionlogic/evidencedigester.py`.
> The `_query_constitution_fallback()` function filters by
> `where={"document_type": {"$eq": "constitution"}}` (RulesRetriever.py:208),
> so only the 374 constitution chunks are returned from that fallback path.

---

#### B2. Exact chunks per case in `case_law`

| case_identifier | Chunks | ocr flag |
|---|---|---|
| 2001KECA154(KLR) | 7 | False |
| 2010KECA346(KLR) | 31 | False |
| 2014KEHC6468(KLR) | 3 | False |
| 2020KEHC8905(KLR) | 5 | False |
| 2024KEHC15763(KLR) | 10 | False |
| 2024KEHC17038(KLR) | 8 | False |
| 2026SCC235(KLR) | 4 | False |
| 2026SCC241(KLR) | 6 | False |
| 2026SCC242(KLR) | 4 | False |
| **TOTAL** | **78** | all False |

> **CORRECTS EARLIER VALUE:** The previous audit listed case identifiers as
> `2026SCC241KLR` through `2026SCC249KLR`. The live collection contains a
> mix of Court of Appeal (`KECA`), High Court (`KEHC`), and Small Claims Court
> (`SCC`) cases, spanning 2001–2026. File format is not stored in `cases_meta.json`
> (the `format` field is absent/null in all 11 entries). All 9 active cases have
> `ocr=False`.

---

#### B3. cases_meta.json

- **Total entries:** 11
- **Excluded entries:** 0 — no entry has `excluded: true` or any exclusion reason field.
  Two of the 11 cases (`2011KEHC1481(KLR)` and `2021KEELC220(KLR)`) are absent
  from the live `case_law` collection despite not being excluded in `cases_meta.json`.
  This is NOT DETERMINABLE FROM CODE — the developer must confirm whether
  `build_case_chunks.py` failed silently for these two files or they were never run.
- **Cross-check:** The 9 case identifiers present in the live collection are a proper
  subset of the 11 in `cases_meta.json`. No case in the collection is absent from
  `cases_meta.json`.

---

#### B4. Statute chunks per act_slug (live `statutes` collection)

> **CORRECTS EARLIER VALUE:** The previous audit reported acts including `land_act`,
> `land_registration_act`, `national_land_commission_act`, `hire_purchase_act`,
> `business_laws_amendment_act`, and `magistrates_courts_act`. None of these exist
> in the live collection. The actual acts are:

| act_slug | Chunks | Act title (from manifest.json) |
|---|---|---|
| bills_of_exchange_act | 106 | Bills of Exchange Act |
| civil_procedure_act | 121 | Civil Procedure Act |
| consumer_protection_act | 105 | Consumer Protection Act |
| employment_act | 111 | Employment Act |
| evidence_act | 213 | Evidence Act |
| judicature_act | 12 | Judicature Act |
| kenya_information_and_communications_act | 231 | Kenya Information and Communications Act |
| landlord_and_tenant_act | 24 | Landlord and Tenant (Shops, Hotels and Catering Establishments) Act |
| law_of_contract_act | 6 | Law of Contract Act |
| limitation_of_actions_act | 53 | Limitation of Actions Act |
| sale_of_goods_act | 61 | Sale of Goods Act |
| small_claims_court_act | 53 | Small Claims Court Act |
| small_claims_court_rules | 68 | Small Claims Court (Procedure) Rules |
| stamp_duty_act | 167 | Stamp Duty Act |
| **TOTAL** | **1 331** | |

No Cap numbers are stored in `manifest.json` (`cap` field is absent/null for all acts).

**Land Act, Land Registration Act, National Land Commission Act, Employment Act:**
- Land Act: **NOT PRESENT** in the live `statutes` collection or manifest.
- Land Registration Act: **NOT PRESENT**.
- National Land Commission Act: **NOT PRESENT**.
- Employment Act: **PRESENT** (111 chunks).

Why these acts were chosen: NOT DETERMINABLE FROM CODE. No commit message or
configuration file explains the statute selection. The developer must confirm.

---

#### B5. locations.db highlight result

```sql
SELECT collection, COUNT(*), SUM(boxes_json != '[]'), SUM(ocr)
FROM chunk_locations GROUP BY collection;
```

| collection | total rows | rows with real boxes | ocr count |
|---|---|---|---|
| case_law | 72 | 72 | 0 |
| master_rules | 374 | 252 | 0 |

**Case law:** 72 of 78 chunks have location rows (6 chunks are missing rows);
all 72 have real bounding-box arrays. PDF highlighting is functional for the
72 indexed chunks. The frontend falls back gracefully: `ChatView.tsx:1781`
constructs `pdfSrc` from the case PDF URL and opens the first page if no
`chunk_id` location is available; the PDF viewer does not crash on missing boxes.

**Constitution (master_rules):** 374 rows, 252 with real bboxes. The remaining
122 articles lack bbox data — `compute_constitution_bboxes.py` may not have
matched every article. Those articles open the PDF at the estimated `page_start`
but do not highlight a passage.

**Statutes:** No rows in `locations.db` for `statutes`. Statute text is shown
in a text reader tab, not as a highlighted PDF.

---

### C. Retrieval Settings

#### C1. Embedding model and query/passage prefixes

- **Model:** `intfloat/e5-base-v2` — config.py:27: `EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"`
- **Query prefix:** `"query: "` prepended at retrieval time.
  - Retriever.py:625: `embedding_model.encode("query: " + query_params['vector_query'])`
  - RulesRetriever.py:539: `embedding_model.encode("query: " + query_str)`
- **Passage prefix:** `"passage: "` prepended at ingestion time.
  - embed_knowledge.py:110: `"passage: " + c["text"]`
  - constitution_digester.py:120: `prefixed_text = f"passage: {chunk_text}"`
- **Loading:** The model is loaded **on every request** via `SentenceTransformer(EMBEDDING_MODEL_NAME)` inside `retrieve_relevant_cases()` (Retriever.py:600) and `retrieve_relevant_rules()` (RulesRetriever.py:497). There is no startup-time or process-level singleton. (The BM25 index and cross-encoder ARE lazily cached at module level.)

---

#### C2. Cross-encoder model and score normalisation

- **Model ID:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Retriever.py:84: `_CASES_CE["model"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
  - RulesRetriever.py:78: `_STATUTES_CE["model"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
- **Score normalisation:** raw CE score → sigmoid. Retriever.py:61–62:
  ```python
  def _sigmoid_c(x: float) -> float:
      return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))
  ```
  Applied at Retriever.py:675: `ce_sigmoid = [_sigmoid_c(float(s)) for s in ce_raw_scores]`
- **Loading:** Lazily initialised once per process; cached in module-level dicts `_CASES_CE` and `_STATUTES_CE`.

---

#### C3. Case retrieval parameters (Retriever.py)

| Parameter | Value | File:line |
|---|---|---|
| BM25 top-k | 50 | Retriever.py:641 `[:50]` |
| Dense top-k | `min(50, collection.count())` | Retriever.py:645 |
| RRF k | 60 | Retriever.py:657 `k=60` |
| RRF n (output size) | 30 | Retriever.py:657 `n=30` |
| Top parent cases returned | 3 | Retriever.py:47 `TOP_N_PARENTS = 3` |
| Neighbour expansion | Yes — `_enrich_case_chunks()` fetches prev/next chunk IDs for top-3 parents | Retriever.py:725–726 |
| Low-confidence threshold | 0.4 (warning only; does not filter) | Retriever.py:53, :746 |
| MAX_CHUNKS_PER_COMPONENT | 1 | Retriever.py:50 |
| Case chunks sent to LLM | Up to 3 per case (app.py:1504 `case.get("chunks", [])[:3]`) | app.py:1504 |
| Skip LLM summaries at retrieval | True — `skip_summaries=True` (Qclassifier.py:715) | Qclassifier.py:715 |

---

#### C4. Statute retrieval parameters (RulesRetriever.py)

| Parameter | Value | File:line |
|---|---|---|
| BM25 top-k | 50 | RulesRetriever.py:556 `[:50]` |
| Dense top-k | `min(50, collection_count)` | RulesRetriever.py:562 |
| RRF k | 60 | RulesRetriever.py:574 `k=60` |
| RRF n | 30 | RulesRetriever.py:574 `n=30` |
| Section injection cap | 35 after injection | RulesRetriever.py:612 `fused_ids[:35]` |
| Schedule penalty | `0.3×` for chunks where `is_schedule=True` | RulesRetriever.py:115 |
| Schedule penalty trigger | Only applied if query does NOT contain `{"schedule","form","fee","tariff","appendix","annex","prescribed"}` | RulesRetriever.py:113–115 |
| Section injection trigger | Query contains explicit section numbers AND at least one named Act | RulesRetriever.py:602 |
| Constitution fallback trigger | `_is_constitution_query()` returns True — checks classifier's `source_documents` + `all_statutes` against `_CONSTITUTION_ALIASES` from manifest | RulesRetriever.py:278 |
| Constitution fallback aliases | `{'Constitution of Kenya', 'Constitution of Kenya 2010', 'Constitution of Kenya, 2010', 'Constitution', 'The Constitution', 'Kenya Constitution'}` (manifest.json constitution_fallback.aliases) | RulesRetriever.py:42 |
| Constitution fallback top-k | 10 | RulesRetriever.py:199 `n_results=10` |
| Constitution fallback filter | `where={"document_type": {"$eq": "constitution"}}` — retrieves only the 374 constitution chunks, excluding the 245 evidence act chunks in master_rules | RulesRetriever.py:208 |
| Final rules returned | 5 (`max_results=5`) | RulesRetriever.py:461,405 |

---

#### C5. Query classification (Qclassifier.py)

| Parameter | Value | File:line |
|---|---|---|
| Mode A | `DIRECT_STATUTORY_LOOKUP` | Qclassifier.py:26 (via system prompt) |
| Mode B | `STRATEGIC_LEGAL_ANALYSIS` | Qclassifier.py system prompt |
| Mode C | `HYBRID_RETRIEVAL` | Qclassifier.py system prompt |
| Retriever trigger condition | app.py:1279–1283: `retrieval_mode in ["DIRECT_STATUTORY_LOOKUP", "HYBRID_RETRIEVAL"] or "RULES" in target_components or "STATUTORY_TEXT" in target_components` | app.py:1279–1283 |
| Case retrieval trigger | Always — `Qclassifier()` calls `retrieve_relevant_cases()` for all modes (Qclassifier.py:712) | Qclassifier.py:712 |
| Retries on malformed JSON | 3 (`max_retries=3`, Qclassifier.py:423 default) | Qclassifier.py:468 `for attempt in range(max_retries)` |
| Classifier temperature | 0.1 | Qclassifier.py:460 |
| Classifier max_tokens | 1500 | Qclassifier.py:461 |

---

#### C6. Labelled sources

| Label type | Maximum labels | File:line |
|---|---|---|
| Statute sources | 5 (S1–S5) | app.py:1469 `for i, rule in enumerate(statutory_provisions[:5], 1)` |
| Case sources | 3 (C1–C3) | app.py:1498 `for i, case in enumerate(case_law_results[:3], 1)` |
| Chunk aliases | S1.1, S1.2 … C1.1, C1.2 … | app.py:1480, app.py:1506 |

Built at: app.py:1463–1518 inside `generate_legal_answer()`.

---

#### C7. Relevance filter threshold

**Not a numeric threshold.** The filter at app.py:1621–1628 checks
`sr.get("relevant") is True` — a boolean field in `source_review`. Sources
where `relevant` is not exactly Python `True` are excluded. If the model
returns no source with `relevant: true`, all sources are kept (fallback at
app.py:1627–1628). There is no score threshold applied after `source_review`.

---

### D. Generation and Verification

#### D1. System prompt location and content

**Location:** app.py:1530–1565 (inside `generate_legal_answer()`)

**First 10 lines exactly:**
```
You are a Kenyan legal research assistant. Output ONLY a single valid JSON object. No text before or after it.

ABSOLUTE RULES:
1. Every legal statement MUST cite using exact aliases from the sources.
   Statute chunk aliases: [S1.1], [S1.2] etc. (use the most specific chunk alias).
   Bare [S#] may be used to refer to a whole statute article if you didn't use a specific chunk.
   Case chunks use [C#.#] (e.g. [C1.2]).
2. Never state a section number, article number, rule number, case name, holding,
   or procedure that does not appear verbatim or near-verbatim in the source text supplied.
3. Never say a source is wrong, outdated, or misapplied. If the user's premise
```

**10 JSON output fields (app.py:1551–1564):**
1. `source_review` — array: `{id, relevant, reason, key_points}`
2. `premise_check` — string
3. `short_answer` — string
4. `applicable_law` — string
5. `case_law_discussion` — string
6. `application` — string
7. `jurisdiction_and_procedure` — string
8. `gaps` — string
9. `follow_up_questions` — array (up to 3)
10. `citations_used` — array

> **CORRECTS EARLIER VALUE:** Previous audit said "9-field JSON schema". There are
> **10 fields** in the live schema (app.py:1551–1564). The earlier count excluded
> `source_review` from the count.

---

#### D2. Citation verification

**Reference types checked (citation_verifier.py:21–38):**

| Type | Regex |
|------|-------|
| sections | `\b(?:section|sec|s)\.?\s*(\d+[A-Z]?(?:\([^)]+\))*)` |
| articles | `\barticle\s+(\d+[A-Z]?(?:\([^)]+\))*)` |
| rules | `\brule\s+(\d+[A-Z]?(?:\([^)]+\))*)` |
| caps | `\bcap\.?\s*(\d+[A-Z]?)\b` |
| KLR citations | `\d{4}(?:KEHC|KECA|SCC|KEELC)\d+\(KLR\)` |
| case names | `X v Y` pattern |

**What counts as supported:** The numeric core of the cited reference (e.g., `39`
for `s.39(1)(a)`) must appear in the `article_number`, `full_text`, or
`text_preview` of at least one relevant source; OR for cases, the case name must
appear (case-insensitive substring) in a case name or chunk text.

**Hallucination log:** `backend/logs/hallucinations.jsonl` (citation_verifier.py:17–18)

**Regenerate-then-strip decision (exact code, citation_verifier.py:184–208):**
```python
# Attempt one regeneration
regen_text: Optional[str] = None
if regen_fn:
    try:
        regen_text = regen_fn(unsupported)
    except Exception:
        regen_text = None

if regen_text:
    valid2 = _valid_refs(relevant_sources)
    cits2 = extract_citations(regen_text)
    still_bad: List[str] = []
    for ctype, refs in cits2.items():
        for ref in refs:
            if not _supported(ref, ctype, valid2):
                type_label = ctype[:-1].upper() if ctype not in ("klr", "cases") else ctype.upper()
                still_bad.append(f"{type_label} {ref}")
    if not still_bad:
        return {"text": regen_text, "unverified_claims_removed": []}
    cleaned = _strip_sentences(regen_text, still_bad)
    return {"text": cleaned, "unverified_claims_removed": still_bad}

# Last resort: strip from original
cleaned = _strip_sentences(analysis_text, unsupported)
return {"text": cleaned, "unverified_claims_removed": unsupported}
```

**Regeneration rule:** one attempt; if still bad, sentence-level strip.

---

#### D3. RRF code (case retriever, exact — Retriever.py:88–94)

```python
def _rrf_fuse_c(ranked_a: list, ranked_b: list, k: int = 60, n: int = 30) -> list:
    rrf: dict = {}
    for rank, cid in enumerate(ranked_a, 1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(ranked_b, 1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(rrf, key=lambda x: -rrf[x])[:n]
```

Called at: Retriever.py:657 `fused_ids = _rrf_fuse_c(vec_ranked_ids, bm25_ranked_ids, k=60, n=30)`

*(Identical implementation in RulesRetriever.py:101–108 as `_rrf_fuse`.)*

---

#### D4. Firestore write

**Document path:** `users/{uid}/conversations/{conversation_id}/turns/{client_turn_id}`
where `client_turn_id` is used as the turn document ID (idempotency key).
(history_store.py:200+)

**Fields written to turn document:**
`status`, `query`, `answer_markdown`, `answer_structured` (dict of 8 structured
fields), `sources` (list, snippets ≤ 300 chars each), `alias_map`, `coverage`,
`retrieval_summary`, `unverified_claims_removed_count`, `model`, `latency_ms`,
`error`, `error_code`, `turn_index`, `created_at`

**Transaction:** `save_turn()` uses a Firestore transaction to atomically write
the turn and update the conversation metadata (history_store.py, `@db.transaction`
decorator).

**Size guard:** NOT DETERMINABLE FROM CODE — the 900 KB limit was referenced in
earlier documentation but the current `history_store.py` code does not contain
an explicit size check. The developer should verify.

---

#### D5. Background title generation

**history_store.py:418–470 (submit_title_generation / set_generated_title):**
- Model: `"qwen/qwen3.8-27b"` (call_groq default, history_store.py:441)
- temperature: `0.2` (history_store.py:446)
- max_tokens: `300` (history_store.py:447)
- timeout: `10` (history_store.py:448)
- Condition for writing: only if `conversation.title_source == "fallback"` at
  write time (history_store.py:466). A user rename (`title_source="user"`) is
  never overwritten.

---

### E. Authentication and Errors

#### E1. Sign-in providers (frontend)

- **Email/password:** `signInWithEmailAndPassword` (sign-in/page.tsx:52), `createUserWithEmailAndPassword` (sign-up/page.tsx:75)
- **Google:** `GoogleAuthProvider` + `signInWithPopup` (sign-in/page.tsx:64–66, sign-up/page.tsx:91–93)
- No other providers (no Facebook, GitHub, Apple, etc.) are present in the frontend code.

---

#### E2. require_role error responses (auth.py)

| Condition | HTTP status | JSON body | File:line |
|---|---|---|---|
| Missing `Authorization: Bearer` header | 401 | `{"error": "Missing Authorization header"}` | auth.py:67 |
| Firebase not configured on server | 503 | `{"error": "Firebase Admin not configured on server"}` | auth.py:46 |
| Token expired | 401 | `{"error": "Token expired"}` | auth.py:51 |
| Invalid/malformed token | 401 | `{"error": "Invalid token"}` | auth.py:53 |
| Firestore profile does not exist | 403 | `{"error": "Profile incomplete", "code": "PROFILE_REQUIRED"}` | auth.py:58 |
| Wrong role | 403 | `{"error": "Forbidden: requires role advocate or admin"}` | auth.py:108 |

---

#### E3. Frontend error and failure display

| Situation | User-visible text | File:line |
|---|---|---|
| History save failed ("not saved" banner) | `"This answer was not saved to history due to a temporary error. It is still accurate."` | ChatView.tsx:949 |
| Loading composer blocked | `"Send is disabled while an answer is generating."` | ChatView.tsx:989 |
| Failed assistant message header | `"Request failed"` | ChatView.tsx:1280 |
| Failed assistant — generic error | `msg.errorMessage \|\| msg.aiResponse \|\| "An error occurred."` | ChatView.tsx:1289 |
| Qretrieve failure | Throws `data.error \|\| "Retrieval failed"` → `errMsg` shown in failed card | ChatView.tsx:470 |
| All Groq keys exhausted (backend RuntimeError) | Returned as `"LLM call failed: …"` in the JSON `error` field → shown in failed card | app.py:1801–1811 |
| Coverage warning | `"Not in knowledge base: <acts>."` | ChatView.tsx:1381 |
| Unverified citations removed | `"<N> unverified citation(s) removed."` | ChatView.tsx:1382 |
| From history — failed turn | `"[Error] <turn.error \|\| 'Unknown error'>"` | ChatView.tsx:317 |
| Retry button | Shown when `historyStatus === "failed"` and `onRetry` is provided | ChatView.tsx:1291–1296 |

---

#### E4. advocateVerified

**Read in LIVE_RUNTIME:** No. The `require_role` decorator (auth.py) checks only
`profile.get("type", "")`. `advocateVerified` is never read in any LIVE_RUNTIME
file.

**Set at creation:** Yes, always `false` — frontend sign-up/page.tsx:78 and
auth/complete-profile/page.tsx:64 set it on write. bootstrap_admin.py:80 also
sets it to `False`.

**Used in frontend TypeScript type:** `frontend/app/lib/firestore.ts:15` — declared
as a field in the `UserProfile` type, but no frontend code reads or branches on it.

**Verdict:** `advocateVerified` is written on account creation and declared in the
type system but is never evaluated in any code path that controls access or
behaviour.

---

#### E5. Firestore rules and indexes

- `backend/firestore.rules` — confirmed to exist and match the data model written
  by LIVE_RUNTIME:
  - `users/{uid}` — write rules enforce 7 required fields, `type="advocate"`,
    `advocateVerified=false`, username uniqueness via `existsAfter`/`getAfter`.
  - `users/{uid}/conversations/{convId}` and `turns/{turnId}` — write=false
    (Admin SDK bypasses rules).
- `backend/firestore.indexes.json` — defines 3 composite indexes matching the
  history queries:
  1. `conversations`: `uid ASC, pinned DESC, updated_at DESC`
  2. `conversations`: `uid ASC, updated_at DESC`
  3. `turns`: `turn_index ASC`

---

### F. Packages and Reproducibility

#### F1. Definitive package list (LIVE_RUNTIME + LIVE_BUILD imports vs. requirements.txt and pip freeze)

| Package (import name) | Installed version | In requirements.txt? | Notes |
|---|---|---|---|
| flask | 3.1.2 | YES | |
| flask_cors | 6.0.1 | YES | |
| firebase_admin | 7.7.0 | YES | |
| google-cloud-firestore (google) | 2.34.1 | NO (implicit via firebase_admin) | |
| requests | 2.34.2 | YES (listed 2.32.5, installed 2.34.2) | version mismatch |
| chromadb | 1.3.5 | YES | |
| sentence_transformers | 5.1.2 | YES | |
| rank_bm25 | 0.2.2 | YES | |
| fitz (PyMuPDF) | 1.26.6 | YES (listed 1.24.14, installed 1.26.6) | version mismatch |
| PIL (Pillow) | 12.0.0 | YES (listed 11.1.0, installed 12.0.0) | version mismatch |
| pytesseract | 0.3.13 | YES | |
| python_dotenv | 1.2.1 | YES | |
| langchain_core | 1.0.6 | YES (via langchain) | |
| langchain_text_splitters | — | YES (via langchain) | |
| numpy | 2.3.5 | YES | |
| pypdf | 6.3.0 | YES | |
| sklearn (scikit-learn) | 1.7.2 | YES | |
| groq | 0.35.0 | YES | |
| openai | 2.9.0 | NO (CaseLaw/firac.py imports it; not in requirements.txt) | MISSING from requirements.txt |
| torch | 2.9.1 | YES (listed 2.9.1) | |
| transformers | 4.57.1 | YES | |
| rapidfuzz | 3.14.6 | NO (compute_constitution_bboxes.py imports it; not in requirements.txt) | MISSING from requirements.txt |
| opentelemetry-* | 1.45.1 | YES | required by chromadb |
| werkzeug | 3.1.3 | NO (indirect Flask dep) | |
| python_docx | 1.2.0 | YES | |
| ocrmypdf | 16.5.0 | YES | (system binary required) |

**Packages installed but NOT imported by any LIVE_RUNTIME or LIVE_BUILD file:**
```
anthropic, fastapi, fpdf2, img2pdf, opentelemetry-exporter-* (all — chromadb transitive deps), 
pdfminer.six, pikepdf, pypdfium2, pytest, pytest-mock, RapidFuzz (see above),
tenacity, uvicorn, uvloop, websocket-client, websockets
```
(These are labelled "NOT USED" in LIVE_RUNTIME/LIVE_BUILD context — they may
be used by admin tooling or legacy code.)

---

#### F2. python-docx

**Used in LIVE_BUILD** by `build_case_chunks.py` (DOCX fallback for `.docx`
case files). **Version:** `1.2.0` (pip freeze).

---

#### F3. Tesseract / ocrmypdf / pytesseract

| Component | LIVE_RUNTIME? | LIVE_BUILD? | File:line |
|---|---|---|---|
| pytesseract | YES (via constitutionlogic/constitutionbreaker.py, CaseLaw/firac.py) | YES (build_case_chunks.py, constitutionlogic/evidencedigester.py) | constitutionbreaker.py:6 |
| ocrmypdf | NO (system binary only; not imported by Python) | YES — `build_case_chunks.py` calls it via subprocess | build_case_chunks.py (subprocess call) |
| Tesseract (system binary) | Indirect — called by pytesseract and ocrmypdf | YES | system dependency |

---

#### F4. Neo4j in LIVE_RUNTIME or LIVE_BUILD

**No.** A search of all LIVE_RUNTIME and LIVE_BUILD files finds no `neo4j` import,
no Neo4j driver, and no Neo4j connection string. Neo4j commits (2026-02-12 and
2026-02-13) added a knowledge-graph tool that is not referenced by any live code path.

---

#### F5. Build and run commands

```bash
# ── BACKEND SETUP ──
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# ── DATA PIPELINE (run in order) ──

# Step 1: Extract constitution articles → JSON
python constitutionlogic/constitutionbreaker.py
# Output: constitutionlogic/Constsection/constitution_embeddable_chunks.json

# Step 2: Embed constitution into master_rules
python constitutionlogic/constitution_digester.py
# Target: constitutionlogic/MasterRulesDB/ collection: master_rules

# Step 3: Extract + embed Evidence Act into same master_rules
python constitutionlogic/evidencedigester.py
# Output: constitutionlogic/EvidenceSection/evidence_chapter3.json
# Target: constitutionlogic/MasterRulesDB/ collection: master_rules

# Step 4: Compute bounding boxes for constitution location rows
python compute_constitution_bboxes.py
# Updates: backend/locations.db (master_rules rows)

# Step 5: Chunk case PDFs/DOCX → JSONL + locations.db case_law rows
python build_case_chunks.py
# Reads: data/processed/cases/cases_meta.json, backend/SCCCases/
# Output: data/processed/cases/cases_embed.jsonl, backend/locations.db

# Step 6: Chunk statute sections → JSONL
python build_statute_chunks.py
# Reads: data/processed/statutes/sections.jsonl
# Output: data/processed/statutes/sections_embed.jsonl

# Step 7: Embed cases + statutes into ChromaDB
python embed_knowledge.py --source cases     # → CaseLawDB/case_law
python embed_knowledge.py --source statutes  # → StatutesDB/statutes

# ── ADMIN USER CREATION ──
python scripts/bootstrap_admin.py <email> <password> <username>

# ── BACKEND START ──
python app.py       # Flask on 0.0.0.0:5000, debug=True (app.py last line)

# ── FRONTEND ──
cd ../frontend
npm install
npm run dev         # Next.js development on port 3000 (default)
# or
npm run build && npm start   # production

# ── HEALTH CHECK ──
curl http://localhost:5000/api/health
# Expected: {"status": "ok"}
```

*(argparse CLIs for `embed_knowledge.py`: `--source {statutes,cases,both}`,
`--limit N`, `--suffix STR`, `--dry-run` — embed_knowledge.py:128–136)*

---

#### F6. README.md

**NOT up to date.** The current README describes a single-script Hugging Face
API system (`HF_API_TOKEN`, no Flask, no Firebase, no ChromaDB collections,
no three-route pipeline). It predates the entire Flask/Firebase/ChromaDB
architecture. Missing:
- Flask backend description
- Firebase Auth + Firestore
- Three ChromaDB collections and their purposes
- The two-call frontend pipeline (`/api/qretrieve` → `/api/generate-legal-answer`)
- Environment variable list for backend and frontend
- Build script run order
- Groq API (not Hugging Face)

---

#### F7. .env.example files vs. actual env vars

**Root `.env.example`:**
```
HF_API_TOKEN=your_huggingface_token_here
```
(completely outdated — no longer used)

**`frontend/.env.example`:**
```
NEXT_PUBLIC_FIREBASE_API_KEY=
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=
NEXT_PUBLIC_FIREBASE_PROJECT_ID=
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=
NEXT_PUBLIC_FIREBASE_APP_ID=
NEXT_PUBLIC_API_URL=http://localhost:5000
```
(frontend .env.example is accurate for the frontend)

**Env vars actually read by LIVE_RUNTIME (backend):**

| Variable | Read by | Present in backend .env.example? |
|---|---|---|
| `GROQ_API_KEY` | groq_client.py:28, Qclassifier.py:14, Retriever.py:39 | NO — no backend .env.example exists |
| `GROQ_API_KEY_2` | Qclassifier.py:14 | NO |
| `GROQ_API_KEY_3` | Qclassifier.py:14 | NO |
| `GROQ_API_KEY_*` (any) | groq_client.py:28–29 | NO |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | auth.py:20 | NO |
| `CORS_ORIGIN` | app.py:36 | NO |

**There is no `backend/.env.example`.** All backend environment variables are
undocumented in example files.

---

### G. Memory

NOT MEASURED (services were not confirmed to be running at audit time).

---

## 7. VERIFIED FACTS for Prompt C (Section 5.2)

| Fact | Value |
|------|-------|
| **Exact LLM model ID** | `"qwen/qwen3.8-27b"` (groq_client.py:20; also Qclassifier.py:26; also Retriever.py:43) |
| **master_rules record count** | **619** (374 Constitution + 245 Evidence Act; confirmed by live `col.count()`) |
| **Measured memory** | NOT MEASURED |
| **requirements.txt complete?** | **NO** — missing: `openai` (CaseLaw/firac.py), `rapidfuzz` (compute_constitution_bboxes.py); version mismatches: requests (2.32.5 vs 2.34.2 installed), PyMuPDF (1.24.14 vs 1.26.6 installed), Pillow (11.1.0 vs 12.0.0 installed) |
| **README up to date?** | **NO** — describes a single-script Hugging Face prototype; no mention of Flask, Firebase, ChromaDB, Groq |
| **Live package versions (key packages)** | flask 3.1.2, chromadb 1.3.5, sentence-transformers 5.1.2, rank-bm25 0.2.2, firebase-admin 7.7.0, PyMuPDF 1.26.6, torch 2.9.1, transformers 4.57.1, groq 0.35.0, requests 2.34.2, python-dotenv 1.2.1 |
| **Constitution pipeline command** | `python constitutionlogic/constitutionbreaker.py` then `python constitutionlogic/constitution_digester.py` |
| **python-docx version** | 1.2.0 |

---

## 8. VERIFIED FACTS for Prompt D (Sections 5.3–5.5)

| Fact | Value |
|------|-------|
| **Exact LLM model ID** | `"qwen/qwen3.8-27b"` |
| **master_rules record count** | 619 (374 constitution + 245 evidence act; constitution fallback filters to 374 via `where document_type="constitution"`) |
| **Exact chunks per case** | 2001KECA154(KLR): 7, 2010KECA346(KLR): 31, 2014KEHC6468(KLR): 3, 2020KEHC8905(KLR): 5, 2024KEHC15763(KLR): 10, 2024KEHC17038(KLR): 8, 2026SCC235(KLR): 4, 2026SCC241(KLR): 6, 2026SCC242(KLR): 4 — all ocr=False |
| **locations.db highlight result** | case_law: 72/78 chunks indexed, all 72 have real bboxes (PDF highlight works for all indexed chunks); master_rules: 374 rows, 252 with real bboxes (122 articles open PDF at page estimate but do not highlight); statutes: no rows (text reader only) |
| **Land Act present?** | NOT PRESENT in live collection or manifest |
| **Employment Act present?** | PRESENT (111 chunks, slug: employment_act) |
| **Reason for act selection** | NOT DETERMINABLE FROM CODE (no commit message or config file explains statute selection) |

### Retrieval and Generation Settings Table

| Parameter | Value | File:line |
|---|---|---|
| Embedding model | intfloat/e5-base-v2 | config.py:27 |
| Embedding loading | Per request (not at startup) | Retriever.py:600, RulesRetriever.py:497 |
| Query prefix | `"query: "` | Retriever.py:625, RulesRetriever.py:539 |
| Passage prefix | `"passage: "` | embed_knowledge.py:110 |
| Cross-encoder model | cross-encoder/ms-marco-MiniLM-L-6-v2 | Retriever.py:84, RulesRetriever.py:78 |
| CE score normalisation | sigmoid | Retriever.py:61–62 |
| BM25 top-k (both retrievers) | 50 | Retriever.py:641, RulesRetriever.py:556 |
| Vector top-k (both retrievers) | min(50, collection.count()) | Retriever.py:645, RulesRetriever.py:562 |
| RRF k | 60 | Retriever.py:657, RulesRetriever.py:574 |
| RRF n | 30 | Retriever.py:657, RulesRetriever.py:574 |
| Cases returned to LLM | top 3 parents × up to 3 chunks each | Retriever.py:47, app.py:1504 |
| Statutes returned to LLM | top 5 | RulesRetriever.py:405,461 |
| Low-confidence threshold | 0.4 (warning only) | Retriever.py:53 |
| Schedule penalty | 0.3× | RulesRetriever.py:115 |
| Constitution fallback top-k | 10 | RulesRetriever.py:199 |
| Classifier model | qwen/qwen3.8-27b | Qclassifier.py:26 |
| Classifier temperature | 0.1 | Qclassifier.py:460 |
| Classifier max_tokens | 1 500 | Qclassifier.py:461 |
| Classifier timeout | 30 s | Qclassifier.py:477 |
| Synthesis model | qwen/qwen3.8-27b | groq_client.py:20 |
| Synthesis temperature | 0 | groq_client.py:37 |
| Synthesis max_tokens | 2 500 | groq_client.py:38 |
| Synthesis timeout | 60 s | groq_client.py:39 |
| JSON output fields | 10 | app.py:1551–1564 |
| Max S labels | 5 (S1–S5) | app.py:1469 |
| Max C labels | 3 (C1–C3) | app.py:1498 |
| Relevance filter | boolean `relevant is True` (no numeric threshold) | app.py:1621–1628 |
| Citation regen attempts | 1 | citation_verifier.py:184 |
| Hallucination log | backend/logs/hallucinations.jsonl | citation_verifier.py:17 |
| Title model | qwen/qwen3.8-27b | groq_client.py:20 (call_groq default) |
| Title temperature | 0.2 | history_store.py:446 |
| Title max_tokens | 300 | history_store.py:447 |
| Title timeout | 10 s | history_store.py:448 |

### RRF Code Extract

```python
# Retriever.py:88–94 — identical logic in RulesRetriever.py:101–108
def _rrf_fuse_c(ranked_a: list, ranked_b: list, k: int = 60, n: int = 30) -> list:
    rrf: dict = {}
    for rank, cid in enumerate(ranked_a, 1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(ranked_b, 1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(rrf, key=lambda x: -rrf[x])[:n]
```

Called: `Retriever.py:657 fused_ids = _rrf_fuse_c(vec_ranked_ids, bm25_ranked_ids, k=60, n=30)`

### Citation Verification Code Extract

```python
# citation_verifier.py:184–208
# Attempt one regeneration
regen_text: Optional[str] = None
if regen_fn:
    try:
        regen_text = regen_fn(unsupported)
    except Exception:
        regen_text = None

if regen_text:
    valid2 = _valid_refs(relevant_sources)
    cits2 = extract_citations(regen_text)
    still_bad: List[str] = []
    for ctype, refs in cits2.items():
        for ref in refs:
            if not _supported(ref, ctype, valid2):
                type_label = ctype[:-1].upper() if ctype not in ("klr", "cases") else ctype.upper()
                still_bad.append(f"{type_TYPE} {ref}")
    if not still_bad:
        return {"text": regen_text, "unverified_claims_removed": []}
    cleaned = _strip_sentences(regen_text, still_bad)
    return {"text": cleaned, "unverified_claims_removed": still_bad}

# Last resort: strip from original
cleaned = _strip_sentences(analysis_text, unsupported)
return {"text": cleaned, "unverified_claims_removed": unsupported}
```

### Exact Error Messages Shown to Users (frontend)

| Context | Text shown |
|---|---|
| Failed card header | "Request failed" |
| Failed card body | `msg.errorMessage || msg.aiResponse || "An error occurred."` |
| Not-saved banner | "This answer was not saved to history due to a temporary error. It is still accurate." |
| Sending blocked | "Send is disabled while an answer is generating." |
| Coverage warning | "Not in knowledge base: <act names>." |
| Citations removed | "<N> unverified citation(s) removed." |
| History failed turn | "[Error] <turn.error || 'Unknown error'>" |
| qretrieve failure | The backend `data.error` string, or "Retrieval failed" |
| All keys exhausted (backend) | "LLM call failed: All Groq keys exhausted" (via requests exception path, app.py:1801–1811) |

---

## 9. STILL NEEDS THE DEVELOPER

The following items cannot be determined from code alone:

1. **Why 2 cases are absent from `case_law`:** `cases_meta.json` lists
   `2011KEHC1481(KLR)` and `2021KEELC220(KLR)` with no exclusion flag, yet they
   are not in the collection. Check whether `build_case_chunks.py` failed silently
   for their source files, or whether `embed_knowledge.py` was run before those
   files were added to `SCCCases/`.

2. **Statute selection rationale:** Why these 14 acts were chosen (Bills of
   Exchange, KICA, Stamp Duty Act, etc.) over other acts mentioned in earlier
   documentation (Land Act, Hire Purchase Act, etc.). No commit message or
   configuration document explains this.

3. **Firestore size guard:** The 900 KB limit on turn documents referenced in
   earlier documentation does not appear in the current `history_store.py`. Confirm
   whether it was removed or is still being enforced elsewhere.

4. **Browser tested:** Not determinable from code — which browser was used for
   end-to-end testing.

5. **Memory usage:** `free -h` and `ps` figures were not collected because the
   services were not confirmed to be running. Run the backend and frontend, wait
   30 seconds for model loading to complete, then run:
   ```bash
   free -h
   ps -eo comm,rss,%mem --sort=-rss | grep -iE "python|node" | head -10
   ```

6. **Constitution bboxes — 122 missing articles:** `compute_constitution_bboxes.py`
   uses `rapidfuzz` word-alignment to locate article text on PDF pages, but 122
   of 374 articles have `boxes_json='[]'`. Confirm whether this is a known
   limitation of the matching algorithm or a script failure.

7. **`Retriever.py` generates LLM case summaries when `skip_summaries=False`:**
   The route `/api/qretrieve` currently calls `retrieve_relevant_cases` with
   `skip_summaries=True` (Qclassifier.py:715). The LLM-summary path
   (`generate_case_summary`, Retriever.py:101) is therefore never executed in the
   live pipeline. Confirm this is intentional (summaries delegated to synthesis stage).
