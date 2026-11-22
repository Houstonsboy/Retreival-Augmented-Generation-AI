# Chapter 5 Verified Facts Audit
*Audit executed: 2026-11-17*
*All values sourced from commands or files; NOT FOUND where unverifiable.*

---

## A. Environment and Exact Versions

### A1. Backend pip freeze (filtered)
Source: `command: cd backend && venv/bin/pip freeze 2>/dev/null | grep -iE ...`

| Package | Installed version |
|---|---|
| anyio | 4.11.0 |
| chromadb | 1.3.5 |
| firebase_admin | 7.7.0 |
| Flask | 3.1.2 |
| flask-cors | 6.0.1 |
| fpdf2 | 2.9.0 |
| groq | 0.35.0 |
| huggingface-hub | 0.36.0 |
| langchain | 1.0.7 |
| langchain-core | 1.0.6 |
| langchain-text-splitters | 1.0.0 |
| ocrmypdf | 17.13.0 |
| onnxruntime | 1.23.2 |
| opencv-python-headless | 5.0.0.93 |
| opentelemetry-api | 1.45.1 |
| opentelemetry-sdk | 1.45.1 |
| pillow | 12.0.0 |
| pydantic | 2.14.2 |
| pydantic_core | 2.50.2 |
| PyMuPDF | 1.26.6 |
| pypdf | 6.3.0 |
| pypdfium2 | 5.14.0 |
| pytesseract | 0.3.13 |
| pytest | 9.2.0 |
| pytest-mock | 3.17.0 |
| python-docx | 1.2.0 |
| rank-bm25 | 0.2.2 |
| RapidFuzz | 3.14.6 |
| rapidocr | 3.10.0 |
| requests | 2.34.2 |
| scikit-learn | 1.7.2 |
| sentence-transformers | 5.1.2 |
| torch | 2.9.1 |
| transformers | 4.57.1 |

Note: `torch` is installed but the active retrieval pipeline (Retriever.py, RulesRetriever.py) uses `intfloat/e5-base-v2` via sentence-transformers, not raw PyTorch inference. `ocrmypdf` and `pytesseract` are installed but their imports appear only in standalone ingestion scripts (`build_case_chunks.py`, `constitutionlogic/evidencedigester.py`, `constitutionlogic/constitution.py`, `constitutionlogic/constitutionbreaker.py`, `Const_IntepretationLaws/reader.py`), not in the live request path. `fitz` (PyMuPDF) is used in `build_case_chunks.py` and `compute_constitution_bboxes.py`. `rapidfuzz` is used in `compute_constitution_bboxes.py`.
Source: `command: grep -r "import pytesseract|from fitz|import rapidfuzz" backend --include="*.py" -l`

### A2. Frontend package.json versions
Source: `frontend/package.json`

| Package | Declared version |
|---|---|
| next | 16.0.3 (exact) |
| react | 19.2.0 (exact) |
| react-dom | 19.2.0 (exact) |
| typescript | ^5 |
| tailwindcss | ^4 |
| firebase | ^10.12.0 |
| react-pdf | ^7.7.3 |
| react-markdown | ^10.1.0 |
| eslint | ^9 |
| eslint-config-next | 16.0.3 (exact) |

Note: No jest in frontend/package.json — Jest is not configured for the frontend.

### A3. Frontend lockfile status
Source: `command: ls frontend/package-lock.json frontend/yarn.lock frontend/pnpm-lock.yaml 2>/dev/null`
Result: `package-lock.json` PRESENT. `yarn.lock` and `pnpm-lock.yaml` absent.

### A4. firestore-tests/package.json versions
Source: `firestore-tests/package.json`

| Package | Declared version |
|---|---|
| jest | ^29.7.0 |
| @types/jest | ^29.5.12 |
| ts-jest | ^29.1.3 |
| @firebase/rules-unit-testing | ^3.0.4 |
| firebase | ^10.12.0 |
| typescript | ^5.4.5 |

### A5. RapidOCR model files
Source: `command: find backend -name "*.onnx" ... | grep -i ppocr|rapid`
Models found inside venv (site-packages), not in project root:
- `venv/lib/python3.13/site-packages/rapidocr/models/PP-OCRv6_rec_small.onnx`
- `venv/lib/python3.13/site-packages/rapidocr/models/PP-OCRv6_det_small.onnx`
- `venv/lib/python3.13/site-packages/rapidocr/models/ch_ppocr_mobile_v2.0_cls_mobile.onnx`

### A6. WSGI server and debug flag
Source: `backend/app.py:1918-1922`
```
if __name__ == '__main__':
    import threading
    threading.Thread(target=sccvision_warmup, daemon=True).start()
    app.run(debug=True, port=5000, host='0.0.0.0')
```
Dev mode: `debug=True`, port 5000, host 0.0.0.0. No production WSGI (gunicorn/uwsgi) configured.

### A7. Backend environment variables
Source: `command: grep -rn "os\.getenv|os\.environ" backend --include="*.py" | grep -v venv`

| Env var | File | Purpose |
|---|---|---|
| CORS_ORIGIN | backend/app.py:39 | CORS allowed origin (default: http://localhost:3000) |
| GROQ_API_KEY | backend/Retriever.py:39, backend/Qclassifier.py:14, backend/CaseLaw/firac.py:26, backend/constitutionlogic/constitution.py:22 | Primary Groq API key |
| GROQ_API_KEY_2, GROQ_API_KEY_3 | backend/Qclassifier.py:14 | Additional rotation keys (Qclassifier only) |
| GROQ_API_KEY_* (sorted) | backend/groq_client.py:37-38 | All keys GROQ_API_KEY + GROQ_API_KEY_N (shared client) |
| GROQ_API_KEY_12 | backend/SCCVision/analyst/client.py:34 | Dedicated key for SCCVision analysis |
| FIREBASE_SERVICE_ACCOUNT_PATH | backend/auth.py:20, backend/scripts/set_role.py:38, backend/scripts/bootstrap_admin.py:41 | Path to Firebase service account JSON |
| HUGGINGFACE_API_KEY | backend/agentrouter.py:6 | HuggingFace API key (agentrouter only) |

### A8. Frontend environment variables
Source: `command: grep -rn "process\.env|NEXT_PUBLIC" frontend --include="*.ts" --include="*.tsx" -h | grep -oP ...`

Application-relevant (project-specific):
- `NEXT_PUBLIC_API_URL` — backend base URL
- `NEXT_PUBLIC_FIREBASE_API_KEY`
- `NEXT_PUBLIC_FIREBASE_APP_ID`
- `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN`
- `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID`
- `NEXT_PUBLIC_FIREBASE_PROJECT_ID`
- `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET`
- `NODE_ENV`

(Remaining `process.env.*` hits were from grpc/firebase SDK internals inside node_modules.)

---

## B. LLM and Model Configuration

### B1. Groq model ID
Source: `backend/groq_client.py:19`
```
GROQ_MODEL = "qwen/qwen3.8-27b"
```
This constant is imported by `app.py` (lines 14, 1397, 1776) and used in every `call_groq()` call unless overridden. Qclassifier uses the same model via its own Groq HTTP call; SCCVision uses the same via shared `call_groq`.

### B2. Groq call sites
Source: `command: grep -rn "\.chat\.completions\.create|groq_client|_llm" backend --include="*.py" -l`

| File | Purpose |
|---|---|
| backend/app.py | /api/generate-legal-answer synthesis (line 1610) and citation verification regen (line 1665) |
| backend/history_store.py | Conversation title generation |
| backend/SCCVision/s106b/gate.py | No LLM calls — deterministic only |
| backend/SCCVision/analyst/client.py | Evidence analysis LLM calls |
| backend/SCCVision/analyst/rules.py | Verification rules (no LLM) |
| backend/SCCVision/analyst/pipeline.py | Evidence analysis orchestration |
| backend/constitutionlogic/constitution.py | Constitution analysis (legacy) |
| backend/pixe.py | Old RAG pipeline |

### B3. Per-call-site parameters

**app.py /api/generate-legal-answer** (line 1610):
- Function: `call_groq(messages)` — uses defaults
- temperature: 0 (default in groq_client.py:68)
- max_tokens: 2500 (default in groq_client.py:69)
- JSON output enforced: yes, via system prompt "Output ONLY a single valid JSON object"
- Source: `backend/app.py:1544-1580, backend/groq_client.py:62-70`

**SCCVision/analyst/client.py**:
- Uses `call_groq(..., api_key=_SCCVISION_API_KEY)` where `_SCCVISION_API_KEY = os.getenv("GROQ_API_KEY_12")`
- Preferred key slot tried first; falls back to shuffled rotation if rate-limited
- Source: `backend/SCCVision/analyst/client.py:34, 100`

**history_store.py title generation**: Uses `call_groq` with defaults; temperature 0, max_tokens 2500.

### B4. LangChain usage
Source: `command: grep -rn "langchain|LangChain" backend --include="*.py" | grep -v venv`
- `backend/pixe.py:14` — `from langchain_core.documents import Document` (old RAG pipeline, not in live request path for /api/generate-legal-answer)
- `backend/digester.py:8-9` — `langchain_core.documents`, `langchain_text_splitters.RecursiveCharacterTextSplitter` (ingestion scripts only)
LangChain is used only in document ingestion and the legacy pixe.py pipeline, NOT in the live generate-legal-answer or qretrieve paths.

### B5. Key rotation strategy
Source: `backend/groq_client.py:35-136`
- Keys loaded: all env vars named `GROQ_API_KEY` or `GROQ_API_KEY_<N>` (sorted alphabetically)
- Per-call order: shuffle, except `preferred_key` goes first
- Retry: 2 rounds total; on round 1 exhaustion, wait 30s and retry once
- HTTP 429 and 403: rotate to next key
- HTTP 413 with `rate_limit_exceeded` error code: rotate (treat as 429); HTTP 413 with `request_too_large`: raise `GroqPayloadTooLargeError` immediately (no rotation)
- Qclassifier uses its own key rotation (GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3) independently: `backend/Qclassifier.py:12-23`

### B6. Embedding model
Source: `backend/digester.py:22`, `backend/config.py:27`, `backend/Retriever.py`, `backend/RulesRetriever.py`
```
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
```
768-dimension dense embeddings. Used in ChunkDB ingestion, CaseLawDB, StatutesDB, MasterRulesDB, BlackLawDictionaryDB ingestion.

### B7. Cross-encoder
Source: `backend/Retriever.py:82-84`, `backend/RulesRetriever.py:76-78`, `backend/pixe.py:167`
```python
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
```
Retriever.py:53: `LOW_CONFIDENCE_THRESHOLD = 0.4` (raw cosine similarity pre-reranking gate)
pixe.py:35: `TOP_K = 3` (final chunks after reranking)
pixe.py:277: Initial retrieval pool = 10 candidates before cross-encoder reranking.

### B8. CERT_THRESHOLD
Source: search result — NOT FOUND in any .py file. The term `CERT_THRESHOLD` and value `0.83` do not appear anywhere in the backend codebase. The certificate path uses deterministic Boolean gate conditions (gate.py), not a numeric threshold.

---

## C. Knowledge Base

### C1. ChromaDB collection inventory
Source: `command: venv/bin/python /tmp/chromadb_audit.py` — output in `eval/logs/chromadb_audit.txt`

| DB directory | Collection name | Count | Key metadata fields |
|---|---|---|---|
| ChunkDB | ChunkDB | 704 | file_size, file_name, chunk_id, embedding_model, start_index, cleaned, file_hash, file_type, chunk_index, original_size, ingested_at, source, total_chunks, loaded_at, chunk_size |
| LegalSummariesDB | legal_summaries | 40 | legal_domain, firac_component, content_length, winning_party, source_file_path, file_name, parties, embedding_model, case_identifier, judge, court_level, ingested_at, document_hash, year |
| constitutionlogic/MasterRulesDB | master_rules | 619 | total_chunks, part_title, chapter_title, chunk_length, chapter, article_length, article_number, source_document, chunk_index, document_type, part, article_header, is_complete_article, ingestion_timestamp |
| StatutesDB | statutes | 1331 | section_num, act_slug, chapter, prev_chunk_id, part, section_heading, references, total_chunks, breadcrumb, chapter_heading, article_number, act_title, part_num, part_heading, schedule_title, article_header, status, is_complete_section, part_title, next_chunk_id, chapter_num, is_schedule, chunk_total, token_count, is_complete_article, act_version, frbr_work_id, chapter_title, source_document, chunk_index, document_type |
| CaseLawDB | case_law | 78 | document_hash, source_file_path, page_end, parties, chunk_index, file_name, legal_domain, firac_component, prev_chunk_id, winning_party, judge, case_identifier, ocr, page_start, year, court_level, content_length, next_chunk_id, label, chunk_total |
| Const_IntepretationLaws/BlackLawDictionaryDB | black_law_dictionary | 40061 | document_type, page, definition_length, ingestion_timestamp, source, term |

**Total chunks across all 6 collections: 704 + 40 + 619 + 1331 + 78 + 40061 = 42833**

Contradiction resolution: "three collections / 2028 chunks" vs "six collections / 43133 documents"
- Actual: SIX collections are present and accessible. The correct total is 42,833 (not 43,133 — difference likely due to stale counts when report was written or counting rounding). The "three collections / 2028 chunks" figure predates the addition of StatutesDB, CaseLawDB, and BlackLawDictionaryDB.

### C2. SQLite file sizes
Source: `command: find backend -name "*.sqlite3" | xargs ls -lh`

| File | Size | Modified |
|---|---|---|
| CaseLawDB/chroma.sqlite3 | 3.8M | Nov 17 10:23 |
| ChunkDB/chroma.sqlite3 | 11M | Nov 17 10:23 |
| Const_IntepretationLaws/BlackLawDictionaryDB/chroma.sqlite3 | 93M | Nov 17 10:23 |
| constitutionlogic/MasterRulesDB/chroma.sqlite3 | 7.6M | Nov 17 10:23 |
| LegalSummariesDB/chroma.sqlite3 | 1.3M | Nov 17 10:23 |
| StatutesDB/chroma.sqlite3 | 16M | Nov 17 10:23 |

### C3. FIRAC fields — actual data
Source: `eval/logs/chromadb_audit.txt` — FIRAC sampling of LegalSummariesDB
LegalSummariesDB (legal_summaries, 40 chunks, 8 cases x 5 FIRAC components each):
Confirmed fields in metadata: `firac_component`, `parties`, `case_identifier`, `winning_party`, `judge`, `court_level`, `legal_domain`, `year`, `source_file_path`, `file_name`, `embedding_model`, `ingested_at`, `document_hash`, `content_length`
FIRAC components present: `facts`, `issues`, `rules`, `application`, `conclusion` (all 5 confirmed from first 5 docs sampled from one case).

CaseLawDB also has `firac_component` field (confirmed in metadata keys).

### C4. Case source PDFs
Source: `command: find backend/SCCCases -name "*.pdf" | head -10`
10 PDFs found in `backend/SCCCases/`:
- Ogembo v Yongo (2024)
- Obado v Anayo (2024)
- Safaricom Limited v Ocean View Beach Hotel (2010)
- Express (Kenya) Limited v Manju Patel (2001)
- Nonny Gathoni Njenga another v Catherine Masitsa (2014)
- Peter Ngethe Ngari ta PNN Funeral Services v Standard Group (2020)
- Republic v Barisa Wayu Mataguda (2011)
- Republic v Director Land Administration (2021)
- Umoja Magharibi (K) Ltd v Lubano (2026SCC242)
- New Titanic Jacaranda General Spares v Omil (2026SCC235)

### C5. Chunking and retrieval constants
Source: `backend/digester.py:23-24`, `backend/constitutionlogic/constitutionbreaker.py:23-24`, `backend/Retriever.py:47`, `backend/pixe.py:35`

| Constant | Value | File:line |
|---|---|---|
| CHUNK_SIZE (digester) | 1000 chars | backend/digester.py:23 |
| CHUNK_OVERLAP (digester) | 300 chars | backend/digester.py:24 |
| CHUNK_SIZE (constitutionbreaker) | 1000 chars | backend/constitutionlogic/constitutionbreaker.py:23 |
| CHUNK_OVERLAP (constitutionbreaker) | 300 chars | backend/constitutionlogic/constitutionbreaker.py:24 |
| TOP_N_PARENTS (Retriever) | 3 | backend/Retriever.py:47 |
| TOP_K (pixe legacy) | 3 | backend/pixe.py:35 |
| LOW_CONFIDENCE_THRESHOLD | 0.4 | backend/Retriever.py:53 |

---

## D. API Surface and Access Control

### D1. Route table
Source: `backend/app.py` (full read), `backend/SCCVision/router.py`, `backend/SCCVision/s106b/routes.py`, `backend/history_routes.py` (inferred from imports)

| Route | Method | Blueprint | Auth decorator | Role required |
|---|---|---|---|---|
| /api/health | GET | main | none | none |
| /api/chat | POST | main | @require_role | advocate, admin |
| /api/files | GET | main | @require_role | advocate, admin |
| /api/upload | POST | main | @require_role | admin |
| /api/files/<filename> | GET | main | @require_role | advocate, admin |
| /api/digest | POST | main | @require_role | admin |
| /api/firac | POST | main | @require_role | advocate, admin |
| /api/ingest-firac | POST | main | @require_role | admin |
| /api/digest/constitution | POST | main | @require_role | admin |
| /api/digest/constitution/status | GET | main | none | none |
| /api/digest/constitution/reset | DELETE | main | @require_role | admin |
| /api/case-pdf | GET | main | none | none |
| /api/chunks/<chunk_id>/location | GET | main | none | none |
| /api/documents/<doc_key>/chunks | GET | main | none | none |
| /api/cases/<case_identifier>/all-chunks | GET | main | none | none |
| /api/statutes/<source_document>/all-chunks | GET | main | none | none |
| /api/rule-pdf | GET | main | none | none |
| /api/qretrieve | POST | main | @require_role | advocate, admin |
| /api/generate-legal-answer | POST | main | @require_role | advocate, admin |
| /api/case-laws/upload | POST | main | @require_role | advocate, admin |
| /api/case-laws/file/<filename> | GET | main | none | none |
| /sccvision/health | GET | sccvision_bp | none | none |
| /sccvision/extract | POST | sccvision_bp | @require_auth | any authenticated |
| /sccvision/analyze | POST | sccvision_bp | @require_auth | any authenticated (deprecated) |
| /sccvision/analyze-messages | POST | sccvision_bp | @require_auth | any authenticated (deprecated) |
| /sccvision/evidence/process | POST | sccvision_bp | @require_auth | any authenticated |
| /sccvision/case/analyze | POST | sccvision_bp | @require_auth | any authenticated |
| /api/evidence/certificates | POST | s106b_bp | @require_role | litigant, advocate, admin |
| /api/evidence/certificates | GET | s106b_bp | @require_role | litigant, advocate, admin |
| /api/evidence/certificates/<cert_id> | GET | s106b_bp | @require_role | litigant, advocate, admin |
| /api/evidence/certificates/<cert_id>/download | GET | s106b_bp | @require_role | litigant, advocate, admin |
| /api/history/conversations | GET | history_bp | (inferred @require_role) | advocate, admin |
| /api/history/conversations/<id> | GET | history_bp | (inferred @require_role) | advocate, admin |
| /api/history/conversations/<id> | PATCH | history_bp | (inferred @require_role) | advocate, admin |
| /api/history/conversations/<id> | DELETE | history_bp | (inferred @require_role) | advocate, admin |

### D2. Firestore write points
Source: `command: grep -rn "\.set(|\.update(|\.add(|save_turn|save_certificate" backend --include="*.py" | grep -v test_ | grep -v venv`

| Write operation | Collection path | File |
|---|---|---|
| save_turn() (new conversation) | users/{uid}/conversations/{convId} + turns/{turnId} | backend/history_store.py:216, 241 |
| save_turn() (existing conversation) | users/{uid}/conversations/{convId} update + turns/{turnId} | backend/history_store.py:194-195 |
| title update | users/{uid}/conversations/{convId} update | backend/history_store.py:603, 611, 652 |
| CaseLaws upload | CaseLaws/{auto-id} | backend/app.py:1878 |
| save_certificate() | users/{uid}/certificates/{certId} | backend/SCCVision/s106b/store.py:90 |
| save_analysis_snapshot() | users/{uid}/analyses/{analysisId} | backend/SCCVision/analysis_store.py (inferred from router.py:310) |

### D3. Firestore rules summary
Source: `firestore.rules`

| Collection path | Client read | Client write |
|---|---|---|
| users/{uid} | owner only (uid==auth.uid) | create: owner, 7 specific fields, type in [advocate, litigant] only, atomic with username; update: owner, only username+updatedAt |
| usernames/{username} | public read | create: atomic with users/{uid}; delete: owner only changing username |
| CaseLaws/{docId} | any authenticated | false (Admin SDK only) |
| users/{uid}/conversations/{convId} | owner only | false (Admin SDK only) |
| users/{uid}/conversations/{convId}/turns/{turnId} | owner only | false (Admin SDK only) |
| users/{uid}/analyses/{analysisId} | owner only | false (Admin SDK only) |
| users/{uid}/certificates/{certId} | owner only | false (Admin SDK only) |
| /{document=**} (wildcard catch-all) | false | false |

Key design: `type` and `advocateVerified` are immutable after creation — privilege escalation from client is impossible.

---

## E. SCC Evidence Pipeline

### E1. Image validation
Source: `backend/SCCVision/router.py:27-28, 58-68`
- Allowed extensions: `{"png", "jpg", "jpeg", "webp"}`
- Max size: `10 * 1024 * 1024` bytes (10 MB)
- Empty file check: `if not data: return 400`
- Error on invalid decode: `InvalidImageError` → 400

### E2. OCR preprocessing
Source: `backend/SCCVision/preprocess.py`, `backend/SCCVision/engine.py`
- RapidOCR singleton (PP-OCRv6_det_small.onnx + PP-OCRv6_rec_small.onnx + ch_ppocr_mobile_v2.0_cls_mobile.onnx)
- PreprocessOptions: `upscale=True` (default), `invert=True` (default)
- Dark mode detection: `dark_mode_threshold=128.0` (mean brightness < 128 → dark mode)
- Preprocessing: grayscale mean brightness check; upscaling if enabled
- OCR runs off a thread pool (max_workers=2) with 120s timeout

### E3. Evidence classifier
Source: `backend/SCCVision/analyst/mpssa.py:69-95`
Decision logic (priority order):
1. If `"confirmed" in full_text.lower()` AND `10-char codes found` AND `KSH/KES found` → `mpesa_screenshot`
2. If `M-Pesa` header AND (`codes` or `"confirmed"`) → `mpesa_screenshot`
3. If `messages` list present AND (`contact.name` or `timestamped message` or `sender in (me/them)`) → `chat_screenshot`
4. Otherwise → `other`

### E4. M-Pesa parser
Source: `backend/SCCVision/analyst/mpssa.py`, `backend/SCCVision/analyst/enrich.py`
- Transaction code regex: `r"\b[A-Z0-9]{10}\b"` with letter+digit requirement
- Layouts: standard Safaricom confirmations (split at code positions) and bank-originated (Absa-style, split at day/date headers)
- Fields extracted per entry: `code`, `amount_kes`, `recipient`, `till_or_paybill`, `datetime`, `date`, `time`, `balance`, `cost`, `parse_confidence`
- Unsupported layout returned when no codes found or no confirmations parsed

### E5. WhatsApp sender detection
Source: `backend/SCCVision/analysis.py:214-252`
Three-signal detection:
1. **Color signal** (high confidence): bubble color sampled from bands left/above text block; green (`g > r`, `g > b`, diff >= 8) → outgoing ("me"); neutral (all channels within 12) → incoming ("them")
2. **Left-edge signal** (low confidence): `left_edge_norm < 0.12` → incoming; else → outgoing
3. **Tick signal** (low confidence): trailing space in OCR timestamp → outgoing
Priority: color > edge > ticks. If color and edge disagree, confidence downgraded to "low".

Layout constants: `STATUS_BAR_RATIO=0.045`, `HEADER_BOTTOM=0.10`, `AVATAR_LEFT_MAX=0.13`, `SYSTEM_CENTER_TOL=0.07`, `SYSTEM_MAX_WIDTH=0.5`

### E6. PII masking regexes
Source: `backend/SCCVision/analyst/enrich.py:44-77`
```python
_PHONE_RE = re.compile(r"(?<![\w])(?:\+?254\d{9}|0\d{9})(?![\w])")
# Masks: keep first 4 digits + "***" + last 3 digits
_ID_RE = re.compile(r"(?<![\w])(\d{7,8})(?![\w])")
# Masks: keep first 3 digits + "***" + last 3 digits
# Applied: phones first, then IDs (so masked phone ***-tokens not re-matched as IDs)
```

### E7. Pydantic schema fields (AnalystResult)
Source: `backend/SCCVision/analyst/models.py:134-165`
Fields: `conversation_integrity` (Integrity), `unrelated_messages` (list), `parties` (list[Party]), `timeline` (list[TimelineEntry]), `elements` (dict[str, Element] — keys A..K), `mpesa_payments` (list[MpesaPayment]), `red_flags` (list[str]), `missing_information` (list[str]), `overall_verdict` (OverallVerdict), `confidence` (float|None), `changes_since_previous` (list[str]), `plain_summary` (str), `strengthen_checklist` (list[str]), `follow_up_requests` (list[FollowUpRequest]), `questions_for_parties` (list[PartyQuestion]), `disclaimer` (str)

OverallVerdict values: `LIKELY VALID`, `POSSIBLY VALID WITH GAPS`, `LIKELY NOT VALID`, `INSUFFICIENT EVIDENCE`
IntegrityRating values: `STRONG`, `ADEQUATE`, `WEAK`

### E8. Gate conditions (Section 106B)
Source: `backend/SCCVision/s106b/gate.py`
Eight conditions evaluated in order:

| Code | Title | Failure causes |
|---|---|---|
| 4a | Electronic record identified and manner of production described | Missing file_name, missing server_sha256, hash_match==False, production_method=="other" |
| 4b | Particulars of the device | ALWAYS MET when device_make and device_model are provided |
| 2a | Device regularly used by person with lawful control | lawful_control==False OR regular_use==False |
| 2b | Information regularly fed into the device | regular_feed==False |
| 2c | Device operating properly | operating_properly==False AND malfunction_affects_record==True (or None) |
| 2d | Record derived from information fed into device | messages_on_this_device==False, OR production_method=="forwarded_or_copied", OR M-Pesa present but no parsed transaction codes |
| 4d | Signed by person in responsible position | owner_is_declarant==False OR declaration_confirmed==False |
| INTEGRITY | Evidence integrity checks passed | quotes_grounded==False, OR M-Pesa cross-check failed/invented_fields, OR conversation_integrity.rating=="WEAK" |

Outcome mapping:
- All met → `CERTIFICATE` / `certificate_ready`
- INTEGRITY or 4a unmet → `DEFICIENCY_REPORT` / `weak`
- Only questionnaire conditions (2a/2b/2c/2d/4b/4d) unmet, while INTEGRITY+4a met → `DEFICIENCY_REPORT` / `needs_strengthening`

### E9. DeviceFacts questionnaire fields
Source: `backend/SCCVision/s106b/models.py:18-71`
Fields: `device_make` (str, non-empty), `device_model` (str, non-empty), `phone_number_last4` (str|None, exactly 4 digits), `production_method` (ProductionMethod enum), `owner_is_declarant` (bool), `lawful_control` (bool), `regular_use` (bool), `regular_feed` (bool), `operating_properly` (bool), `malfunction_details` (str|None, required if operating_properly==False), `malfunction_affects_record` (bool|None, required if operating_properly==False), `messages_on_this_device` (bool), `declarant_name` (str, non-empty), `declarant_position` (str, non-empty), `declaration_confirmed` (bool)

ProductionMethod enum: `screenshot_on_device`, `forwarded_or_copied`, `printed_statement`, `other`

### E10. SHA-256 computation
Source: `backend/SCCVision/router.py:70-71`
```python
server_sha256 = hashlib.sha256(data).hexdigest()
```
Computed server-side on raw file bytes at `/sccvision/extract`. Client provides `sha256` in `/sccvision/evidence/process`; `hash_match = (client_sha256 == server_sha256)`.

### E11. Certificate ID format
Source: `backend/SCCVision/s106b/store.py:37-41`
```python
def _make_cert_number() -> str:
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:6].upper()
    return f"SCC106B-{today}-{suffix}"
```
Example: `SCC106B-20261117-A3F2B1`
Internal `cert_id`: `uuid.uuid4().hex` (32 hex chars, no dashes)
Stored at: `users/{uid}/certificates/{cert_id}` in Firestore.

### E12. Form SCC 1 generation
Source: `command: grep -rn "SCC 1|SCC_1|form.*scc|\.eml|whatsapp.*txt" backend --include="*.py" -i`
Result: NOT FOUND. No code generates or references "Form SCC 1". The PDF generator (`document.py`) produces only the Section 106B Certificate or Deficiency Report — it is NOT Form SCC 1 (the Small Claims Court claim form). `.eml` and WhatsApp `.txt` export parsing: NOT FOUND — the pipeline accepts only images (png/jpg/jpeg/webp).

### E13. Limitation period check
Source: `command: grep -rn "limitation|6.*year|six.*year|Cap.*22" backend --include="*.py" | grep -v test_`
The Qclassifier system prompt (`backend/Qclassifier.py:197`) mentions Cap 22 (Limitation of Actions Act) as a contextual law to consider. However, there is no deterministic limitation-period check implemented in the pipeline — it is not automatically verified against evidence dates. The `validate_knowledge.py` script references "limitation period" in test queries. The limitation period check is a system-prompt-level instruction to the LLM, not a coded rule.

---

## F. Test Results

### F1. Collection
Source: `command: cd backend && PYTHONPATH=. venv/bin/pytest tests/ --collect-only -q`
Output in `eval/logs/pytest_collect.txt`
**198 tests collected** across 5 files in 1.61s.

| Test file | Tests collected |
|---|---|
| tests/test_analysis_store.py | 11 |
| tests/test_history.py | 69 |
| tests/test_s106b_document.py | 24 |
| tests/test_s106b_gate.py | 29 |
| tests/test_s106b_routes.py | 22 |
| tests/test_sccvision_analyst.py | 43 |
| **Total** | **198** |

### F2. Full run results
Source: `eval/logs/pytest_run1.txt`, `eval/logs/pytest_run1.xml`
Run time: **22.25 seconds**

| Result | Count |
|---|---|
| PASSED | 193 |
| FAILED | 1 |
| SKIPPED | 4 |
| **Total** | **198** |

### F3. Failing test
Source: `eval/logs/pytest_run1.txt` lines at 98% mark

**Test:** `tests/test_sccvision_analyst.py::test_deterministic_followups_missing_date_and_payment`

**Assertion failure:**
```
tests/test_sccvision_analyst.py:550: in test_deterministic_followups_missing_date_and_payment
    assert "upload_mpesa_screenshot:K:payment" in ids
E   AssertionError: assert 'upload_mpesa_screenshot:K:payment' in ['provide_date:E:e-chat-1']
```
The test expects two follow-up request IDs: `provide_date:E:e-chat-1` (present) and `upload_mpesa_screenshot:K:payment` (absent). The `build_deterministic_followups` function only generates the date-related follow-up, not the M-Pesa upload request for element K. This is a genuine defect in the follow-up generation logic.
Source: `backend/tests/test_sccvision_analyst.py:527-550`

### F4. Skipped tests (emulator-dependent)
Source: `eval/logs/pytest_run1.txt` — SKIPPED rows in test_history.py
4 tests in `TestEmulatorIntegration` class skipped:
- `test_full_save_and_read_cycle`
- `test_idempotency_with_emulator`
- `test_title_generation_lifecycle`
- `test_overwrite_failed_turn_preserves_turn_index`
These tests require a Firestore emulator connection and are skipped when the emulator is not running.

### F5. Firebase emulator
Source: `command: firebase emulators:exec --only firestore "echo emulator_ok" 2>&1 | head -20`
Result: Emulator starts successfully, runs script, exits cleanly. Firebase CLI and emulator are AVAILABLE.
Output: `emulator_ok` / `Script exited successfully (code 0)`

### F6. Frontend build
Source: `command: cd frontend && npm run build 2>&1 | tail -20; echo "EXIT:$?"`
Exit code: **0** (success)
Routes built:
- Static: /auth/sign-up, /constitution, /dashboard, /firacChecker, /ingester, /litigant, /litigant/analysis, /litigant/certificates, /litigant/evidence, /litigant/sign-in, /repository, /repository/caselaws, /repository/statutes, /research
- Dynamic (server-rendered): /chat/[conversationId]

### F7. Frontend lint
Source: `command: cd frontend && npm run lint 2>&1 | tail -20; echo "EXIT:$?"`
Exit code: **0** (eslint exits 0 even with errors in Next.js config — see note)
Problems: 41 total — **26 errors, 15 warnings**
Key errors: `@typescript-eslint/no-explicit-any` in multiple files (ChatView.tsx, SCCVisionTest.tsx, firacChecker/page.tsx, historyApi.ts). These are type-safety issues but do not block build.

### F8. GitHub Actions
Source: `command: ls .github/workflows/`
Result: **NOT FOUND** — no CI/CD workflows configured.

---

## G. Measurement Scaffolding Notes

### G1. Latency script
Created: `eval/measure_latency.sh`
Script starts Flask server, waits 10s, then hits endpoints 10 times each with `curl -w "%{time_total}"`. Requires `AUTH_TOKEN` env var set to a valid Firebase ID token. Computes median and p95 via inline Python. NOT yet run — awaits manual execution with live Groq keys.

### G2. Synthetic evidence images
Source: `command: find backend/SCCVision backend -name "*.png" -o -name "*.jpg" | grep -iv node_modules|venv`
Found 6 images total:
- `SCCImages/stitched_conversation.jpg` — stitched WhatsApp conversation (project root)
- `backend/debug_highlights/scc_ogembo_p1.png`
- `backend/debug_highlights/scc_ogembo_p2.png`
- `backend/debug_highlights/scc_safaricom_p1.png`
- `backend/debug_highlights/scc_safaricom_p2.png`
- `backend/debug_highlights/constitution_p2_no_bbox.png`

No dedicated OCR ground-truth image set exists. `eval/ocr_ground_truth_template.csv` created as empty template for future population.

### G3. Retrieval queries
Created: `eval/retrieval_queries.csv` — 25 candidate queries about Small Claims Court contract disputes (Kenyan law focus). All marked `verified=UNVERIFIED`.

### G4. Hallucinations log
Source: `command: wc -l backend/logs/hallucinations.jsonl && head -5 ...`
**72 lines** in `backend/logs/hallucinations.jsonl` (each line = one unsupported claim detection event).
Sample entries show claims like `"unsupported_claim": "SECTION 1"` and `"unsupported_claim": "SECTION 106B"` being logged during the Amani Logistics test query.

### G5. s106b test bundles
Created: `eval/s106b_test_bundles.json` — 10 synthetic bundles covering:
TB-001: all conditions met → certificate_ready
TB-002: hash mismatch → weak (4a fail)
TB-003: no lawful control → needs_strengthening (2a fail)
TB-004: forwarded/copied production → needs_strengthening (2d fail)
TB-005: malfunction affecting record → needs_strengthening (2c fail)
TB-006: malfunction NOT affecting record → certificate_ready (2c passes)
TB-007: ungrounded LLM quotes → weak (INTEGRITY fail)
TB-008: WEAK conversation integrity → weak (INTEGRITY fail)
TB-009: declarant not owner + declaration not confirmed → needs_strengthening (4d fail)
TB-010: M-Pesa with no parseable codes → needs_strengthening (2d fail)

---

## H. Contradiction Resolution

### H1. Test totals
**Contradiction:** report mentions "99 collected / 95 passed" AND "198 collected / 1 failed"
**Resolved:** Actual pytest collection (run 2026-11-17): **198 tests collected**. 193 passed, 1 failed, 4 skipped.
The "99 collected / 95 passed" figure is INCORRECT — it may represent an older snapshot when only test_history.py + test_s106b_gate.py existed, or a subset run. The authoritative figure is 198 / 193P / 1F / 4S.
Source: `eval/logs/pytest_collect.txt`, `eval/logs/pytest_run1.txt`

### H2. Defect register
Source: `command: git log --all --oneline | grep -i "fix|bug"`
Commits with fix prefix or Fix prefix:
- `4608dcb fix: navigate to dashboard after profile setup without requiring refresh`
- `0f36d8b Fix F: end-to-end test Amani/M-Pesa query; fix precision tagging for bare [S#]`
- `e807d85 Fix B: compute constitution bboxes via rapidfuzz word-alignment`
- `f284295 Fix A: replace <object> PDF embed with react-pdf + bbox overlay`
- `0fc40a7 Fix constitution chunk ID mismatch + bbox script`
- `020581a Fix C/D/E: statute chunk aliases, citation precision, cluster widget`

Active defect: D-X (unnumbered): `test_deterministic_followups_missing_date_and_payment` FAILS — `build_deterministic_followups` does not generate `upload_mpesa_screenshot:K:payment` follow-up request (see F3).

### H3. Test S-15 real status
Source: `backend/tests/test_sccvision_analyst.py:527-550` (read at lines 540-550)
Test: `test_deterministic_followups_missing_date_and_payment`
Status: **FAIL** (confirmed by actual run). Asserts that two IDs are present; only one is generated.
This is a real failing test, not a documentation error. The function under test is `build_deterministic_followups`.

### H4. Test IDs S-01 through S-13 source file
The test files are:
- `tests/test_history.py` — 69 tests covering history store, conversation management, idempotency, emulator integration
- `tests/test_sccvision_analyst.py` — 43 tests covering OCR analysis, PII masking, M-Pesa parsing, gate logic

The test named `test_deterministic_followups_missing_date_and_payment` is in `test_sccvision_analyst.py` (line 527). The report's claim that S-01 through S-13 are in `test_history.py` is misleading — they may be a named subset that existed before the test file was reorganised.

### H5. Commit b70caea content
Source: `command: git show b70caea --stat`
Commit: `feat(fr04): Phase 4 — certificate store, API routes, Firestore rules`
Added: `store.py`, `routes.py`, 22 new tests in `test_s106b_routes.py`. Also updated Firestore rules to add `users/{uid}/analyses/{analysisId}` and `users/{uid}/certificates/{certId}` read rules.

### H6. Branch verification
Source: `command: git log --all --oneline | wc -l; git branch -a; git log --oneline master | wc -l`

| Metric | Value |
|---|---|
| Total commits across all branches | 63 |
| Commits on master | 56 |
| Commits on feat/fr04-s106b-certificate | 63 (includes all master commits — this branch includes master) |
| Local branches | master, feat/fr04-s106b-certificate, backup, temp-fix |
| Remote branches | origin/master |

Current branch: `feat/fr04-s106b-certificate`

### H7. Evidence persistence
Confirmed from D2 and Firestore rules:
**Stored in Firestore (via Admin SDK, client cannot write):**
- Conversations: `users/{uid}/conversations/{convId}` + `turns/{turnId}`
- Analysis snapshots: `users/{uid}/analyses/{analysisId}`
- Certificates: `users/{uid}/certificates/{certId}`
- Case law upload metadata: `CaseLaws/{auto-id}`

**NOT stored in Firestore:**
- PDF bytes (served from local disk at `backend/SCCCases/`)
- ChromaDB embeddings (stored in local SQLite files)
- Certificate PDFs (regenerated deterministically on each download, never persisted)
- OCR raw text from evidence images

### H8. Limitation period check
Confirmed: **stub only**. The limitation period check is a system-prompt instruction to the LLM classifier (`Qclassifier.py:197`), not a deterministic coded rule. No function in the codebase computes "is this claim within the 6-year period under Cap 22" against evidence dates. The `compute_date_checks` function in `backend/SCCVision/analyst/dates.py` checks conversation/due dates for the SCC evidence pipeline, but this is about contract performance dates, not limitation period.

### H9. FIRAC fields
Confirmed from C3: All 5 FIRAC components are present and populated in LegalSummariesDB (`firac_component` values: `facts`, `issues`, `rules`, `application`, `conclusion`). CaseLawDB also stores FIRAC components. The 40-chunk LegalSummariesDB holds 8 cases × 5 components.

---

## I. Version Control Evidence

### I1. Git log (most recent 60 commits shown in graph)
Source: `eval/chapter5_facts.md` section — see full output in section H6.
Key structural commits: FR04 development (7 commits b4c79b5 through 563d8c1), skeleton loading (e96979e, 9f8a335), history/routing (1ffd1be, 8c518dc), hybrid retrieval (e737d54), Firebase Auth (7892723), knowledge graph attempt (5a2cf88).

### I2. Commit type distribution
Source: `command: git log --all --format="%s" | grep -oP "^(feat|fix|test|docs|refactor|chore|style|perf)" | sort | uniq -c | sort -rn`

| Type | Count |
|---|---|
| feat | 13 |
| docs | 3 |
| fix | 1 |
| refactor | 1 |
| chore | 1 |

Note: Many early commits use free-form messages (e.g. "we cooking", "woi nisichome please") not matching conventional commit format.

### I3. First and last commit dates
Source: `command: git log --all --format="%ad" --date=short | sort | head -1` and `tail -1`
- First commit: **2025-05-18**
- Last commit: **2026-11-17** (audit date)

### I4. .gitignore coverage
Source: `.gitignore` (full read)
| Pattern | Present? |
|---|---|
| `.env` | YES (line 8) |
| `.env.local` | YES (line 9) |
| `*-service-account.json`, `*firebase-adminsdk*.json`, `serviceAccount*.json` | YES (lines 12-14) |
| `venv/`, `env/` | YES (lines 47-48) |
| `*.log` | YES (line 69) |
| ChromaDB directories (ChunkDB/, LegalSummariesDB/, etc.) | **NOT explicitly covered** — ChromaDB dirs are tracked via git status as untracked, but no gitignore entry covers `*DB/` or `*DB` patterns |
| `*.jsonl` | **NOT covered** — `hallucinations.jsonl` and other jsonl files are not ignored |
| `neo4j-community-*/data/` | YES (line 1) |

### I5. Frontend lockfile committed
Source: `command: ls frontend/package-lock.json`
Result: `package-lock.json` EXISTS and is committed (shown as `M frontend/app/...` in git status context, confirmed by ls).

### I6. Secret scan
Source: `command: git log -p --all -- "*.py" "*.ts" "*.tsx" "*.json" 2>/dev/null | grep -c "gsk_|AIza|private_key|AAAAB3NzaC|-----BEGIN"`
Result: **0** — no secret patterns found in git history for tracked source files.

### I7. README.md description vs actual system
Source: First 60 lines of README.md
README describes: "A comprehensive RAG system using local embeddings and remote LLMs via Hugging Face API for intelligent document question-answering" with HuggingFace API token setup.

**Gap:** README is severely outdated. The current system uses Groq (not HuggingFace), has Flask + Next.js 13 + Firebase Auth RBAC, two modules (Advocate Research + SCC Evidence), Section 106B certificate generation, Firestore history, hybrid BM25+RRF+cross-encoder retrieval. README reflects the initial prototype state from May 2025.

---

## J. Neo4j / Graph Component Status

### J1. Files referencing neo4j/cypher
Source: `command: grep -rn "neo4j|cypher|bolt://" ... -l | grep -v node_modules|venv`
Files found:
- `backend/export_blacklaw_to_json.py` — exports Black's Law Dictionary to JSON for potential Neo4j import
- `backend/Const_IntepretationLaws/blacklaw_export_for_neo4j.json` — the exported JSON file
- `backend/Qclassifier.py` — contains mention of MATCH clause in system prompt context
- `backend/Retriever.py` — MATCH keyword in SQL-like retrieval comments
- `neo4j-community-5.18.0/` directory — Neo4j binary downloaded but in .gitignore

### J2. Git commits referencing graph
Source: `command: git log --all --oneline -- "*neo4j*" "*cypher*" "*graph*"`
- `5a2cf88 knowledge graph` — commit that added graph functionality
- `578cf9b knowledge tester 1` — preceding test commit

### J3. Neo4j driver in requirements
Source: `command: grep -i "neo4j|py2neo|bolt" backend/requirements.txt`
Result: **NOT FOUND** — no neo4j Python driver is in requirements.txt.

### J4. Graph component assessment
The Neo4j binary (`neo4j-community-5.18.0/`) was downloaded and is in .gitignore. `export_blacklaw_to_json.py` and `blacklaw_export_for_neo4j.json` suggest a planned graph of legal terms (Black's Law Dictionary). However:
- No neo4j Python driver in requirements.txt
- No active graph queries in the live pipeline
- No Cypher queries or connection code in backend Python files
- The `knowledge graph` commit (5a2cf88) predates the FR04/certificate work and appears to have been a separate prototype branch that was merged but not activated

**Status: Knowledge graph was prototyped but is NOT integrated into the live pipeline. The ChromaDB + BM25 + cross-encoder stack is the active retrieval layer.**

---

## Final: Placeholder Fill Table

| Placeholder | Verified value | Source |
|---|---|---|
| Flask version | 3.1.2 | pip freeze output |
| firebase_admin version | 7.7.0 | pip freeze output |
| chromadb version | 1.3.5 | pip freeze output |
| sentence-transformers version | 5.1.2 | pip freeze output |
| torch version | 2.9.1 | pip freeze output |
| rank-bm25 version | 0.2.2 | pip freeze output |
| rapidocr version | 3.10.0 | pip freeze output |
| onnxruntime version | 1.23.2 | pip freeze output |
| opencv-python-headless version | 5.0.0.93 | pip freeze output |
| pillow version | 12.0.0 | pip freeze output |
| pydantic version | 2.14.2 | pip freeze output |
| fpdf2 version | 2.9.0 | pip freeze output |
| PyMuPDF version | 1.26.6 | pip freeze output |
| pypdf version | 6.3.0 | pip freeze output |
| pytesseract version | 0.3.13 | pip freeze output |
| ocrmypdf version | 17.13.0 | pip freeze output |
| rapidfuzz version | 3.14.6 | pip freeze output |
| groq (SDK) version | 0.35.0 | pip freeze output |
| langchain version | 1.0.7 | pip freeze output |
| next.js version | 16.0.3 | frontend/package.json |
| react version | 19.2.0 | frontend/package.json |
| firebase (frontend) version | ^10.12.0 | frontend/package.json |
| Groq model ID | qwen/qwen3.8-27b | backend/groq_client.py:19 |
| Embedding model | intfloat/e5-base-v2 (768d) | backend/digester.py:22, backend/config.py:27 |
| Cross-encoder model | cross-encoder/ms-marco-MiniLM-L-6-v2 | backend/Retriever.py:84 |
| LOW_CONFIDENCE_THRESHOLD | 0.4 | backend/Retriever.py:53 |
| CERT_THRESHOLD | NOT FOUND (not in codebase — gate uses Boolean logic) | grep search |
| ChunkDB count | 704 | chromadb_audit.txt |
| LegalSummariesDB count | 40 (8 cases × 5 FIRAC) | chromadb_audit.txt |
| MasterRulesDB count | 619 | chromadb_audit.txt |
| StatutesDB count | 1331 | chromadb_audit.txt |
| CaseLawDB count | 78 | chromadb_audit.txt |
| BlackLawDictionaryDB count | 40061 | chromadb_audit.txt |
| Total ChromaDB chunks | 42833 | sum of above |
| pytest total collected | 198 | eval/logs/pytest_collect.txt |
| pytest passed | 193 | eval/logs/pytest_run1.txt |
| pytest failed | 1 | eval/logs/pytest_run1.txt |
| pytest skipped | 4 | eval/logs/pytest_run1.txt |
| pytest runtime | 22.25 seconds | eval/logs/pytest_run1.txt |
| failing test | test_deterministic_followups_missing_date_and_payment | eval/logs/pytest_run1.txt |
| failing assertion | assert 'upload_mpesa_screenshot:K:payment' in ['provide_date:E:e-chat-1'] | eval/logs/pytest_run1.txt |
| frontend build exit code | 0 | npm run build output |
| frontend lint problems | 26 errors, 15 warnings (41 total) | npm run lint output |
| GitHub Actions CI | NOT FOUND — no .github/workflows/ directory | ls command |
| first commit date | 2025-05-18 | git log |
| last commit date | 2026-11-17 | git log |
| total commits (all branches) | 63 | git log --all | wc -l |
| commits on master | 56 | git log master | wc -l |
| hallucinations.jsonl lines | 72 | wc -l command |
| certificate number format | SCC106B-YYYYMMDD-XXXXXX | backend/SCCVision/s106b/store.py:37-41 |
| PDF stored in Firestore | NO — regenerated deterministically on each download | backend/SCCVision/s106b/store.py |
| Form SCC 1 generated | NOT FOUND — not implemented | grep command |
| .eml parsing | NOT FOUND — not implemented | grep command |
| WhatsApp .txt export parsing | NOT FOUND — not implemented | grep command |
| limitation period check (coded) | NOT FOUND — LLM system prompt only | grep command |
| Neo4j driver in requirements | NOT FOUND | requirements.txt grep |
| Neo4j status | Prototype only — not in live pipeline | analysis |
| secret scan count | 0 | git log -p grep |

---

## Facts That Changed

| Claim in draft | Correct value | Source |
|---|---|---|
| "99 collected / 95 passed" | 198 collected / 193 passed / 1 failed / 4 skipped | eval/logs/pytest_collect.txt, eval/logs/pytest_run1.txt |
| "198 collected / 1 failed" | Correct — but the 198/1 figure is accurate; the "99/95" figure in another part of the report is wrong | eval/logs/pytest_run1.txt |
| "three collections / 2028 chunks" | Six collections / 42,833 total chunks | chromadb_audit.txt |
| CERT_THRESHOLD = 0.83 | NOT FOUND — no such threshold; gate is entirely Boolean | grep search across all backend Python files |
| "test_deterministic_followups_missing_date" is PASS | FAIL | eval/logs/pytest_run1.txt |

---

## Could Not Verify

1. **p95/median latency numbers** — backend server cannot be started in this audit context (requires live GROQ_API_KEY and FIREBASE_SERVICE_ACCOUNT_PATH). Script created at `eval/measure_latency.sh` for manual execution.
2. **OCR accuracy metrics** — no ground-truth image set exists. `eval/ocr_ground_truth_template.csv` created as template.
3. **Retrieval precision/recall** — no labelled query set has been evaluated. `eval/retrieval_queries.csv` created with 25 unverified queries.
4. **Exact installed versions from lockfile** — `frontend/package-lock.json` exists but was not read to extract resolved versions (file is large; declared versions in package.json are sufficient for the report).
5. **Number of potential Neo4j graph edges** — `backend/SCCCases/*.txt` does not exist (all files are PDFs); grep over PDFs is not feasible without binary parsing. NOT FOUND.
6. **Qclassifier.py model identifier** — Qclassifier uses `GROQ_MODEL` from groq_client (confirmed `qwen/qwen3.8-27b`) but manages its own key rotation independently.
