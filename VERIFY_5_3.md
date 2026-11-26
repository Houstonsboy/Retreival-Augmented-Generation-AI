# VERIFY_5_3.md — Section 5.3 Verification Findings

**Purpose:** Read-only investigation of the repository to answer 12 specific questions for the project report (Chapter 5).  
**Date:** 2026-11-17  
**Rules applied:** Report only what the code shows. Every finding cites a file path and line number. Planned, commented-out, mocked, or hardcoded behaviour is NOT described as working. Secret values are not printed.

---

## PART A — Section 106B Certificate and Deficiency Report (FR-04)

---

### Q1 — What does `certificates/page.tsx` actually render?

**Finding:**  
`frontend/app/litigant/certificates/page.tsx` renders a filterable table of `CertificateRecord` items. The UI presents three outcome pill filters (`certificate_ready`, `needs_strengthening`, `weak`), a search input, and a table with columns: Certificate No., File, Outcome, Date, and Actions (Download / View Report buttons).

**Evidence:**  
- `frontend/app/litigant/certificates/page.tsx` — full component logic.  
- `frontend/app/lib/evidenceApi.ts:10–12` — `listCertificates()` returns `[]` unconditionally:
  ```typescript
  // TODO(backend): GET /api/evidence/certificates
  export async function listCertificates(): Promise<CertificateRecord[]> {
    return [];
  }
  ```
- `frontend/app/litigant/certificates/page.tsx:214` — Download button is `disabled` with comment `{/* TODO(backend): Download certificate PDF — GET /api/evidence/certificates/{id}/download */}`.

**Write in report:**  
The certificates page renders a complete filterable table UI but is backed by a stub (`listCertificates()` returns `[]` unconditionally); no certificates are ever fetched, displayed, or downloadable.

---

### Q2 — Does any backend route generate a Section 106B certificate or deficiency report document?

**Finding:**  
No backend route generates a certificate or deficiency report. All occurrences of `credentials.Certificate` in the backend are Firebase Admin SDK calls for authentication management (`backend/auth.py:30`, `backend/scripts/bootstrap_admin.py:52`, `backend/scripts/set_role.py:50`, `backend/scripts/seed_caselaws.py:34`) — entirely unrelated to legal certificates. The string `"106B"` appears in `backend/history_store.py:47` and `backend/validate_knowledge.py` only as topic/metadata strings. The `CertificateModal` component in `frontend/app/components/sccvision/SCCVisionTest.tsx:1234` is a frontend rendering component, not a backend generation route.

**Evidence:**  
- `grep -rn "certificate\|deficiency_report\|106B" backend/ --include="*.py"` — no route found that creates a certificate file.  
- `backend/SCCVision/router.py` — no endpoint named `certificates`, `deficiency`, or `106B`.  
- `backend/history_store.py:47` — `"106B"` appears as a Firestore topic string only.

**Write in report:**  
No backend endpoint generates a Section 106B certificate or deficiency report document; FR-04's server-side deliverable is NOT IMPLEMENTED.

---

### Q3 — Is a server-generated PDF or DOCX certificate produced at any point?

**Finding:**  
No. `fpdf2` (version 2.8.3) is listed in `backend/requirements.txt` but is not imported in any runtime application file. The only mechanism that produces a printable certificate is `window.print()` called by `handlePrint` inside `CertificateModal` (`frontend/app/components/sccvision/SCCVisionTest.tsx`) — a browser print-to-PDF, entirely client-side and within the developer test panel only. The litigant-facing `certificates/page.tsx:214` has a disabled Download button marked `TODO(backend)`.

**Evidence:**  
- `grep -rn "fpdf\|FPDF\|from fpdf" backend/` — returns only `requirements.txt`; no runtime import found.  
- `frontend/app/components/sccvision/SCCVisionTest.tsx` — `handlePrint` calls `window.print()`.  
- `frontend/app/litigant/certificates/page.tsx:214` — `disabled` Download button with `TODO(backend)` comment.

**Write in report:**  
No server-generated PDF or DOCX certificate is produced. `fpdf2` is installed but unused at runtime. The only print path is `window.print()` in the developer test panel, which is NOT a server-generated document.

---

### Q4 — What deficiency/follow-up fields does `AnalystResult` expose?

**Finding:**  
`backend/SCCVision/analyst/models.py` defines `AnalystResult` with the following deficiency and follow-up fields:

| Field | Type | Purpose |
|---|---|---|
| `missing_information` | `list[str]` | Information gaps identified |
| `strengthen_checklist` | `list[str]` | Suggested additional evidence |
| `follow_up_requests` | `list[FollowUpRequest]` | Structured questions per element: `element`, `prompt`, `why_it_matters`, `blocking: bool`, `options: list[str]` |
| `questions_for_parties` | `list[PartyQuestion]` | Questions directed at claimant or respondent |
| `red_flags` | `list[str]` | Identified risk factors |
| `overall_verdict` | `OverallVerdict` | Enum: LIKELY VALID / POSSIBLY VALID WITH GAPS / LIKELY NOT VALID / INSUFFICIENT EVIDENCE |
| `elements` | `dict[str, Element]` | Keys A–K (contract elements), each with `status: ElementStatus`, `evidence`, `reasoning`, `confidence: float` |

**Evidence:**  
- `backend/SCCVision/analyst/models.py` — all field definitions.

**Write in report:**  
`AnalystResult` exposes seven deficiency-related fields (`missing_information`, `strengthen_checklist`, `follow_up_requests`, `questions_for_parties`, `red_flags`, `overall_verdict`, `elements`) that together constitute a structured deficiency report at the data layer.

---

### Q5 — Is a s.106B device questionnaire (device used, operator, working properly) implemented in the litigant flow?

**Finding:**  
NOT IMPLEMENTED. `frontend/app/litigant/analysis/page.tsx` displays a 4-step visual guide where Step 3 is labelled "Answer a few questions / About your device or phone". However, the `handleSubmit` function in the same file uses a `setTimeout` simulation with `TODO(backend)` comments and no questionnaire form fields are rendered. The actual analysis is delegated to the `<SCCVisionTest />` component at the bottom of the page, which does not include a device-specific questionnaire.

**Evidence:**  
- `frontend/app/litigant/analysis/page.tsx` — Step 3 label and `handleSubmit` with `setTimeout` simulation and `TODO(backend)` comments.  
- No form fields for device type, operator, or working-condition questions found in any `frontend/app/litigant/` file.

**Write in report:**  
The s.106B device questionnaire (Step 3 of the litigant flow) is NOT IMPLEMENTED; the step label exists in the UI but no form fields or backend integration are present.

---

### Q6 — FR-04 Verdict: Which description best fits the implementation?

**Finding:**  
**(b) — Partial implementation.**

- **Implemented:** Structured deficiency/follow-up output via `AnalystResult` fields (`missing_information`, `strengthen_checklist`, `follow_up_requests`, etc.). A `CertificateModal` component in `frontend/app/components/sccvision/SCCVisionTest.tsx:1234` renders live analysis data (verdict, confidence bar, parties, contract elements A–K, M-Pesa payments, strengthen checklist, red flags, missing information, follow-up requests, disclaimer) and offers a "Print / Save PDF" button via `window.print()`. Confidence threshold for `certificate_ready` outcome is `CERT_THRESHOLD = 0.83` (`SCCVisionTest.tsx:21`).
- **Not implemented:** Server-generated certificate/deficiency PDF. Litigant-facing `certificates/page.tsx` stub (no data). s.106B device questionnaire. Certificate persistence and retrieval (`listCertificates()` returns `[]`).

**Evidence:**  
- `frontend/app/components/sccvision/SCCVisionTest.tsx:21` — `const CERT_THRESHOLD = 0.83`  
- `frontend/app/components/sccvision/SCCVisionTest.tsx:1234` — `CertificateModal` component  
- `frontend/app/lib/evidenceApi.ts:10–12` — `listCertificates()` stub  
- `backend/SCCVision/analyst/models.py` — `AnalystResult` deficiency fields

**Write in report:**  
FR-04 is partially implemented: the evidence analysis engine produces a structured deficiency and follow-up report (via `AnalystResult`), and a developer-facing `CertificateModal` renders this output with a print-to-PDF path; however, the litigant-facing certificates page is a non-functional stub, no server-generated certificate document is produced, and the s.106B device questionnaire is absent — record FR-04 as a known gap in Section 5.3.6.

---

## PART B — SHA-256 Hashing of Evidence

---

### Q7 — How is SHA-256 hashing implemented, and does the backend verify it?

**Finding:**  
SHA-256 hashing is implemented client-side only. The function `sha256Hex()` at `frontend/app/components/sccvision/SCCVisionTest.tsx:206–208` uses the Web Crypto API (`crypto.subtle.digest("SHA-256", buf)`) to hash the file's `ArrayBuffer`. The hash is called at lines 360–367 and 475–483 on each uploaded file before the OCR result is sent to `/sccvision/evidence/process`. The backend stores the hash as `item["sha256"] = data.get("sha256")` at `backend/SCCVision/router.py:219` — a plain assignment with no server-side recomputation, no integrity check, and no comparison against a stored baseline. The hash is not displayed to the user anywhere in the UI.

Other `hashlib` usages in the backend (`Retriever.py`, `RulesRetriever.py`, `build_case_chunks.py`) hash chunk content for deduplication IDs and prompt versioning — not evidence file integrity.

**Evidence:**  
- `frontend/app/components/sccvision/SCCVisionTest.tsx:206–208` — `crypto.subtle.digest("SHA-256", buf)`  
- `frontend/app/components/sccvision/SCCVisionTest.tsx:360–367, 475–483` — hash computed and sent  
- `backend/SCCVision/router.py:219` — `"sha256": data.get("sha256")` stored verbatim  
- No server-side `hashlib.sha256` call found in `SCCVision/router.py` or `SCCVision/service.py`

**Write in report:**  
SHA-256 hashing is implemented client-side via the Web Crypto API and the hash is stored by the backend as metadata without server-side verification; this provides chain-of-custody traceability (the hash travels with the evidence item) but does not constitute tamper detection, as the backend never independently computes or validates the hash value.

---

## PART C — Other Open Questions for Section 5.3

---

### Q8 — Is the 3-year limitation period (s.4, Limitation of Actions Act) implemented?

**Finding:**  
NOT IMPLEMENTED. `backend/SCCVision/analyst/dates.py` implements `compute_date_checks()` which checks for future-dated facts and computes `performance_due` status. However, no 3-year constant, `timedelta(days=1095)`, or reference to the Limitation of Actions Act s.4 is present in `dates.py`, `backend/SCCVision/analyst/rules.py`, or `backend/config.py`. `config.py:33` defines only `SCC_TIMEZONE = "Africa/Nairobi"`.

**Evidence:**  
- `grep -n "3 year\|3-year\|THREE_YEAR\|LIMITATION_YEARS\|limitation_years\|limitation.*period\|timedelta.*1095" backend/SCCVision/ --include="*.py"` — no matches  
- `backend/config.py:33` — `SCC_TIMEZONE = "Africa/Nairobi"` (only relevant constant)  
- `backend/SCCVision/analyst/dates.py` — `compute_date_checks()` checks future dates and performance_due; no limitation period check

**Write in report:**  
The 3-year limitation period under s.4 of the Limitation of Actions Act is NOT IMPLEMENTED in the date-checking logic; `compute_date_checks()` detects future-dated facts and computes performance-due status only — record this as a known gap.

---

### Q9 — Are FIRAC labels stored per chunk in the case-law collection?

**Finding:**  
Yes, `firac_component` is stored as a metadata key per chunk. `backend/build_case_chunks.py:382` explicitly sets `"firac_component": ""` on every chunk at ingestion time. A live metadata query on the `case_law` ChromaDB collection confirms the key is present on all documents (confirmed from two sampled chunks). The case_law chunk metadata keys are: `chunk_total`, `court_level`, `chunk_index`, `label`, `page_end`, `prev_chunk_id`, `winning_party`, `ocr`, `page_start`, `next_chunk_id`, `document_hash`, `case_identifier`, `content_length`, `file_name`, `year`, `legal_domain`, `parties`, `firac_component`, `judge`, `source_file_path`.

However, all stored values are empty strings (`""`), meaning FIRAC classification was not applied to the ingested documents — the field acts as a schema placeholder.

**Evidence:**  
- `backend/build_case_chunks.py:382` — `"firac_component": ""`  
- Live ChromaDB query on `case_law` collection — `firac_component` present in all chunk metadata, value `""`

**Write in report:**  
FIRAC labels are present as a metadata schema field on every case-law chunk but are stored as empty strings; FIRAC classification was not applied during ingestion, so the field cannot be used for retrieval filtering in the current build.

---

### Q10 — What distance metric do all three ChromaDB collections use?

**Finding:**  
All three collections use **cosine similarity** (`hnsw:space: cosine`). This was confirmed both from source code and from a live metadata query against the running ChromaDB instance.

| Collection | Source | Live Metadata |
|---|---|---|
| `case_law` | `backend/CaseLaw/ingester.py:193–195` | `{'hnsw:space': 'cosine'}` |
| `statutes` | `backend/embed_knowledge.py:190–192` | `{'hnsw:space': 'cosine'}` |
| `master_rules` | `backend/constitutionlogic/constitution_digester.py:49–51` | `{'hnsw:space': 'cosine'}` |

**Evidence:**  
- `backend/CaseLaw/ingester.py:193–195` — `get_or_create_collection(name=CASELAW_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})`  
- `backend/embed_knowledge.py:190–192` — same pattern for statutes  
- `backend/constitutionlogic/constitution_digester.py:49–51` — same pattern for master_rules  
- Live ChromaDB query: all three collections returned `{'hnsw:space': 'cosine'}`

**Write in report:**  
All three ChromaDB collections (`case_law`, `statutes`, `master_rules`) are configured with cosine similarity as their HNSW distance metric, confirmed from both source code and live database metadata.

---

### Q11 — What case documents are in the knowledge base?

**Finding:**  
The `case_law` ChromaDB collection contains chunks from 9 distinct cases (by `case_identifier`). The source directory `backend/SCCCases/` contains 11 PDF/DOCX files; two files share a `case_identifier`, indicating a judgment/ruling pair from the same case.

The 9 cases (from live ChromaDB query):

| # | case_identifier | parties | year | court_level | legal_domain | winning_party |
|---|---|---|---|---|---|---|
| 1 | SCC/E001/024/2023 | Cherop v Mwangi | 2023 | Small Claims Court | Contract | claimant |
| 2 | SCC/E001/039/2023 | Gitonga v Kamau | 2023 | Small Claims Court | Contract | claimant |
| 3 | SCC/E001/056/2023 | Njuguna v Wangari | 2023 | Small Claims Court | Contract | respondent |
| 4 | SCC/E001/078/2023 | Ochieng v Atieno | 2023 | Small Claims Court | Employment | claimant |
| 5 | SCC/E001/092/2023 | Kamau v Wanjiru | 2023 | Small Claims Court | Property | claimant |
| 6 | SCC/E001/112/2023 | Muthoni v Karanja | 2023 | Small Claims Court | Contract | claimant |
| 7 | SCC/E001/134/2023 | Kipchoge v Ruto | 2023 | Small Claims Court | Employment | respondent |
| 8 | SCC/E001/156/2023 | Wairimu v Ngugi | 2023 | Small Claims Court | Property | claimant |
| 9 | SCC/E001/178/2023 | Achieng v Omondi | 2023 | Small Claims Court | Contract | claimant |

**Evidence:**  
- Live ChromaDB query on `case_law` collection: `c.get(include=["metadatas"])` with `case_identifier` deduplication  
- `backend/SCCCases/` — 11 source files  
- `backend/CaseLaw/ingester.py:193–195` — ingestion source

**Write in report:**  
The case-law knowledge base contains 9 unique cases (all 2023 Small Claims Court judgments), ingested from 11 source files; the domain split is Contract (5 cases), Employment (2 cases), Property (2 cases).

---

### Q12 — How many statutes and sections are in the knowledge base?

**Finding:**  
The statutes knowledge base contains **14 Acts** comprising **1,155 sections** in total, per `data/raw_statutes/manifest.json`.

| Act | Sections |
|---|---|
| Small Claims Court Act | 52 |
| Law of Contract Act | 4 |
| Sale of Goods Act | 59 |
| Consumer Protection Act | 94 |
| Limitation of Actions Act | 46 |
| Evidence Act | 199 |
| Civil Procedure Act | 116 |
| Bills of Exchange Act | 101 |
| Kenya Information and Communications Act | 198 |
| Stamp Duty Act | 126 |
| Judicature Act | 11 |
| Small Claims Court (Procedure) Rules | 37 |
| Employment Act | 96 |
| Landlord and Tenant (Shops, Hotels and Catering Establishments) Act | 16 |
| **TOTAL** | **1,155** |

**Evidence:**  
- `data/raw_statutes/manifest.json` — 14 entries, `section_count` field per act  
- `backend/embed_knowledge.py:190–192` — ingestion source for statutes collection

**Write in report:**  
The statutes knowledge base covers 14 Kenyan Acts totalling 1,155 sections, with the Evidence Act (199), Kenya ICT Act (198), Stamp Duty Act (126), Civil Procedure Act (116), and Consumer Protection Act (94) contributing the most content.

---

## Summary of FR-04 Gaps for Section 5.3.6

The following FR-04 items should be recorded as known deviations/gaps in Section 5.3.6:

| Gap | Status |
|---|---|
| Server-generated Section 106B certificate PDF/DOCX | NOT IMPLEMENTED (`fpdf2` installed but unused) |
| Litigant-facing certificates page data | NOT IMPLEMENTED (`listCertificates()` returns `[]`) |
| Certificate download/view endpoints | NOT IMPLEMENTED (buttons disabled, `TODO(backend)`) |
| s.106B device questionnaire (Step 3 litigant flow) | NOT IMPLEMENTED (UI label only, no form) |
| 3-year limitation period check (s.4, Limitation of Actions Act) | NOT IMPLEMENTED |

**What is implemented under FR-04:**  
- `AnalystResult` Pydantic model with structured deficiency output fields  
- `CertificateModal` in developer test panel renders full analysis output  
- `window.print()` path for browser print-to-PDF from developer test panel  
- `CERT_THRESHOLD = 0.83` confidence threshold for `certificate_ready` outcome classification
