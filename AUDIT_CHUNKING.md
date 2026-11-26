# Chunking & Ingestion Audit — Kenyan Legal RAG

**Date:** 2026-10-31  
**Scope:** Read-only audit of document ingestion, chunking, embedding and vector-store write paths.  
**Auditor:** Claude Code  

---

## 1. Pipeline Map

### 1.1 File roles and call order

| # | File | Role |
|---|------|------|
| 1a | `backend/build_case_chunks.py` | PDF/DOCX → paragraph-chunked JSONL for case law |
| 1b | `backend/build_statute_chunks.py` | Pre-parsed statute sections → token-windowed JSONL |
| 1c | `backend/constitutionlogic/constitutionbreaker.py` | Constitution PDF → character-windowed JSONL with OCR fallback |
| 1d | `backend/digester.py` | Generic multi-format ingestion (legacy); LangChain splitter |
| 2a | `backend/embed_knowledge.py` | JSONL → e5-base-v2 embeddings → ChromaDB upsert (cases & statutes) |
| 2b | `backend/CaseLaw/ingester.py` | FIRAC-component ingestion; inline embed + chromadb add |
| 3a | `backend/Retriever.py` | Hybrid retrieval (vector + BM25 + cross-encoder) for case law |
| 3b | `backend/RulesRetriever.py` | Hybrid retrieval for statutes; constitution fallback |
| 4 | `backend/app.py` | Flask API: synthesis, PDF serving, upload |

### 1.2 Data-flow diagram

```
PDF / DOCX files
      │
      ├─ build_case_chunks.py  ──────────────────────────────┐
      │     pypdf.PdfReader / python-docx                    │
      │     paragraph-split (3 000 char, 1-para overlap)     │
      │     → output: JSONL files                            │
      │                                                       │
      ├─ build_statute_chunks.py                             │
      │     pre-parsed sections.jsonl                        │
      │     e5-base-v2 tokenizer sliding window (480 tok)    │
      │     → output: JSONL files                            │
      │                                                       │
      ├─ constitutionbreaker.py                              │
      │     fitz.open / pytesseract OCR fallback             │
      │     character window (1 000 char, 300 overlap)       │
      │     → output: JSONL files                            │
      │                                                       │
      └─ digester.py (legacy)                                │
            pypdf / LangChain RecursiveCharTextSplitter      │
            (1 000 char, 300 overlap)                        │
            → output: inline ChromaDB write (ChunkDB)        │
                                                             │
                    embed_knowledge.py                        │
                    SentenceTransformer e5-base-v2 <─────────┘
                          │
                    ChromaDB PersistentClient
                          │
              ┌───────────┼───────────┬──────────────┐
              ▼           ▼           ▼              ▼
         case_law     statutes   master_rules   ChunkDB
       (CaseLawDB)  (StatutesDB) (MasterRulesDB) (ChunkDB)
                                                  LegalSummariesDB
                                                  (FIRAC, via ingester.py)
```

---

## 2. Libraries in Use

| Concern | Library | Version | Source |
|---------|---------|---------|--------|
| PDF extraction (cases, generic) | `pypdf` | 6.3.0 | requirements.txt:12 |
| PDF extraction (constitution) | `PyMuPDF` (fitz) | *not pinned* | constitutionbreaker.py:1 ⚠️ |
| OCR fallback (constitution) | `pytesseract` | *not pinned* | constitutionbreaker.py:8 |
| DOCX extraction | `python-docx` | 1.2.0 | requirements.txt:13 |
| Chunker — cases | Custom paragraph splitter | — | build_case_chunks.py:66–91 |
| Chunker — statutes | Token sliding window (e5-base-v2 tokenizer) | — | build_statute_chunks.py:48–72 |
| Chunker — constitution | Custom character window | — | constitutionbreaker.py:257–315 |
| Chunker — generic | `langchain` `RecursiveCharacterTextSplitter` | 1.0.7 | digester.py:10,214 |
| Embedding model | `sentence-transformers` `intfloat/e5-base-v2` | 5.1.2 | embed_knowledge.py, config.py |
| Vector store | `chromadb` PersistentClient | 1.3.5 | requirements.txt:23 |
| BM25 | `rank-bm25` | 0.2.2 | requirements.txt:16 |
| Cross-encoder reranker | `sentence-transformers` CrossEncoder | 5.1.2 | Retriever.py |

**⚠️ PyMuPDF (fitz) and pytesseract are imported in `constitutionbreaker.py` but absent from `requirements.txt`.** A fresh `pip install -r requirements.txt` will fail when that script is run.

---

## 3. Chunk Schema

### 3.1 Collections summary

| Database path | Collection | Count |
|---------------|------------|-------|
| `backend/CaseLawDB` | `case_law` | 59 |
| `backend/LegalSummariesDB` | `legal_summaries` | 40 |
| `backend/StatutesDB` | `statutes` | 1 331 |
| `backend/constitutionlogic/MasterRulesDB` | `master_rules` | 619 |
| `backend/ChunkDB` | `ChunkDB` | 704 |

### 3.2 Schema field audit

#### Case law (`case_law` / `legal_summaries`)

| Field | Status | Notes |
|-------|--------|-------|
| `chunk_id` (stable) | **PRESENT** | `{case_id}_{index}` e.g. `2026SCC242KLR_0`; built in `build_case_chunks.py:94–97` |
| `prev_chunk_id` / `next_chunk_id` | **PRESENT** | Linked-list chain post-processed at `build_case_chunks.py:186–195` |
| `source_file_path` | **PRESENT** | Absolute path to original file (`Path.resolve()`); `build_case_chunks.py:170` |
| `document_hash` | **PRESENT** | SHA-256 of full extracted text, first 16 hex chars; `build_case_chunks.py:156` |
| `case_identifier` | **PRESENT** | KLR citation string e.g. `2026SCC242(KLR)` |
| `file_name`, `parties`, `court_level`, `judge`, `year` | **PRESENT** | From pre-filled metadata JSON |
| `legal_domain`, `winning_party` | **PRESENT** | From metadata JSON (must be pre-filled manually) |
| `firac_component` | **PRESENT** | `facts`/`issues`/`rules`/`application`/`conclusion` for `legal_summaries`; empty string for raw `case_law` chunks |
| `chunk_index`, `chunk_total` | **PRESENT** | Sequential 0-based index; `build_case_chunks.py:167,169` |
| `content_length` | **PRESENT** | `len(chunk_text)` |
| **page number (start/end)** | **MISSING** | pypdf iterates pages (`build_case_chunks.py:48`) but page number is never written to metadata |
| **bounding boxes** | **MISSING** | `pypdf.page.extract_text()` returns plain string; coordinates never extracted |
| **character offsets** | **MISSING** | Not tracked at any stage |
| **structural label** | **PARTIAL** | `firac_component` labels component type; no paragraph/section number within judgment |
| `ingested_at` | **PRESENT** (legal_summaries only) | ISO timestamp; absent from raw case_law chunks |
| `embedding_model` | **PRESENT** (legal_summaries only) | `intfloat/e5-base-v2` |

#### Statutes (`statutes`)

| Field | Status | Notes |
|-------|--------|-------|
| `chunk_id` (stable) | **PRESENT** | `{act_slug}__{version}__s{section}_{idx}` e.g. `small_claims_court_act__2022-12-31__s1__0` |
| `source_document` / `act_title` | **PRESENT** | Human-readable act title |
| `frbr_work_id` | **PRESENT** | FRBR URI e.g. `ke/act/2016/2` — globally unique |
| `act_version` | **PRESENT** | Date string e.g. `2022-12-31` |
| `section_num`, `section_heading` | **PRESENT** | `build_statute_chunks.py:81–131` |
| `part_num`, `part_heading` | **PRESENT** | |
| `chapter_num`, `chapter_heading` | **PRESENT** | (empty when not applicable) |
| `breadcrumb` | **PRESENT** | e.g. `PRELIMINARY / Section 1` |
| `is_complete_section` | **PRESENT** | Boolean — true when section fits in one window |
| `chunk_index`, `chunk_total`, `token_count` | **PRESENT** | |
| `status` | **PRESENT** | `current` / `amended` |
| `references` | **PRESENT** | Comma-separated cross-reference targets |
| **page number** | **MISSING** | Statutes ingested from pre-parsed JSONL, not from PDF directly; no page mapping |
| **bounding boxes** | **MISSING** | Same reason |
| **character offsets** | **MISSING** | |

#### Constitution (`master_rules`)

| Field | Status | Notes |
|-------|--------|-------|
| `chunk_id` (stable) | **PRESENT** | `const_{filename}_art{N}_chunk{idx}_{hash8}` |
| `source_document` | **PRESENT** | PDF filename |
| `article_number`, `article_header` | **PRESENT** | |
| `chapter`, `chapter_title` | **PRESENT** | |
| `is_complete_article` | **PRESENT** | |
| `chunk_index`, `chunk_length` | **PRESENT** | |
| `ingestion_timestamp` | **PRESENT** | Unix epoch |
| **page number** | **MISSING** | `constitutionbreaker.py:36` iterates pages and logs them (`lines 42, 51, 57`) **but never writes `page_num` to chunk metadata** |
| **bounding boxes** | **MISSING** | fitz supports `page.get_text("dict")` with bbox per block but `page.get_text()` (plain string) is used instead |
| **character offsets** | **MISSING** | |
| **structural label** | **PARTIAL** | article_number + chapter_title present; no sub-article or clause reference |

### 3.3 Sample chunk records

**Record 1 — Case law, `case_law` collection**
```
id:       2026SCC242KLR_0
metadata: {
  "case_identifier":   "2026SCC242(KLR)",
  "file_name":         "Umoja Magharibi (K) Ltd v Lubano (Commercial Case E091of2026) 2026SCC242(KLR) (18September2026) (Judgment)",
  "parties":           "Umoja Magharibi (K) Ltd v Lubano",
  "court_level":       "Small Claims Court",
  "judge":             "A.Z. Ogange",
  "year":              "2026",
  "legal_domain":      "Contract Law",
  "winning_party":     "Respondent",
  "chunk_index":       0,
  "chunk_total":       5,
  "prev_chunk_id":     "",
  "next_chunk_id":     "2026SCC242KLR_1",
  "document_hash":     "c99f57515eeed3db",
  "source_file_path":  "/home/gift/Retreival-Augmented-Generation-AI/backend/SCCCases/Umoja Magharibi...",
  "content_length":    1651,
  "firac_component":   ""
}
doc:  "REPUBLIC OF KENYA\nIN THE SMALL CLAIMS COURT AT MALABA\nSCCCOMM NO. E091 OF 2026\nUMOJA MAGHARIBI (K) LIMITED..."
```

**Record 2 — FIRAC, `legal_summaries` collection**
```
id:       Muruatetu___another_v_Republic__Katiba_Institute___facts_1767042049256
metadata: {
  "firac_component":   "facts",
  "case_identifier":   "Muruatetu & another v Republic; Katiba Institute & 5 others (Amicus Curiae)",
  "court_level":       "Supreme Court of Kenya",
  "year":              "2021",
  "legal_domain":      "Criminal Law",
  "document_hash":     "20a5c00bd0be6ab5025c1d6e3d5ac160f187ad622d444f72000ad9ac78e3c810",
  "source_file_path":  "/home/gift/Retreival-Augmented-Generation-AI/backend/Repo/Res_ipsa_loquitur/Muruatetu...",
  "content_length":    1132,
  "ingested_at":       "2025-12-30 00:00:49",
  "embedding_model":   "intfloat/e5-base-v2"
}
doc:  "The appellants, Francis Karioko Muruatetu and Wilson Thirimbu Mwangi, were convicted of murder and sentenced to death..."
```

**Record 3 — Statute, `statutes` collection**
```
id:       small_claims_court_act__2022-12-31__s1__0
metadata: {
  "act_slug":           "small_claims_court_act",
  "act_title":          "Small Claims Court Act",
  "act_version":        "2022-12-31",
  "frbr_work_id":       "ke/act/2016/2",
  "section_num":        "1",
  "section_heading":    "Short title",
  "part":               "I",
  "part_heading":       "PRELIMINARY",
  "breadcrumb":         "PRELIMINARY / Section 1",
  "is_complete_section": true,
  "is_complete_article": true,
  "token_count":        20,
  "chunk_index":        0,
  "chunk_total":        1,
  "status":             "current",
  "source_document":    "Small Claims Court Act",
  "document_type":      "small_claims_court_act"
}
doc:  "PRELIMINARY / Section 1\n1. Short title\nThis Act may be cited as the Small Claims Court Act."
```

---

## 4. Page/Coordinate Loss Audit

This is the critical failure: location data exists during extraction but is discarded.

### 4.1 Case law — `build_case_chunks.py`

```python
# build_case_chunks.py:46–51
reader = PdfReader(str(path))
for page in reader.pages:          # page object carries .page_number
    t = page.extract_text()        # plain string — coordinates gone
    if t:
        pages.append(t)            # list of strings; index = page num
full_text = "\n\n".join(pages)     # pages merged; original page index discarded
```

`PdfReader` page objects expose `.page_number` (0-based). The loop index is available as `enumerate(reader.pages)` but is never captured. After `"\n\n".join(pages)` at line 51, the page boundary is collapsed to a double-newline — indistinguishable from a paragraph break inside a page.

The downstream chunker (`build_case_chunks.py:72`) splits on `"\n\n"` which can fall anywhere across the merged text, making it impossible to recover which page any chunk came from after the fact.

### 4.2 Constitution — `constitutionbreaker.py`

```python
# constitutionbreaker.py:36–57
for page_num in range(total_pages):
    page = doc[page_num]
    text = page.get_text().strip()          # plain string; blocks/coords ignored
    if len(text) < 50:
        # OCR path
        img = page.get_pixmap(dpi=300)
        ...
        text = pytesseract.image_to_string(img, ...)   # also returns plain string
    full_text += text + "\n"               # concatenated; page_num logged but not stored
```

PyMuPDF's `page.get_text("dict")` returns a dict with `"blocks"` each containing `"bbox"` (x0, y0, x1, y1) and `"lines"` → `"spans"`. The plain `page.get_text()` call discards all of this. The `page_num` variable is used only in `print()` statements (lines 42, 51, 57) and never written to any chunk field.

### 4.3 Digester — `digester.py`

```python
# digester.py:50–54
for page_num, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    if text:
        text_parts.append(text)
        # page_num is available here but never stored
full_text = "\n".join(text_parts)   # merged at line 66; page boundary lost
```

Same pattern: `page_num` is in scope but unused. After join, LangChain's splitter at line 214 operates on the merged string without any page anchor.

### 4.4 Summary

| Script | Page num available? | Stored? | Coordinates available? | Stored? |
|--------|--------------------|---------|-----------------------|---------|
| `build_case_chunks.py` | Yes (loop index) | **No** | No (pypdf plain text) | **No** |
| `constitutionbreaker.py` | Yes (`page_num` var) | **No** | Yes (fitz `get_text("dict")`) | **No** |
| `digester.py` | Yes (`enumerate`) | **No** | No (pypdf plain text) | **No** |
| `build_statute_chunks.py` | N/A (no PDF) | — | N/A | — |

---

## 5. Scanned PDF Detection

Script run against `backend/SCCCases/` and `backend/Repo/`:

**Results:**

| File | Chars extracted | Status |
|------|-----------------|--------|
| All 10 files in `backend/SCCCases/` | 5 000–68 000 | OK — text layer present |
| 15 of 16 files in `backend/Repo/` | 8 000–120 000 | OK |
| `backend/Repo/COURT DOCUMENTS.pdf` | **0** | **SCANNED / NO TEXT LAYER** |

`COURT DOCUMENTS.pdf` has zero extractable characters. If it reaches any ingestion script it will produce an empty-string chunk. `constitutionbreaker.py` has an OCR fallback (`pytesseract`) but the other scripts do not — they will silently write empty documents.

No other scanned files detected in the audited directories. Files outside these two directories were not checked.

---

## 6. Chunk Boundary Quality

### 6.1 Case law — paragraph split at page header (acceptable)

Chunk `2026SCC242KLR_1` ends with:
```
...be disregarded on the basis that the
Judgment SCCCOMM No. E091 OF 2026                         Hon. A.Z Ogange -RM pg. 2
```

Chunk `2026SCC242KLR_2` begins with:
```
Analysis and Determination
5. The Claimant's case is principally founded upon the loan application...
```

**Assessment:** The page header (`Judgment SCCCOMM No. ...`) was absorbed into chunk 1's tail — it is noise that could confuse a retriever. The section heading `Analysis and Determination` starts cleanly at chunk 2. This is a common artifact of pypdf's plain-text extraction, which preserves running headers inline with body text. A heading-aware splitter would avoid absorbing the header.

### 6.2 Statute — short section, single chunk (good)

`small_claims_court_act__2022-12-31__s1__0`:
```
PRELIMINARY / Section 1
1. Short title
This Act may be cited as the Small Claims Court Act.
```

**Assessment:** Good. The breadcrumb is prepended and the section is complete (20 tokens ≤ 480 cap).

### 6.3 Statute — long definition list, single chunk (borderline)

A definitions section (e.g. `section 2 Interpretation`) ingested as one window of 421 tokens. The tokenizer sliding window keeps the entire definition list together, which is correct for retrieval but means the chunk is dense and may match poorly on narrow queries for a single definition.

### 6.4 Constitution — character window cuts mid-sentence (problematic)

Because the character window (`CHUNK_SIZE=1000`, `CHUNK_OVERLAP=300`) snaps only to word boundaries (not sentence or clause boundaries), a window can end mid-sentence within a constitutional sub-clause — e.g., cutting `"...the national government shall—\n(a) make provision for..."` at the list preamble. The next chunk starts with `"(a) make provision..."` with no surrounding context. The 300-char overlap partially mitigates this but does not guarantee clause integrity.

---

## 7. Citation Path

### 7.1 Chunk ID flow

```
ChromaDB query
     │
     ▼
Retriever.py: format_output()
     │  returns per-chunk dict with chunk_id (line 533)
     │
     ▼
/api/qretrieve response
     │  results[].chunks[].chunk_id  ← available to frontend
     │
     ▼
Frontend passes results to /api/generate-legal-answer
     │
     ▼
app.py: generate_legal_answer()
     │  builds source blocks: [C1], [C2], [C3] by case rank (lines 1215–1226)
     │  chunk text (top 3 chunks × top 3 cases) included in LLM prompt
     │  chunk_id NOT included in the LLM prompt
     │
     ▼
LLM synthesises answer citing [C1] / [S1] (case/statute level, not chunk level)
     │
     ▼
API response includes:
     source_review[].id  = "C1" | "S1" etc.   (case/statute level)
     citations_used[]    = ["C1 — Case name"]  (case level)
     coverage, premise_check, unverified_claims_removed
```

**Finding:** Chunk IDs are available in the retrieval response (frontend can read them) but are **not forwarded into the LLM prompt**. The LLM therefore cannot cite a specific chunk — only the parent case or statute. There is no mechanism for the LLM to say "paragraph 3 of chunk `2026SCC242KLR_2`".

### 7.2 LLM prompt detail

`app.py` lines 1202–1226 build `source_blocks` as:
```
[C1] Case Name (Year, Court)
[component] chunk text (first 500 chars)
[component] chunk text ...
```

The `firac_component` label is included per chunk. No chunk ID, no page reference.

### 7.3 API response shape

```json
{
  "source_review": [{"id":"C1","relevant":true,"key_points":[...]}],
  "citations_used": ["C1 — Ndeti v Kaguchwa (2026SCC241(KLR))"],
  "coverage": {"in_corpus":[...],"not_in_corpus":[...]},
  "premise_check": "...",
  "unverified_claims_removed": []
}
```

The frontend can display citations at case/statute level. It cannot highlight a specific chunk in a PDF viewer without additional mapping.

---

## 8. PDF Serving Route — Security Review

**Route:** `GET /api/case-pdf?case_id=...` — `app.py` lines 941–1035  
**Route:** `GET /api/case-laws/file/<path:filename>` — `app.py` (serves uploaded SCC cases)

### 8.1 `/api/case-pdf` (case law viewer)

```python
# app.py ~1013–1018
pdf_path_resolved = pdf_path.resolve()
allowed_roots = [
    RES_IPSA_LOQUITUR_DIR.resolve(),
    SCC_CASES_DIR.resolve()
]
in_allowed = any(
    os.path.commonpath([pdf_path_resolved, root]) == str(root)
    for root in allowed_roots
)
if not in_allowed:
    return jsonify({'error': 'Access denied: path outside allowed directory'}), 403
```

| Check | Result |
|-------|--------|
| `../` traversal | ✅ Safe — `.resolve()` normalises before commonpath |
| Symlink escape | ✅ Safe — `.resolve()` follows symlinks first |
| Double URL encoding | ✅ Safe — single `urllib.parse.unquote()` at line 959 |
| Allowlist enforced | ✅ Safe — exact root match via `commonpath` |

### 8.2 `/api/case-laws/file/<filename>` (SCC uploads)

```python
resolved = (SCC_CASES_DIR / filename).resolve()
if not str(resolved).startswith(str(SCC_CASES_DIR.resolve())):
    return jsonify({'error': 'Invalid path'}), 400
```

Uses `str.startswith()` on resolved paths. This is safe when the root path does not have a trailing slash confusion risk (e.g., `/data/scc` vs `/data/scc2`). Confirmed safe for this project's directory structure.

**Overall verdict on path safety: SECURE.** Both routes implement path traversal protection correctly.

---

## 9. Verdict

### (a) Text-search-based highlighting by chunk ID

**Feasible now, with frontend work.**  
Each chunk carries `chunk_id`, `source_file_path`, and the full chunk text. The frontend already receives `chunk_id` in `/api/qretrieve` results. A PDF viewer (e.g. `react-pdf` or `pdf.js`) can search for the chunk text string within the rendered PDF and highlight matches. This is fragile (text normalisation, hyphenation, running headers) but possible without re-ingestion.

### (b) Bounding-box highlights on the original PDF

**Not possible with current data.**  
No page numbers and no coordinates are stored for any chunk from any collection. The extraction libraries used (pypdf plain text, fitz plain text) either cannot provide coordinates (pypdf) or the coordinate API is not called (fitz). Re-ingestion with coordinate capture is required before bounding-box highlighting is feasible.

---

## 10. Prioritised Change List

| # | Finding | File(s) | Effort | Re-ingest? |
|---|---------|---------|--------|-----------|
| **P1** | Store page numbers during extraction | `build_case_chunks.py:46–51`, `constitutionbreaker.py:36–57`, `digester.py:50–54` | **S** | Yes — all affected collections |
| **P2** | Capture bounding boxes (fitz only for constitution; switch cases to fitz) | `constitutionbreaker.py:38` → use `get_text("dict")`; `build_case_chunks.py` → replace pypdf with fitz | **M** | Yes — after P1 |
| **P3** | Add PyMuPDF + pytesseract to `requirements.txt` | `requirements.txt` | **S** | No |
| **P4** | Strip running page headers from case PDFs before chunking | `build_case_chunks.py` — add regex to drop `"Judgment ... pg. N"` lines | **S** | Yes — case_law |
| **P5** | Pass chunk IDs into LLM prompt for sub-case citation | `app.py:1215–1226` — append `chunk_id` to each block label | **S** | No |
| **P6** | Include chunk IDs in API response `citations_used` | `app.py` — extend citation formatter | **S** | No |
| **P7** | Pre-ingestion scanned-PDF scan; reject or OCR-route non-text files | New helper (≈30 lines) called before any ingestion | **S** | No |
| **P8** | Statute-to-PDF page mapping | Requires mapping section numbers back to source PDFs; no PDF ingestion path today | **L** | Yes — statutes |
| **P9** | Sentence-boundary chunking for constitution | `constitutionbreaker.py:282–304` — add NLTK/spacy sentence tokeniser | **M** | Yes — master_rules |
| **P10** | Store absolute character offsets within merged text | All ingestion scripts — track `start_char` / `end_char` as pages are merged | **M** | Yes — all collections |

### Effort key
- **S** — Small: < 1 day, isolated change  
- **M** — Medium: 1–3 days, moderate refactor  
- **L** — Large: > 3 days, structural change or external dependency

### Minimum viable path to bounding-box highlighting

1. P3 (add deps) — no re-ingest  
2. P1 (page numbers) — re-ingest all  
3. P2 (bbox capture via fitz) — re-ingest all  
4. P5/P6 (wire chunk IDs into LLM + API) — no re-ingest  

The constitution collection can implement P2 immediately (fitz is already used; only the API call needs changing from `get_text()` to `get_text("dict")`). The case law collection requires replacing pypdf with fitz first, which is a one-file change in `build_case_chunks.py`.
