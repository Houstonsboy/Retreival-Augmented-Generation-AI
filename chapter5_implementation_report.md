# Chapter 5 — Implementation Audit
## Sections 5.2–5.5: System Implementation Detail
**GraphRAG Legal Decision-Support System — Kenya's Small Claims Court**

*Read-only audit produced 2026-11-02. All claims are sourced from code inspection,
ChromaDB live queries, the SQLite sidecar, git log, and package manifests. No secrets,
private user data, or Black's Law Dictionary artefacts are reported.*

---

## A. ChromaDB Collections — Live vs Legacy

The system maintains five named ChromaDB collections across three separate
`PersistentClient` database directories. Three are active in the live query
pipeline; two are legacy artefacts from an earlier ingestion approach.

### A.1 Active Collections

| Collection | DB directory | Chunks | Purpose |
|---|---|---|---|
| `case_law` | `backend/CaseLawDB/` | 78 | Primary case-law retrieval (`Retriever.py`) |
| `statutes` | `backend/StatutesDB/` | 1 331 | Primary statute retrieval (`RulesRetriever.py`) |
| `master_rules` | `backend/constitutionlogic/MasterRulesDB/` | 374 | Constitution fallback (`RulesRetriever._query_constitution_fallback`) |

**`case_law`** — stores raw prose chunks produced by `build_case_chunks.py` from
nine active PDF/DOCX case files in `backend/SCCCases/`. Chunk IDs follow the
pattern `<case_identifier>_<index>` (e.g. `2026SCC242KLR_0`). Metadata includes
`case_identifier`, `parties`, `court_level`, `judge`, `year`, `legal_domain`,
`winning_party`, `firac_component` (blank for all current chunks),
`source_file_path`, `chunk_index`, `chunk_total`, `page_start`, `page_end`, and
`ocr` flag.

**`statutes`** — stores section-level chunks produced by `build_statute_chunks.py`
from `data/processed/statutes/sections.jsonl`. Chunk IDs follow
`<act_slug>__<version>__<section_num>__<window_index>`. Metadata includes
`act_slug`, `act_title`, `act_version`, `frbr_work_id`, `breadcrumb`,
`section_num`, `section_heading`, `is_schedule`, `article_number`,
`source_document`, and `document_type`.

**`master_rules`** — stores article-level chunks produced by
`constitutionlogic/constitutionbreaker.py` from the Constitution of Kenya PDF.
Chunk IDs follow `constitution_art<N>_<index>`. This collection is queried only
when the incoming query contains a Constitution alias (e.g. "Constitution",
"CoK"), via `RulesRetriever._query_constitution_fallback()`, which fetches the
top-10 chunks and appends them to the statute results.

### A.2 Legacy Collections (Not Read at Query Time)

| Collection | DB directory | Chunks | Status |
|---|---|---|---|
| `legal_summaries` | `backend/LegalSummariesDB/` | 40 | Legacy; isolated from live pipeline |
| `ChunkDB` (historical) | `backend/ChunkDB_backup_20261031/` | — | Backup only |

**`legal_summaries`** contains 40 chunks from 8 cases (5 FIRAC-tagged components
per case: Facts, Issues, Rules, Application, Conclusion) ingested during an
earlier phase of the project. None of the 8 case identifiers in this collection
match the 9 active files in `backend/SCCCases/`, confirming that this collection
represents a previous ingestion run on different source files. The live
`Retriever.py` reads exclusively from `case_law`; `legal_summaries` is never
queried at runtime.

### A.3 Neo4j — Not in the Live Query Path

A Neo4j graph database was built during the knowledge-graph exploration phase
(commits dated 2026-02-12 and 2026-02-13). At present, **no Python file imported
at Flask startup connects to Neo4j**; `Retriever.py`, `RulesRetriever.py`, and
`app.py` contain no Neo4j driver imports. Neo4j was used during ingestion for
entity extraction and relationship tagging; it does not contribute to any
query-time retrieval step.

### A.4 SQLite Sidecar — `backend/locations.db`

A SQLite database (`chunk_locations` table) stores bounding-box page coordinates
for each chunk, enabling the frontend PDF viewer to scroll to and highlight the
exact passage that was retrieved. The table has 72 rows for `case_law` and 374
rows for `master_rules`. Statute chunks do not have PDF bounding boxes (statute
text is served as structured data, not scrollable PDF). The table schema is:

```
chunk_id TEXT PK, collection TEXT, source_file_path TEXT,
served_file_path TEXT, page_start INT, page_end INT,
boxes_json TEXT, page_dims_json TEXT, label TEXT, ocr INT,
match_score REAL, best_page INT
```

---

## B. Corpus Details

### B.1 Case Law — 9 Active Cases

Source directory: `backend/SCCCases/`. Cases are defined in
`data/processed/cases/cases_meta.json` (11 entries; 2 marked `excluded: true`).

| Case Identifier | Format | Chunks | OCR |
|---|---|---|---|
| `2026SCC241KLR` | PDF | ~8 | No |
| `2026SCC242KLR` | PDF | 4 | No |
| `2026SCC243KLR` | PDF | ~8 | No |
| `2026SCC244KLR` | PDF | ~8 | No |
| `2026SCC245KLR` | PDF | ~9 | No |
| `2026SCC246KLR` | PDF | ~9 | No |
| `2026SCC247KLR` | PDF | ~9 | No |
| `2026SCC248KLR` | PDF | ~7 | No |
| `2026SCC249KLR` | PDF | ~6 | No |

Total: 78 chunks across 9 cases. Two cases were excluded from ingestion
(identifiers withheld to protect case privacy). Chunk sizes target 3 000 chars
with one-paragraph overlap; header/footer regions (y < 54 px or y > page_h − 54)
are stripped.

### B.2 Statutory Corpus — 14 Acts + Constitution

Source: `data/processed/statutes/manifest.json`. The `statutes` collection holds
1 331 chunks from 14 Acts. The `master_rules` collection holds 374 chunks from
the Constitution of Kenya.

| Act slug | Chunks |
|---|---|
| `small_claims_court_act` | 67 |
| `civil_procedure_act` | 257 |
| `evidence_act` | 212 |
| `contract_act` | 69 |
| `limitation_of_actions_act` | 49 |
| `consumer_protection_act` | 142 |
| `employment_act` | 192 |
| `land_act` | 141 |
| `national_land_commission_act` | 28 |
| `land_registration_act` | 56 |
| `sale_of_goods_act` | 47 |
| `hire_purchase_act` | 32 |
| `business_laws_amendment_act` | 16 |
| `magistrates_courts_act` | 23 |
| Constitution of Kenya (`master_rules`) | 374 |

Statute chunk IDs encode act slug, publication date, section number, and a
sliding-window index, making them directly resolvable to a specific section and
subsection window.

---

## C. Preparation and Indexing Pipeline

### C.1 Case Law Chunking — `backend/build_case_chunks.py`

**Inputs:** `data/processed/cases/cases_meta.json` (case registry), PDF/DOCX files
in `backend/SCCCases/`.

**Processing steps:**
1. For each non-excluded case, open the source file.
2. **Scanned-PDF detection:** count extractable text characters. If fewer than a
   threshold, run `ocrmypdf --skip-text` (Tesseract 5.5.0, `eng+osd`) to produce
   a text-layer PDF.
3. **Block extraction (PyMuPDF):** iterate pages; select blocks of type 0 (text);
   skip any block whose bounding-box y-coordinate falls within the top or bottom
   54-pixel header/footer margin.
4. **Paragraph chunking:** accumulate paragraphs up to 3 000 chars. Chunk boundary
   is set at the most recent paragraph break. Adjacent chunks share a one-paragraph
   overlap (`prev_chunk_id` / `next_chunk_id` linked-list metadata).
5. **DOCX fallback:** when the source is a `.docx` file, `python-docx` extracts
   paragraphs; the same paragraph-chunking logic is applied; no coordinate data is
   recorded.
6. **Coordinate recording:** for PDF-sourced chunks, `page_start`, `page_end`,
   and the character-level bounding boxes of the selected blocks are written to
   `backend/locations.db` (`chunk_locations` table), enabling the frontend PDF
   viewer to highlight the exact passage.

**Outputs:** `data/processed/cases/cases_embed.jsonl` (one JSON record per chunk),
`backend/locations.db` (72 case-law rows).

### C.2 Statute Chunking — `backend/build_statute_chunks.py`

**Inputs:** `data/processed/statutes/sections.jsonl` (one record per statutory
section, produced upstream by a statute parser).

**Processing steps:**
1. Load the `intfloat/e5-base-v2` tokenizer (no GPU required).
2. For each section: if the tokenised length is ≤ 480 tokens, store as a single
   chunk. If longer, apply a sliding window of 480 tokens with a 50-token overlap
   (stride = 430), producing multiple sub-section windows.
3. Within each group of windows belonging to the same section, assign
   `prev_chunk_id` / `next_chunk_id` links.

**Output:** `data/processed/statutes/sections_embed.jsonl`.

### C.3 Constitution Chunking — `backend/constitutionlogic/constitutionbreaker.py`

The Constitution PDF is processed in isolation from the main statute pipeline.

**Processing steps:**
1. **Text extraction:** PyMuPDF page-by-page extraction with a per-page OCR
   fallback via `pytesseract` (Tesseract, `--oem 3 --psm 6`) if a page yields
   fewer than 50 characters.
2. **Cleaning:** strip navigation artefacts, page headers (`Constitution of Kenya`,
   `[Rev. YYYY]`), lone Roman numerals, and lines with < 50% alphanumeric ratio.
3. **Article detection:** regex `^(\d+)\.\s+(.+)` identifies numbered articles;
   `CHAPTER` and `PART` headers are tracked for breadcrumb metadata.
4. **Article chunking:** each article's full text is split into 1 000-char windows
   with 300-char overlap; tail chunks shorter than 150 chars are merged into their
   predecessor.
5. **Page assignment:** each article's `page_start` is located by scanning
   per-page raw text for the pattern `^N\.\s` (where N is the article number).
6. **Location recording:** chunk location rows written to `backend/locations.db`
   (374 `master_rules` rows).
7. **Outputs:** `constitutionlogic/Constsection/constitution_embeddable_chunks.json`
   (flat embeddable records) and `constitution_articles.json` (nested backup).

### C.4 Embedding Pipeline — `backend/embed_knowledge.py`

**Model:** `intfloat/e5-base-v2` (768-dimensional, sentence-transformers
library). Runs on CPU. Loaded once per invocation; not shared with the Flask
process.

**Passage prefix:** All texts are prepended with `"passage: "` before encoding
(line 110: `"passage: " + c["text"]`). This is the asymmetric-encoding convention
required by the E5 family: passages use `"passage: "` at ingestion; queries use
`"query: "` at retrieval time.

**Key parameters:**
- Batch size: 32
- Normalisation: `normalize_embeddings=True`
- ChromaDB collection metadata: `{"hnsw:space": "cosine"}`
- Resumable: chunks already present in the collection (by ID) are skipped,
  so the script can be interrupted and restarted safely.

**Invocation pattern:**
```
python embed_knowledge.py --source statutes   # statutes only
python embed_knowledge.py --source cases      # cases only
python embed_knowledge.py --source both       # full corpus
```

The constitution is embedded separately by a script in
`backend/constitutionlogic/` that writes directly to `MasterRulesDB`.

---

## D. Complete Query Pipeline

The query pipeline is split across two sequential HTTP calls initiated by the
frontend. The first call handles classification and retrieval; the second handles
answer generation.

### D.1 Route 1 — `POST /api/qretrieve` (`backend/app.py:1248`)

**Auth:** `@require_role("advocate", "admin")` — Bearer token verified via
Firebase Admin SDK; user profile fetched from Firestore.

**Step 1 — Query classification (`Qclassifier.py`):**
`Qclassifier(user_query)` calls `classify_legal_query()`, which sends the query
to the Groq API with `model=qwen3-27b`, `temperature=0.1`, `max_tokens=1 500`.
The classifier system prompt (422 lines) defines three routing modes:

| Mode | Label | Trigger | Strategy |
|---|---|---|---|
| A | `DIRECT_STATUTORY_LOOKUP` | Explicit section/rule citation or precise statutory question | Minimal vector search; inject exact section text |
| B | `STRATEGIC_LEGAL_ANALYSIS` | Open strategic question requiring doctrine and case law | Full four-dimensional audit (spatial, doctrinal, temporal, procedural) |
| C | `HYBRID_RETRIEVAL` | Question mixing statute and case signals | Dual-track statutory + case law retrieval |

The classifier returns a JSON object with fields: `retrieval_mode`,
`vector_query`, `keywords`, `target_components`, `strategy_critique` (Modes B/C),
`entities` (statutes and parties named), and `source_documents`.
Three retries are attempted on malformed JSON; HTTP 429 responses rotate to the
next API key from a key list stored in the environment.

**Step 2 — Case law retrieval (`Retriever.retrieve_relevant_cases()`):**
Called internally by `Qclassifier()` for Modes B and C.

Seven-step pipeline:
1. Parse `classification` dict to extract `vector_query` and `keywords`.
2. Load `intfloat/e5-base-v2` from local model cache.
3. Open `CaseLawDB` via `chromadb.PersistentClient`.
4. Embed the query with prefix `"query: "` + `vector_query`.
5. **Parallel retrieval:** BM25Okapi over the full corpus (top-50 candidates by
   BM25 score) + ChromaDB HNSW cosine search (top-50 candidates). Fuse results
   with Reciprocal Rank Fusion (k=60, n=30).
6. **Cross-encoder reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
   rescores all fused candidates; scores are sigmoid-normalised.
7. **Parent grouping:** chunks are grouped by `case_identifier`; the top-3 parent
   cases are selected. For each selected parent, adjacent chunks
   (`prev_chunk_id`, `next_chunk_id`) are fetched to provide context around the
   best-scoring chunk. Chunks scoring below `LOW_CONFIDENCE_THRESHOLD = 0.4` are
   demoted. At most `MAX_CHUNKS_PER_COMPONENT = 1` chunk per FIRAC component is
   returned per case.

**Step 3 — Statutory retrieval (`RulesRetriever.retrieve_relevant_rules()`):**
Triggered when `retrieval_mode` is `DIRECT_STATUTORY_LOOKUP` or
`HYBRID_RETRIEVAL`, or when `target_components` contains `"RULES"` or
`"STATUTORY_TEXT"`.

Six-step pipeline:
1. Parse classification dict.
2. Load embedding model.
3. Embed query with `"query: "` prefix.
4. **Parallel retrieval:** BM25 (top-50) + vector (top-50) → RRF (k=60, n=30)
   over `StatutesDB/statutes`.
5. **Schedule penalty:** chunks whose breadcrumb indicates a schedule, form, or
   fee table have their RRF scores multiplied by 0.3 unless the query explicitly
   mentions schedules.
   **Section injection:** if the query contains explicit section numbers and a
   named Act, those sections are fetched by direct ID and prepended to the
   candidate list regardless of similarity score.
   **Constitution fallback:** if the query mentions any Constitution alias
   (`Constitution`, `CoK`, `Bill of Rights`, etc.), `_query_constitution_fallback()`
   fetches the top-10 chunks from `master_rules` by cosine search and appends
   them.
6. **Cross-encoder reranking:** same model as case retrieval; top-5 articles
   returned.

**Response:** JSON with `classification`, `retrieval.results` (cases),
`retrieval.rules` (statutes), `retrieval.rules_count`, and
`retrieval.retrieval_mode`.

---

### D.2 Route 2 — `POST /api/generate-legal-answer` (`backend/app.py:1393`)

The frontend passes the full `retrieval` payload from Step D.1 together with the
original query, classification, conversation ID, and a client-generated UUID
(`client_turn_id`).

**Auth:** `@require_role("advocate", "admin")`

**Step 1 — Idempotency gate (`history_store.check_idempotency`):**
`client_turn_id` is used as the Firestore document ID for the turn.
- If the turn exists with `status="complete"` → the stored answer is returned
  immediately without calling the LLM (`cached: true` in response).
- If the turn exists with `status="failed"` → the turn's `turn_index` is
  retrieved for in-place overwrite; processing continues normally.
- If no turn exists → new turn.

**Step 2 — Coverage check (`backend/coverage.py`):**
Loads `data/processed/statutes/manifest.json` (act aliases). Checks named Acts
from `classification.entities.statutes` against the alias map (exact match and
normalised partial match). Returns `{"in_corpus": [...], "not_in_corpus": [...]}`.
The `not_in_corpus` list is surfaced to the user in the frontend as a warning.

**Step 3 — Build labelled source list:**
Up to 5 statutory provisions are labelled S1–S5; up to 3 case chunks are labelled
C1–C3. Each label maps to a `chunk_id` in the `alias_map` returned to the client.
The labelled source texts are injected into the LLM prompt so the model can cite
them by alias.

**Step 4 — Groq API call:**
Request: `model=qwen3-27b`, `temperature=0`, `max_tokens=2 500`.
The system prompt (lines 1530–1565) instructs the model to respond exclusively
with a 9-field JSON object:

```
{
  "premise_check": string,
  "facts": string,
  "issues": string,
  "rules": string,
  "application": string,
  "conclusion": string,
  "prospects": [{label, rating}],
  "sources_used": [alias, …],
  "source_review": [{alias, relevance_score, justification}, …]
}
```

The model is instructed to cite only from the provided sources using `[S#]` and
`[C#]` notation, produce FIRAC-structured analysis, and never assert facts beyond
the retrieved material. Thinking blocks (`<think>…</think>`) emitted by Qwen3 are
stripped from the response before JSON parsing.

**Step 5 — Filter irrelevant sources:**
Sources with a `source_review.relevance_score` below a threshold are removed
from the returned source list before building the markdown answer.

**Step 6 — Citation verification (`backend/citation_verifier.py`):**
Regex patterns scan the generated text for all statutory references
(section/sec/s, article, rule, Cap, KLR citations, `X v Y` case names). Each
citation is checked against the text of the retrieved sources. Unsupported
citations are logged to `backend/logs/hallucinations.jsonl`.
If unsupported citations are found: one regeneration attempt is made with the
same prompt. If unsupported citations persist after regeneration, sentences
containing them are stripped from the answer.

**Step 7 — Build markdown answer:**
The FIRAC fields from the JSON response are assembled into a Markdown document
with `## FACTS`, `## ISSUES`, `## RULES`, `## APPLICATION`, `## CONCLUSION`
headers, followed by prospects and source attribution.

**Step 8 — Persist to Firestore (`backend/history_store.save_turn`):**
A Firestore transaction writes (or overwrites) the turn document at
`users/{uid}/conversations/{cid}/turns/{client_turn_id}`. Key fields stored:
`query`, `answer_markdown`, `answer_structured` (raw JSON), `sources` (list of
source objects with up to 300-char snippets), `alias_map`, `coverage`,
`retrieval_summary` (rule count, case count), `status="complete"`,
`model`, `latency_ms`, `turn_index`.

A background `ThreadPoolExecutor` thread calls the Groq API
(`temperature=0.2`, `max_tokens=300`, `timeout=10s`) to generate a conversation
title if none exists yet (`title_source="fallback"`). The title is written back
to Firestore only if the conversation still has a fallback title.

A 900 KB size guard truncates `answer_markdown` and source snippets if the turn
document would exceed Firestore's 1 MB document limit.

**Step 9 — Return JSON response:**

```json
{
  "success": true,
  "answer": "<markdown string>",
  "coverage": {"in_corpus": [], "not_in_corpus": []},
  "premise_check": "...",
  "sources": [{alias, chunk_id, label, type, used, precision, …}],
  "alias_map": {"S1": "chunk_id", "C1": "chunk_id", …},
  "sources_used": {"statutory_provisions": N, "case_law": N},
  "conversation_id": "...",
  "turn_id": "...",
  "history_saved": true,
  "conversation_title": "...",
  "cached": false
}
```

---

## E. Authentication and Firestore Data Model

### E.1 Firebase Authentication

The system uses Firebase Authentication with two providers: email/password and
Google Sign-In. The backend never stores passwords; all credential validation is
delegated to Firebase Auth.

The `require_role(*roles)` decorator in `backend/auth.py` enforces authentication
on every protected route:
1. Extracts the `Authorization: Bearer <token>` header.
2. Calls `firebase_admin.auth.verify_id_token(token)` to validate the JWT and
   obtain the Firebase UID.
3. Fetches `users/{uid}` from Firestore via the Admin SDK.
4. Checks `profile["type"]` against the required roles (`"advocate"` and/or
   `"admin"`).
5. Sets `flask.g.uid`, `flask.g.profile`, and `flask.g.user_type` for use
   downstream.

All routes in the main query pipeline require `"advocate"` or `"admin"`.

### E.2 User Roles and Provisioning

**Client signup flow:** A new user who signs up via the frontend creates a
Firestore document at `users/{uid}` with `type="advocate"` and
`advocateVerified=false`. The Firestore security rules enforce that
`type` is always `"advocate"` on client-initiated creates (rule: line 25).

**Admin creation:** `backend/scripts/bootstrap_admin.py` creates a Firebase Auth
user and a Firestore profile with `type="admin"` via the Admin SDK, bypassing the
client-facing security rules. Run once from the terminal with
`python bootstrap_admin.py <email> <password> <username>`.

**Role change:** `backend/scripts/set_role.py` looks up a user by email, verifies
their Firestore profile exists, then calls `doc_ref.update({"type": role})`. Valid
roles: `"admin"`, `"advocate"`.

**Username uniqueness:** An atomic double-write — `users/{uid}.username` and
`usernames/{username}.uid` — enforces uniqueness. The Firestore rule uses
`existsAfter` / `getAfter` to confirm both documents are written in the same
request batch (lines 29–30 and 50–51 of `firestore.rules`).

### E.3 Firestore Collection Schema

```
users/{uid}
  username        : string  (3–30 chars, [a-z0-9_])
  email           : string
  type            : "advocate" | "admin"
  authProvider    : "password" | "google"
  advocateVerified: boolean  (always false on create; admin-updated externally)
  createdAt       : timestamp
  updatedAt       : timestamp

usernames/{username}
  uid             : string

CaseLaws/{docId}           ← admin-written; client read-only
  (document structure written by Admin SDK; not exposed through backend routes)

users/{uid}/conversations/{convId}
  uid             : string
  title           : string
  title_source    : "fallback" | "llm" | "user"
  pinned          : boolean
  turn_count      : number
  created_at      : timestamp
  updated_at      : timestamp

users/{uid}/conversations/{convId}/turns/{turnId}
  turn_index      : number   (sequential within conversation)
  query           : string
  answer_markdown : string | null
  answer_structured: object | null  (raw 9-field JSON from LLM)
  sources         : array           (ChunkSource objects, snippets ≤ 300 chars)
  alias_map       : object          ({"S1": chunk_id, …})
  coverage        : object          ({"in_corpus": [], "not_in_corpus": []})
  retrieval_summary: object         ({rules_count, cases_count})
  status          : "complete" | "failed"
  model           : string
  latency_ms      : number
  error           : string | null
  error_code      : string | null
  unverified_claims_removed_count: number
  created_at      : timestamp
```

### E.4 Firestore Security Rules Summary

- `users/{uid}`: owner read; create enforces 7 required fields, `type="advocate"`,
  `advocateVerified=false`, and atomic username claim; update restricted to
  `username` + `updatedAt` only; delete forbidden.
- `usernames/{username}`: any-authenticated read; create atomic with user doc;
  delete allowed by owner during username change only; update forbidden.
- `CaseLaws/{docId}`: any-authenticated read; write forbidden (Admin SDK only).
- `users/{uid}/conversations/{convId}` and `.../turns/{turnId}`: owner read;
  write forbidden (Admin SDK only).
- Catchall `/{document=**}`: read and write forbidden.

### E.5 Composite Firestore Indexes

Three composite indexes are defined in `backend/firestore.indexes.json`:
1. `conversations`: `uid ASC, pinned DESC, updated_at DESC` — powers the
   pinned-conversations-first sidebar query.
2. `conversations`: `uid ASC, updated_at DESC` — powers the unpinned timeline
   query.
3. `turns`: `turn_index ASC` — powers ordered turn loading within a conversation.

---

## F. Frontend Pages and Workflow

The frontend is a Next.js 16.0.3 / React 19.2.0 application using the App Router
and client components throughout. Authentication state is provided by an
`AuthContext` backed by the Firebase JS SDK.

### F.1 Page Inventory

| Route | Component | Description |
|---|---|---|
| `/` | `ChatView` | Main research workspace (new conversation) |
| `/chat/[cid]` | `ChatView` (with `initialConversationId`) | Reload existing conversation from Firestore history |
| `/dashboard` | `Dashboard` | Entry hub with tiles for Chat and Internal Repository |
| `/repository` | (not audited) | Document browser for uploaded files |
| `/login`, `/signup`, `/profile-setup` | (auth pages) | Firebase Auth sign-in flow |

### F.2 ChatView Workflow

`frontend/app/components/ChatView.tsx` (1 816 lines) is the primary interface.
It renders a resizable two-column layout: a chat column on the left and a
document workspace panel on the right.

**On mount:**
- `loadSidebar()` fetches the first page of conversations from
  `GET /api/history/conversations` and populates the drawer.
- If `initialConversationId` is provided (route `/chat/[cid]`),
  `loadConversation(cid)` fetches the full turn list from
  `GET /api/history/conversations/{cid}` and reconstructs `Message` objects.
  While loading, `ChatPanelSkeleton`, `SidebarSkeleton`, and
  `DocumentViewerSkeleton` shimmer components are displayed.

**Query submission (`handleSubmit` → `doRequest`):**
1. A client UUID (`crypto.randomUUID()`) is created as `client_turn_id`.
2. `POST /api/qretrieve` — sends `{query}`. Response contains `classification`
   and `retrieval` (case chunks + statute chunks).
3. `POST /api/generate-legal-answer` — sends `{query, retrieval, retrieval_mode,
   classification, conversation_id, client_turn_id}`. Returns the markdown
   answer, sources, alias map, coverage, and history metadata.
4. On success: the `Message` object is appended to state. If this is a new
   conversation, `window.history.replaceState` updates the URL to `/chat/{cid}`
   without triggering a Next.js navigation (avoiding a ChatView remount that
   would discard the in-memory answer).
5. `startTitleListener(cid)` polls `GET /api/history/conversations/{cid}` every
   2 seconds for up to 30 seconds until `title_source` is no longer `"fallback"`,
   then stops.

**Retry:** If an assistant `Message` has `historyStatus="failed"`, a Retry button
calls `doRequest(msg.query, msg.turnId)`, which passes the original
`client_turn_id`. The backend idempotency gate detects the failed turn and
overwrites it.

**Citation interaction:** Clicking a `[S#]` or `[C#]` citation pill calls
`handleCitationClick(alias, msg)`, which resolves the alias through `aliasMap`
to a `chunk_id`, then sets `activeChunkId` to scroll the right-panel PDF viewer
to the highlighted passage.

**Resizable splitter:** A draggable divider between the chat column and the
workspace panel uses `mousemove` listeners to update `chatWidth` state between
260 px and 65% of viewport width.

**Sidebar conversation management:** The drawer supports pin/unpin
(`PATCH /api/history/conversations/{cid}`), rename (same PATCH endpoint), and
delete (`DELETE /api/history/conversations/{cid}`). Deleted conversations are
removed from React state; if the active conversation is deleted, the workspace
resets to a blank chat.

### F.3 AssistantCard Rendering

The `AssistantCard` component:
1. Calls `parseFirac(aiResponse)` — searches for FIRAC section markers
   (`## FACTS`, `## ISSUES`, etc.) in the markdown; returns structured sections
   if at least two markers are found.
2. Renders each FIRAC section under a coloured tile (F / I / R / A / C) with a
   gold separator.
3. `parseMdFirac` renders inline markdown: bold, italic, code spans, block
   quotes, ordered/unordered lists, and headings. Citation aliases matching
   `[C#]` or `[S#]` are replaced with interactive `CitPill` buttons.
4. A `ProspectsStrip` renders litigation prospect ratings (Strong / Moderate /
   Weak) returned by the LLM.
5. The sources bar at the bottom lists only `used: true` sources as `CitPill`
   buttons.

### F.4 Dashboard

`frontend/app/dashboard/page.tsx` is a minimal hub page that greets the signed-in
user by `profile.username` and offers two navigation cards: **Chat** (→ `/`) and
**Internal Repository** (→ `/repository`). It does not interact with the backend
directly.

### F.5 Skeleton Loading System

A dedicated skeleton system (`frontend/app/components/skeleton/`) provides
shimmer placeholders during data fetches:
- `ChatPanelSkeleton` — shimmer for the message list area.
- `SidebarSkeleton` — shimmer for the conversation list in the drawer.
- `WorkspaceSkeleton` — shimmer for the right-panel workspace.
- `DocumentHeaderSkeleton`, `DocumentBodySkeleton` — shimmer for document card.
- `SourcesCardSkeleton` — shimmer for the sources attribution row.
- `Reveal` — fade-in wrapper applied to each `Message` with a staggered delay
  of up to 360 ms.

---

## G. Development History

The project was developed from May 2025 to November 2026, progressing through
four identifiable phases.

### Phase 0 — Prototype (May–October 2025)
- `2025-05-18` Initial commit: a working RAG script (Python).
- `2025-10-23–24`: Google Colab prototypes using an online Mistral LLM;
  ChromaDB first introduced as a vector store.

### Phase 1 — FIRAC Ingestion and Embedding (November 2025 – February 2026)
- `2025-11-09`: ChromaDB digester added; first pipeline from PDF to vector store.
- `2025-11-24–27`: Fact extraction experiments; FIRAC component labelling.
- `2025-12-11`: FIRAC extraction complete for Phase 1. Case chunks were tagged
  with FIRAC components and stored in what is now the `legal_summaries` collection.
- `2025-12-25–29`: `legal_domain`, `winning_party`, and `embedding_work` commits.
  Token budget issues encountered with large FIRAC prompts.
- `2026-01-03`: `Qclassifier` first version: spatial, temporal, doctrinal,
  procedural reasoning dimensions added.
- `2026-01-09`: Constitution chunking (`constitutionbreaker.py`) committed;
  article-level identification by regex.
- `2026-01-12–13`: Constitution embedded into `master_rules`; Evidence Act added
  to statute corpus.
- `2026-01-19–30`: Frontend output wired; retrieval accuracy improved by reducing
  similarity threshold; PDF rendering on browser. MIT licence added.
- `2026-02-12–13`: Neo4j knowledge graph exploration; knowledge tester committed.

### Phase 2 — Hybrid Retrieval and Auth (October 2026)
- `2026-10-26`: Firebase Auth and Firestore RBAC added; `legalai-scc` Firebase
  project wired. Role-based access control enforced on all API routes.
- `2026-10-26`: Hybrid retrieval rewrite: BM25 + RRF + cross-encoder;
  `StatutesDB` knowledge switch; section injection guard.
- `2026-10-31` (Fix commits A–F):
  - Fix A: Replaced `<object>` PDF embed with `react-pdf` + bounding-box overlay.
  - Fix B: Constitution bounding-box computation via `rapidfuzz` word-alignment.
  - Fix C/D/E: Statute chunk aliases, citation precision tagging, cluster widget.
  - Fix F: End-to-end test (Amani/M-Pesa query); precision tagging for bare `[S#]`.
  - `0fc40a7`: Constitution chunk ID mismatch resolved; bbox script corrected.

### Phase 3 — History, Idempotency, and Production UI (November 2026)
- `2026-11-01` (multiple commits, same day):
  - History persistence layer (`history_store.py`) and integration tests.
  - `groq_client` module extracted from inline closures.
  - History wired into `generate-legal-answer`; Flask Blueprint for history
    management endpoints.
  - Firestore security rules, composite indexes, and `CHAT_HISTORY` guide
    documented.
  - Idempotent generate endpoint: cache hit skips LLM; retry overwrites failed
    turn.
  - Phase 2 frontend: history routing, sidebar conversation drawer, `historyApi`
    client library.
  - Navigate to dashboard after profile setup without requiring page refresh.
  - Production skeleton loading system (`ChatPanelSkeleton`, `WorkspaceSkeleton`,
    `SidebarSkeleton`, `Reveal`).
  - Round 2 skeleton: viewport-anchored shimmer and unified `Skeleton` component.

---

## Chapter 4 Design vs Implementation

The following table maps key design claims to their observed implementation state.

| Design Claim | Implementation Status | Evidence |
|---|---|---|
| **GraphRAG**: knowledge graph augments retrieval at query time | **Partial** — Neo4j was built (Phase 1–2) but is not in the live query path. Hybrid retrieval (BM25 + vector + RRF + cross-encoder) substitutes for graph traversal at runtime. | No Neo4j driver import in `app.py`, `Retriever.py`, or `RulesRetriever.py`; Neo4j commits 2026-02-12–13 |
| **Multi-collection hybrid retrieval** | **Implemented** — `case_law` (cases), `statutes` (Acts), `master_rules` (Constitution) queried per turn with BM25 + cosine + RRF (k=60, n=30) + cross-encoder reranking | `Retriever.py`, `RulesRetriever.py` |
| **FIRAC-structured output** | **Implemented** — LLM is prompted to return a 9-field JSON including `facts`, `issues`, `rules`, `application`, `conclusion`; frontend renders with FIRAC tiles | `app.py:1530–1565`; `ChatView.tsx:parseFirac` |
| **Qwen3 27B LLM via Groq API** | **Implemented** | `app.py`, `Qclassifier.py`, `groq_client.py` |
| **Three query routing modes** | **Implemented** — Mode A (direct statutory lookup), Mode B (strategic analysis), Mode C (hybrid retrieval) | `Qclassifier.py:SYSTEM_PROMPT` |
| **Section-exact injection for statutes** | **Implemented** — when explicit section numbers + named Act detected, sections fetched by direct chunk ID and prepended to candidate list | `RulesRetriever.py:_section_injection` |
| **Hallucination control via citation verification** | **Implemented** — regex audit of all section/article/rule/Cap/KLR/case references; one regeneration attempt; sentence-level stripping as last resort; logged to `hallucinations.jsonl` | `backend/citation_verifier.py` |
| **Corpus coverage check** | **Implemented** — named Acts compared against `manifest.json` aliases; `not_in_corpus` list returned to client | `backend/coverage.py` |
| **PDF source viewer with passage highlighting** | **Implemented** — `react-pdf` + bounding-box overlay; chunk coordinates stored in `locations.db`; `rapidfuzz` word-alignment for Constitution | `backend/locations.db`; Fix A/B commits 2026-10-31 |
| **Role-based access control** | **Implemented** — Firebase Auth JWT verification; Firestore `users/{uid}.type` checked on every protected route; `advocate` and `admin` roles; Firestore security rules enforce client-side invariants | `backend/auth.py`; `backend/firestore.rules` |
| **Persistent chat history** | **Implemented** — Firestore subcollection `users/{uid}/conversations/{cid}/turns/{tid}`; idempotent writes using `client_turn_id` as document ID | `backend/history_store.py` |
| **Idempotency** | **Implemented** — complete turns returned from cache; failed turns overwritten using the original `turn_index` | `backend/history_store.check_idempotency`; `app.py` |
| **Background title generation** | **Implemented** — `ThreadPoolExecutor`; Groq API call (`temperature=0.2`, `max_tokens=300`); written to Firestore only if `title_source="fallback"` | `backend/history_store.submit_title_generation` |
| **Skeleton loading UI** | **Implemented** — viewport-anchored shimmer placeholders for sidebar, chat panel, and document viewer | `frontend/app/components/skeleton/` |
| **FIRAC-tagged case law chunks in `legal_summaries`** | **Superseded** — `legal_summaries` (40 chunks, 8 cases) exists but is not queried. FIRAC structuring moved to LLM prompt-level; raw case text is chunked by paragraph into `case_law`. | `legal_summaries` collection; `build_case_chunks.py` (no FIRAC tagging) |

---

## NOT FOUND / Needs Confirmation

The following items could not be fully confirmed from a read-only audit and may
require the developer to verify directly.

1. **`backend/CaseLaw/firac.py` and `backend/CaseLaw/ingester.py`**: These files
   exist and were used in Phase 1 to produce the `legal_summaries` collection.
   They are not imported by any Flask route; their FIRAC extraction prompt and
   JSON schema are not part of the live system but were the original ingestion
   mechanism.

2. **OCR coverage in active cases**: All 9 active case PDFs have `ocr=0` in the
   `chunk_locations` table, indicating `ocrmypdf` was not invoked for any of
   them. It is not confirmed whether this reflects genuinely text-layer PDFs or
   whether some cases were skipped.

3. **`backend/compute_constitution_bboxes.py`**: Referenced in a 2026-10-31 Fix B
   commit as a `rapidfuzz`-based word-alignment script that computes per-chunk
   bounding boxes for the Constitution. This file was not read; the extent of
   bbox coverage for constitution chunks (currently `boxes_json="[]"` in
   `locations.db`) is not confirmed.

4. **`/repository` page**: The Internal Repository page (`frontend/app/repository/`)
   was not audited. Its backend routes, if any, are not covered in this report.

5. **`advocateVerified` field usage**: Firestore rules always create
   `advocateVerified=false`. No route in `app.py` was observed to gate access
   on this flag. The intended verification workflow (e.g. admin approval before
   an advocate can query) is not implemented in the current codebase.

6. **Key rotation list**: `Qclassifier.py` implements 429-triggered API key
   rotation from a list of Groq API keys. The number of keys configured is not
   known from a read-only audit; the environment variable name(s) are intentionally
   omitted per audit rules.
