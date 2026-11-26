# Backend Retrieval Pipeline — Kenyan Legal RAG

## Architecture Overview

The backend is a Flask application that exposes two distinct retrieval paths depending on the entry point:

| Path | Entry Route | Purpose |
|---|---|---|
| **Chat RAG** (`pixe.py`) | `POST /api/chat` | General Q&A over ingested case law chunks |
| **Classifier RAG** (`Qclassifier` → `Retriever` + `RulesRetriever`) | `POST /api/qretrieve` | Structured legal research with mode-aware routing |

Both paths share the same embedding model (`intfloat/e5-base-v2`, 768-dimensional) and the same cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`). All LLM calls go to Groq using model `qwen/qwen3.8-27b`.

---

## Path A — Chat RAG (`pixe.py`)

Used by `/api/chat`. Simpler pipeline; treats all queries the same.

### Step 1 — Query Expansion
```
User query → Groq LLM → expanded query (synonyms, related terms)
```
- Temperature 0.3, max 150 tokens
- Falls back to original query on LLM failure
- Controlled by `USE_QUERY_EXPANSION = True`

### Step 2 — Hybrid Search
```
Expanded query → TF-IDF (keyword) + e5-base-v2 cosine (semantic)
                 combined as: 0.7 × semantic + 0.3 × keyword
```
- Both scores are min-max normalized before combining
- Falls back to semantic-only if corpus has < 2 chunks

### Step 3 — Two-Stage Retrieval
```
Hybrid scores → top 10 candidates → CrossEncoder rerank → top 3
```
- CrossEncoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`, max_length=512
- Reranked by raw CE logit (higher = more relevant)

### Step 4 — Deduplication
- Word-overlap deduplication: if two chunks share > 70% of words, the lower-ranked one is dropped
- Prevents the LLM from seeing near-identical context blocks

### Step 5 — Response Generation
- System prompt: **answer ONLY from provided context**; cite as `[Chunk N]`
- Chunk numbers are referenced in the response; citation metadata appended
- Temperature 0.3, max 800 tokens
- ChromaDB source: `backend/ChunkDB` (FIRAC-segmented case law)

---

## Path B — Classifier RAG

Used by `/api/qretrieve`. Three-module pipeline: classify → retrieve cases → retrieve statutes.

---

### Module 1 — Query Classifier (`Qclassifier.py`)

Every query is classified by Groq (temperature 0.1) before any retrieval. The classifier outputs a JSON object with routing metadata.

#### Routing Decision

The classifier places every query into one of three modes:

| Mode | Triggers | Example |
|---|---|---|
| `DIRECT_STATUTORY_LOOKUP` | Explicit citation, content verification, "What does Article X say?" | "What does Article 37 say about assembly?" |
| `STRATEGIC_LEGAL_ANALYSIS` | Litigation strategy, doctrine application, precedent search | "Can I sue the government for wrongful arrest?" |
| `HYBRID_RETRIEVAL` | Both statutory text + case law needed | "Show me Section 204 and cases that challenged it" |

#### Classifier Output (abridged schema)

```json
{
  "retrieval_mode": "DIRECT_STATUTORY_LOOKUP | STRATEGIC_LEGAL_ANALYSIS | HYBRID_RETRIEVAL",
  "intents": ["DIRECT_ARTICLE_RETRIEVAL", "OUTCOME_ANALYSIS", ...],
  "target_components": ["STATUTORY_TEXT"] | ["FACTS", "ISSUES", "RULES", "APPLICATION", "CONCLUSION"],
  "legal_domains": [{ "domain": "Criminal Law", "confidence_score": 0.9 }],
  "citation_analysis": {
    "explicit_citations": ["37"],
    "implied_subject": ["freedom of assembly"],
    "citation_confidence": "HIGH | MEDIUM | LOW | MISMATCH"
  },
  "strategy_critique": {
    "spatial": "ELC required for land disputes",
    "temporal": "3-year limitation period under Cap 22",
    "procedural": "Exhaust internal ADR first",
    "doctrinal": null
  },
  "entities": {
    "statutes": { "primary": ["Article 37, Constitution of Kenya 2010"], "contextual": [] },
    "article_numbers": ["37"],
    "section_numbers": [],
    "source_documents": ["Constitution of Kenya 2010"],
    "cases": [],
    "judges": []
  },
  "vector_query": "Article 37 Constitution Kenya",
  "vector_query_statutory": "Article 37 Constitution Kenya",  // HYBRID only
  "vector_query_caselaw": "Article 37 freedom of assembly court challenge",  // HYBRID only
  "retrieval_strategy": "STATUTORY_FIRST_THEN_CASELAW",  // HYBRID only
  "reasoning_summary": "..."
}
```

**Citation mismatch handling**: if the user states an incorrect article ("Article 50 says you can protest"), the classifier flags `citation_confidence: "MISMATCH"` and retrieves both the stated and inferred articles, then the LLM explains the discrepancy.

**API key rotation**: supports `GROQ_API_KEY`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3`. On 429, rotates to next key before sleeping.

---

### Module 2 — Case Law Retrieval (`Retriever.py`)

Active when `retrieval_mode` is `STRATEGIC_LEGAL_ANALYSIS` or `HYBRID_RETRIEVAL`.

Searches `ChunkDB` (FIRAC-segmented case law). Returns top **3 parent documents**.

#### Step 1 — BM25 (Sparse Retrieval)
```
All ChunkDB documents → BM25Okapi.get_scores(query.split()) → top 50 chunk IDs
```
- Index is loaded once per process into module-level `_CASES_BM25`
- Tokenized by whitespace

#### Step 2 — Vector Search (Dense Retrieval)
```
"query: " + vector_query → e5-base-v2 embed → ChromaDB cosine → top 50 chunk IDs
```
- For `HYBRID_RETRIEVAL`, uses `vector_query_caselaw` field

#### Step 3 — RRF Fusion
```
RRF_score(chunk) = 1/(60 + rank_bm25) + 1/(60 + rank_vec)
Top 30 chunks by RRF score kept for reranking
```

#### Step 4 — CrossEncoder Rerank
```
(query_str, chunk_text) pairs → CE logit → sigmoid → effective score
Sorted descending
```

#### Step 5 — Group by Parent + Enrich
- Chunks grouped by `case_identifier` metadata field
- Adjacent chunks (`prev_chunk_id`, `next_chunk_id`) fetched and appended
- Overlapping regions between consecutive chunks stripped by suffix-prefix match (`_stitch_chunks`)
- Parents sorted by best CE-sigmoid score; top 3 selected

#### Step 6 — LLM Case Summaries
For each of the top 3 cases, Groq generates a 4-section analysis:
- **CASE OVERVIEW** — parties, dispute, outcome
- **KEY LEGAL PRINCIPLES** — doctrines established
- **RELEVANCE TO YOUR QUERY** — why this case applies
- **CRITICAL FACTS** — decisive factual elements

Temperature 0.3, max 1000 tokens. Falls back to raw FIRAC text concatenation on API error.

#### Output per Case
```json
{
  "rank": 1,
  "case_name": "Republic v Kamau [2018]",
  "score": 0.847,
  "ai_summary": "...",
  "metadata": { "court_level": "High Court", "judge": "...", "year": 2018, "legal_domain": "Criminal Law", "winning_party": "..." },
  "pdf_url": "/api/case-pdf?case_id=...",
  "matched_components": ["RULES", "APPLICATION"],
  "best_chunk": "stitched text of top evidence chunks",
  "chunks": [{ "component": "RULES", "text": "...", "score": 0.831, ... }],
  "evidence_count": 3
}
```

---

### Module 3 — Statute Retrieval (`RulesRetriever.py`)

Active when `retrieval_mode` is `DIRECT_STATUTORY_LOOKUP` or `HYBRID_RETRIEVAL`, or when `target_components` includes `"RULES"`.

Searches `StatutesDB` (14 Kenyan Acts). Returns top **5 articles**.

#### Where Filter Construction

Before retrieval, metadata filters are built from classifier output in priority order:

1. `article_numbers` → `{ "article_number": { "$in": ["37"] } }`
2. `section_numbers` → same field (Evidence Act uses same metadata key)
3. `source_documents` → mapped to actual PDF filenames via `STATUTE_MAP` (statute manifest)
4. Fallback: legacy `entities.statutes` list

No filter (empty dict) = search across all 14 acts.

#### Hybrid Search + Exact Injection
```
BM25 (filtered) top 50 + Vector (filtered) top 50 → RRF top 30
```
**Section exact-lookup injection**: if the query names explicit section/article numbers AND names a specific Act, the retriever additionally injects any chunks from that Act with matching `article_number` metadata that weren't already in the RRF top 30. Up to 35 candidates total.

#### Constitution Fallback
If the classifier signals a constitution query (`needs_constitution = True`), an additional search runs against `MasterRulesDB` filtered to `document_type = "constitution"` (top 10). Results are merged into the CE candidate pool.

#### CrossEncoder + Schedule Penalty
```
CE logit → sigmoid → effective_score × schedule_mult
schedule_mult = 0.3  if chunk.is_schedule AND query doesn't mention "schedule/form/fee/tariff"
schedule_mult = 1.0  otherwise
```
This suppresses schedule/annexure chunks from surfacing for substantive legal queries.

#### Group by Article
Chunks grouped by `(document_type, article_number)` key. Adjacent chunks fetched and stitched for full article text. Articles sorted by best CE score; top 5 returned.

#### Output per Article
```json
{
  "rank": 1,
  "document_label": "Constitution of Kenya",
  "article_number": "37",
  "article_header": "Assembly, demonstration, picketing and petition",
  "chapter": "4",
  "chapter_title": "The Bill of Rights",
  "source_document": "Constitution of Kenya 2010.pdf",
  "score": 0.912,
  "text_preview": "Every person has the right, peaceably and unarmed, to...",
  "full_text": "...",
  "pdf_url": "/api/rule-pdf?doc_type=constitution&source=..."
}
```

---

## ChromaDB Collections

| Collection | Config Key | Content | Used By |
|---|---|---|---|
| `ChunkDB` | `CASELAW_DB_DIR` / `CASELAW_COLLECTION_NAME` | FIRAC-segmented case law chunks | `pixe.py`, `Retriever.py` |
| StatutesDB | `STATUTES_DB_DIR` / `STATUTES_COLLECTION_NAME` | 14 Kenyan Acts chunked by article/section | `RulesRetriever.py` |
| `MasterRulesDB` | `MASTER_RULES_DB_DIR` / `MASTER_RULES_COLLECTION_NAME` | Constitution provisions (document_type=constitution) | `RulesRetriever.py` (fallback) |

---

## LLM Configuration

| Parameter | Value |
|---|---|
| Provider | Groq API |
| Classifier model | `qwen/qwen3.8-27b` |
| Retriever summary model | `qwen/qwen3.8-27b` |
| Chat RAG model | `qwen/qwen3.8-27b` |
| Classifier temperature | 0.1 (deterministic routing) |
| Summary temperature | 0.3 (factual) |
| Query expansion temperature | 0.3 |
| Chat response temperature | 0.3 |
| API key env vars | `GROQ_API_KEY`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3` |

---

## API Route Summary

| Route | Auth | Retrieval Used |
|---|---|---|
| `POST /api/chat` | advocate, admin | `pixe.py` — TF-IDF + semantic + CE rerank over ChunkDB |
| `POST /api/qretrieve` | advocate, admin | `Qclassifier` → `Retriever` (cases) + `RulesRetriever` (statutes) |
| `POST /api/generate-legal-answer` | advocate, admin | LLM generation only (no retrieval) |
| `POST /api/firac` | advocate, admin | `CaseLaw/firac.py` — extract FIRAC from a PDF |
| `GET /api/files` | advocate, admin | List Repo/ directory |
| `GET /api/files/<filename>` | advocate, admin | Serve protected PDF |
| `POST /api/upload` | admin | Save PDF to Repo/ |
| `POST /api/digest` | admin | FIRAC extract → embed into ChunkDB |
| `POST /api/ingest-firac` | admin | FIRAC data → embed into ChunkDB |
| `POST /api/digest/constitution` | admin | Extract constitution articles → embed into MasterRulesDB |
| `DELETE /api/digest/constitution/reset` | admin | Drop and recreate MasterRulesDB |
| `GET /api/health` | public | Health check |
| `GET /api/case-pdf` | public | Serve case PDF for browser rendering |
| `GET /api/rule-pdf` | public | Serve statute PDF for browser rendering |
| `GET /api/digest/constitution/status` | public | Constitution ingestion status |

---

## Chunk Metadata Schema

### Case Law Chunks (ChunkDB)
| Field | Description |
|---|---|
| `case_identifier` | Parent document ID (used for grouping) |
| `firac_component` | `FACTS`, `ISSUES`, `RULES`, `APPLICATION`, `CONCLUSION` |
| `chunk_index` | Position within case |
| `prev_chunk_id`, `next_chunk_id` | Adjacent chunk IDs for stitching |
| `file_name`, `parties`, `court_level`, `judge`, `year`, `legal_domain`, `winning_party` | Case metadata |
| `source_file_path` | PDF path used to build `/api/case-pdf` URL |

### Statute Chunks (StatutesDB)
| Field | Description |
|---|---|
| `article_number` | Article/section number (string) |
| `article_header` | Article title |
| `chapter`, `chapter_title` | Chapter location |
| `part`, `part_title` | Optional part within chapter |
| `document_type` | Act slug (e.g., `constitution`, `penal_code`) |
| `source_document` | Full act title for PDF lookup |
| `is_schedule` | Boolean — suppressed 0.3x by schedule penalty |
| `chunk_index`, `prev_chunk_id`, `next_chunk_id` | For overlap stitching |

---

## Improvement Vectors for Agent Analysis

1. **Query expansion in Retriever path**: `pixe.py` expands queries before retrieval; the `Qclassifier` → `Retriever` path does not. Adding expansion to the `vector_query` before embedding could improve sparse-retrieval recall.

2. **RRF k=60 tuning**: reducing k (e.g., 30) gives higher weight to top-ranked items and narrows the candidate pool; increasing it smooths rank fusion. Currently hardcoded at 60 in both retrievers.

3. **Separate BM25 indices per retriever**: both `Retriever.py` and `RulesRetriever.py` load their own BM25 index at startup. Since they target different collections, this is correct — but each loads the full corpus into memory on first query. Consider warm-up at app start.

4. **Schedule penalty**: statutes marked `is_schedule=True` are penalized by 0.3× unless the query explicitly mentions schedule-related terms. Adjust terms in `_SCHEDULE_TERMS` or the penalty factor to change suppression behaviour.

5. **Constitution dual-source**: constitution articles may appear in both `StatutesDB` and `MasterRulesDB`. The fallback merges both into the CE candidate pool. If duplicate article chunks appear, the CE ranking resolves them — but deduplication at the article-group level is not applied.

6. **Chunk adjacency enrichment**: stitching via `prev_chunk_id`/`next_chunk_id` runs after retrieval and is non-fatal (ignores missing IDs). If ingestion doesn't set these fields, stitching silently skips. Verify during ingest that adjacent IDs are always written.

7. **CrossEncoder model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` is a general-purpose MS MARCO model. A legal-domain fine-tuned cross-encoder would improve reranking precision for Kenyan case law phrasing.

8. **FIRAC component targeting**: `Retriever.py` scores higher for chunks whose `firac_component` matches `target_components`. For `DIRECT_STATUTORY_LOOKUP` queries, `target_components` is `["STATUTORY_TEXT"]` — which never matches any `firac_component` in ChunkDB, so component boost is never applied. This is intentional (case retrieval is skipped for pure statutory lookups).

9. **LLM summary cost**: each `/api/qretrieve` call that returns 3 cases makes 3 separate Groq summary calls (plus 1 classifier call) = 4 API calls minimum. Rate-limit impact scales with concurrency.
