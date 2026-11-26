# PREFLIGHT — Case-Law GraphRAG v1

Generated: 2026-11-24  
Branch: feat/fr04-s106b-certificate

---

## 1. How retrievers read the `case_law` collection

### Retriever.py (case_law)
Collection: `case_law` from `backend/CaseLawDB`.  
Metadata keys consumed at retrieval time:
| Key | Usage |
|-----|-------|
| `file_name` | Displayed in result card |
| `parties` | Displayed in result card |
| `court_level` | Displayed in result card |
| `judge` | Displayed in result card |
| `year` | Displayed in result card |
| `legal_domain` | Domain-boost scoring (`calculate_domain_boost`) |
| `winning_party` | Displayed in result card |
| `firac_component` | FIRAC component scoring (`calculate_chunk_score`) |
| `chunk_index` | Used to sort and stitch chunks in `_stitch_chunks` |
| `case_identifier` | Parent-document grouping key |
| `prev_chunk_id` | Adjacent-chunk fetching |
| `next_chunk_id` | Adjacent-chunk fetching |

BM25 index is built from the full collection on first call (`_ensure_cases_bm25`), using all document strings. No metadata filter is applied to `case_law`; all records are loaded into BM25.

ID format: existing IDs are `<case_identifier>_<firac_component>_<chunk_index>` (e.g. `Ogembo_2024_KEHC15763_facts_0`).

**New `case_law_v2` IDs** will be `para_key` format: `KEHC-2024-15763#p44`.  
The BM25 build is ID-agnostic (works from documents), so the ID change is safe.

### RulesRetriever.py (statutes / master_rules)
Does **not** touch `case_law` at all. Safe.

### Qclassifier.py
Calls `Retriever.retrieve_relevant_cases` with the classifier JSON. No direct collection access. Safe.

**Breaking-change risk for `case_law_v2`:**  
The retriever reads `firac_component`, `chunk_index`, `case_identifier`, `prev_chunk_id`, `next_chunk_id` to group and stitch chunks. The new collection is one paragraph per record (no stitching needed). We must keep those keys in `case_law_v2` metadata with sensible mappings to avoid KeyError fallbacks:
- `firac_component` → mapped from `firac` field (paragraph role from LLM call 1)
- `chunk_index` → `para_no`
- `case_identifier` → `case_key`
- `prev_chunk_id` → `{case_key}#p{para_no-1}` (or empty for p0)
- `next_chunk_id` → `{case_key}#p{para_no+1}` (or empty for last)
- `file_name`, `parties`, `court_level`, `judge`, `year`, `legal_domain`, `winning_party` → mapped from new header fields

The retriever switch from `case_law` → `case_law_v2` happens only after A5 regression check passes.

---

## 2. Current case-law ingestion

Script: `backend/CaseLaw/ingester.py`  
Collection: `legal_summaries` in `backend/CaseLaw/LegalSummariesDB` (old, pre-config-migration)  
Current live collection: `case_law` in `backend/CaseLawDB` (from `backend/build_case_chunks.py`)  

Chunking: FIRAC-component chunks, overlap-stitched. Embedding prefix: `passage:` (confirmed in `build_case_chunks.py`). Model: `intfloat/e5-base-v2`.

The new pipeline (`backend/ingest/caselaw/`) replaces this with paragraph-level chunking while keeping the same embedding prefix and model.

---

## 3. graph/ setup and Neo4j connectivity

`backend/graph/` exists with:
- `admin.py`, `client.py`, `config.py`, `health.py`, `keys.py`, `models.py`, `repository.py`
- `migrations/`: `0001_schema_v1.cypher` (constraints + indexes)
- `README.md`, `prompts/`

**Migration 0001 status:** Cannot verify — Neo4j is not reachable.  
**Neo4j status:** `GRAPH_ENABLED=false` (default). Running `python -m graph.admin status` raises `RuntimeError: Graph layer is disabled`.  
**Docker:** `docker compose` plugin not installed; `docker run` available. Neo4j container `legalai-neo4j` has not been started.  

**Action required before graph writes:** Start Neo4j (`docker run neo4j:5.26-community` with the credentials in `infra/graph/.env`), set `GRAPH_ENABLED=true NEO4J_PASSWORD=legalai_dev_2026` in `.env`, then run `python -m graph.admin init`.

All ingest code that only writes to Chroma (`case_law_v2`) works without Neo4j. The `--no-llm` Chroma-only path and the full A5 regression check can run offline. Neo4j writes are skipped when `GRAPH_ENABLED=false`.

---

## 4. Black's Law pipeline status

`grep -ri "black_law\|blacklaw\|BlackLaw" --include="*.py" --include="*.yaml" --include="*.json"` returns:
- `backend/Const_IntepretationLaws/testBlackLaw.py` — test script, not imported by any pipeline
- `backend/Const_IntepretationLaws/reader.py` — standalone reader, not imported
- `backend/export_blacklaw_to_json.py` — one-shot export script, not imported
- `backend/Const_IntepretationLaws/embedding_test_output.json` — static artefact

None of these are imported by `app.py`, `Retriever.py`, `RulesRetriever.py`, or `Qclassifier.py`. The Black's Law PDF (`BlackLaw7thEdition.pdf`) is present in `Const_IntepretationLaws/` but not indexed in any active collection.

**Verdict:** The active pipeline is clean. The residual files are dead code. They are **not** deleted here — a separate cleanup commit should remove them once confirmed unused, so this prompt's scope stays focused.

---

## 5. Instruction-by-instruction review and adjustments

### A1 — Ligature repair
**Logical and optimal.** PyMuPDF available (2.1 installed). The `/usr/share/dict/words` wordlist is present on this Kali system. No conflicts.

### A2 — Header parsing
**Logical.** `graph.keys.case_key` exists and handles all citation formats. `overrides.yaml` is the right hook for Ndeti (DOCX, no neutral citation in header).  
**Adjustment:** The existing `keys.case_key` function uses `_CITATION_RE` which requires `[YYYY] COURT NUMBER` format. For SCC cases (`2026SCC241(KLR)`) the year is a prefix, not bracketed. Added `SCC` court prefix detection to `extract.py`.

### A3 — Paragraph segmentation
**Logical.** Quote-span detection is best-effort regex; will log all spans found.  
**Adjustment:** `opinion_kind` default is `unknown` (not `lead`) for Safaricom/Express where the relationship between opinions can't be determined from text alone.

### A4 — Re-index Chroma
**Logical.** Building `case_law_v2` as a new collection in `backend/CaseLawDB_v2/` preserves rollback.  
**Adjustment:** `CaseLawDB_v2` path is derived from `config.py` with a `_v2` suffix rather than a separate config key — the retriever config switch (renaming the constant in `config.py`) happens after A5 passes.

### A5 — Regression check
**Logical.** The existing retriever (`Retriever.retrieve_relevant_cases`) is callable directly. The eval script computes recall@5 and MRR against both collections.  
**Adjustment:** `scc_caselaw_seed_graph.json` does not yet exist in the repository. B7 seed-graph validation will log "seed graph not found, skipping" and generate a skeleton template instead of failing. The 20 eval questions (A5) are independent of the seed graph.

### B1 — Migration 0002
**Logical.** Migration 0001 already has `opinion_key`, `paragraph_key` constraints. Migration 0002 adds the remaining node types.  
**No conflict.**

### B2 — Edge types
**Logical.** The `RelType` enum in `models.py` is extended. Existing edges (`CITES`, `HAS_PARAGRAPH`, etc.) are preserved.

### B3 — Deterministic extraction
**Logical.** `rapidfuzz` installed (3.14.6). Entity resolution uses `token_set_ratio ≥ 90`.

### B4 — LLM extraction
**Logical.** Groq client exists (`backend/groq_client.py`). Cache is keyed `doc_hash + prompt_version` stored as JSON in `ingest/caselaw/cache/`. Model: `qwen/qwen3.8-27b` at T=0.

### B6 — Writing to Neo4j
**Logical.** The existing `repository.Neo4jRepository.upsert_node` and `upsert_edge` are used.  
**Adjustment:** `upsert_edge` currently MATCHes on `case_key`; a more general version matching any unique key property is added in `pipeline.py`.

### B7 — Seed graph validation
**Adjustment:** `scc_caselaw_seed_graph.json` is absent. The script generates a stub and reports "seed graph missing — skipping recall checks". The specific named assertions (Barisa in-degree, Ogembo→Nonny treatment, etc.) are implemented as Python assertions that also skip gracefully when the graph is empty.

### B8 — Query helpers
**Logical.** `backend/graph/queries.py` is a new file with no existing conflicts.

---

## Summary

| Check | Status |
|-------|--------|
| Retrievers — metadata key audit | Done — compat mappings defined |
| Current ingestion script | Identified (`build_case_chunks.py` → `CaseLawDB`) |
| Neo4j reachable | **BLOCKED** — container not running |
| Migration 0001 applied | Unknown (blocked by Neo4j) |
| Black's Law removed from pipeline | Yes — dead code remains, not pipeline-active |
| All instructions reviewed | Done — 7 adjustments documented above |

**Next step:** Start Neo4j container, then run `python -m graph.admin init` to apply migrations 0001 + 0002.
