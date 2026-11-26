# GraphRAG Fast-Path Implementation Plan

## Overview

This document records the plan, deviations, and rollback instructions for the
GraphRAG fast-path integration (`feat/graph-fastpath` branch).

The goal is a working, measurable graph layer in the advocate pipeline using the
hand-checked seed graph, without breaking the existing retrieval path.

**Default stays OFF.** `GRAPH_ENABLED=false` (the default) produces byte-identical
answers to the stable baseline.

---

## Stable baseline

- **Tag**: `stable-pre-graph` → commit `d555021`
- **Branch for new work**: `feat/graph-fastpath` (started from `1e52760`)
- **Tests**: 374 passed, 4 skipped
- **Frontend build**: exit 0
- **Baseline queries**: `eval/baseline/` — 5 queries, chunk IDs recorded
- **Data backup**: `~/backups/legalai-20261124-141151/` (SHA-256 in README.md there)

---

## Rollback instructions

### Code rollback

```bash
git switch feat/fr04-s106b-certificate   # return to stable branch
# or
git checkout stable-pre-graph            # detach at the exact stable tag
```

### Data rollback

```bash
BACKUP=~/backups/legalai-20261124-141151
cp -r $BACKUP/CaseLawDB          backend/
cp -r $BACKUP/ChunkDB             backend/
cp -r $BACKUP/MasterRulesDB       backend/constitutionlogic/
cp -r $BACKUP/Const_IntepretationLaws  backend/
cp    $BACKUP/locations.db        backend/
cp -r $BACKUP/logs                backend/
```

### Graph rollback

```bash
make graph-down       # stops Neo4j; app works normally with graph off or Neo4j stopped
```

Verification: after switching to `feat/fr04-s106b-certificate`, run:

```bash
cd backend && PYTHONPATH=. venv/bin/pytest tests/ -q
```

Expect: 374 passed, 4 skipped.

---

## Phase plan

### Step 0: Verify and commit seed graph

- File: `backend/graph/seed/scc_caselaw_graph.json`
- 149 nodes, 269 edges, version `seed-0.1`
- SHA-256: `9006fea588d480a871113681aee98d280defd89f89d814769fdf380aebcd9a65`
  (differs from reference `48d0c5ef...` due to formatting only — content checks pass)
- Added `backend/graph/seed/README.md`
- **Commit**: `de9b81a chore(graph): add hand-checked seed graph v0.1 and baseline eval`

### Phase 0: SentenceTransformer singleton (D-08)

**Hook points**: `Retriever.py:601`, `RulesRetriever.py:498`, `pixe.py:97`, `digester.py:351`

Create `backend/embeddings.py` with a module-level `get_embedder()` that returns
a lazily-loaded `SentenceTransformer` singleton. All four files import and call it
instead of constructing the model inline.

Regression gate: re-run baseline queries after the change; chunk IDs must be identical.

### Phase 1: Start Neo4j

- Create `infra/graph/.env` from `.env.example` (gitignored)
- Run `make graph-up`, `python -m graph.admin init`, `python -m graph.admin status`
- Run 6 skipped graph integration tests against `make graph-test-up` (port 7688)

### Phase 2: Load seed graph

- Create `backend/graph/seed_loader.py`
  - CLI: `python -m graph.seed_loader load [--dry-run] [--file PATH]`
  - Maps seed node types to graph labels
  - Builds canonical keys via `graph/keys.py`
  - Links cases to Chroma `case_identifier` values
  - Uses `backend/graph/seed/case_identifier_overrides.yaml` for edge cases
  - Fully idempotent (MERGE throughout)
- Validation after load:
  - 149 seed nodes + existing reference nodes from 0002
  - 269 seed relationships
  - Barisa has ≥3 incoming CITES
  - Ogembo CITES Nonny DISTINGUISHED para:44
  - `pr:scc-not-strict-evidence` QUALIFIES `pr:cert-mandatory`

### Phase 3: Query-time expansion

Add `expand_for_answer(case_identifiers, query_text, forum, cap)` to `backend/graph/queries.py`.

Returns `[]` when `GRAPH_ENABLED=false`.

When enabled:
1. Anchor cases via `chroma_case_identifier`
2. Full-text issue search
3. Collect principles from HOLDS/STATES_OBITER + ANSWERS issues
4. Always pull QUALIFIES/CONFLICTS_WITH neighbours (counter-authority)
5. Attach supporting chunks from Chroma `case_law`
6. Rank: QUALIFIES/CONFLICTS first, ratio before obiter, binding before non-binding
7. Cap at `cap` (default 5)

**Hook point**: After `retrieve_relevant_cases` returns, before generation, only
when `_graph_enabled()`. Not injected into `cand_ids`.

**Unit tests** (require graph test container with seed loaded):
- Ogembo + cert query → returns `pr:cert-mandatory` + QUALIFIES note
- Ndeti + default judgment → `pr:regular-default-discretion` from two cases
- Nonny + late cert → both `pr:cert-late-allowed` and `pr:cert-late-not-after-trial` CONFLICTS

### Phase 4: Wire into advocate flow

**Flag-guarded everywhere.** No change to `cand_ids` or ranking.

1. `/api/qretrieve` → call `expand_for_answer()` after case retrieval
2. `/api/generate-legal-answer`:
   - Add G-alias loop after `app.py:1540`
   - Render `[G1]` and `[G1.1]` blocks in user message
   - Add G-block grounding rules to system prompt
   - Extend `_alias_pat` regex to `[CSG]`
3. `citation_verifier.py`:
   - Add `graph_authorities` to `_valid_refs()`
   - Add `_GRAPH_RE` to `extract_citations()`
4. Frontend: show Graph authority card in sources panel

### Phase 5: On/off evaluation

12 questions in `eval/caselaw_questions.yaml`.
Script: `eval/run_graph_eval.py` → `eval/results/graph_eval_summary.md`.
Metrics: recall@3, qualification captured, flagged÷total citations, latency.

### Phase 6: Housekeeping

- Add `.gitignore` rules for ChromaDB directories and `backend/logs/*.jsonl`
- Update `backend/graph/README.md` with "Fast path (seed graph)" section
- List dead artefacts to archive later (do not delete in this task):
  - `backend/Const_IntepretationLaws/BlackLawDictionaryDB/` (black_law_dictionary)
  - `backend/LegalSummariesDB/` (legal_summaries)

---

## Deviations from plan

| Phase | Deviation | Reason |
|-------|-----------|--------|
| Step 0 | SHA-256 differs from reference (`9006fea…` vs `48d0c5ef…`) | Formatting only; all content checks pass |
| Step 0 | Tagged `d555021` as `stable-pre-graph`; branch started from `1e52760` | `1e52760` is "stable snapshot before Neo4j / GraphRAG integration" on the same branch — both represent the stable state |

---

## Dead artefacts (to archive later, NOT deleted in this task)

- `backend/Const_IntepretationLaws/BlackLawDictionaryDB/` — black_law_dictionary layer, unused in live pipeline
- `backend/LegalSummariesDB/` and `backend/LegalSummariesDB.bak/` — legacy legal summaries DB, superseded by ChunkDB
