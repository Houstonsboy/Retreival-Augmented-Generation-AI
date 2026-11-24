# backend/graph — Python-native Neo4j layer

**Status: seed graph loaded, fast-path wired into advocate flow.**
**Default: `GRAPH_ENABLED=false` — the RAG pipeline is unchanged without the flag.**

## Quick start

```bash
# 1. Copy and edit secrets
cp infra/graph/.env.example infra/graph/.env   # set NEO4J_PASSWORD

# 2. Start Neo4j (requires Docker + docker compose)
make graph-up

# 3. Apply schema (from backend/)
GRAPH_ENABLED=true python -m graph.admin init

# 4. Verify
curl http://localhost:5000/graph/health
```

## Package layout

| File | Purpose |
|---|---|
| `config.py` | Env-var settings; `GRAPH_ENABLED` gate |
| `client.py` | Driver singleton, `read()` / `write()`, health |
| `migrations/runner.py` | Applies numbered `.cypher` files once; checksums guard edits |
| `migrations/0001_schema_v1.cypher` | Constraints, indexes, full-text, vector index |
| `models.py` | Pydantic node/edge models; `RelType` enum |
| `keys.py` | Deterministic canonical key builders |
| `repository.py` | Abstract + Neo4j impl; `upsert_node`, `upsert_edge`, `neighbors` |
| `admin.py` | CLI: `init` / `status` / `migrate` / `reset --confirm-wipe` |
| `health.py` | Flask blueprint: `GET /graph/health` |
| `seed_loader.py` | CLI: `python -m graph.seed_loader load [--dry-run] [--file PATH]` |
| `queries.py` | Query helpers: `expand_for_answer()`, `doctrinal()`, `from_paragraphs()` |
| `authority.py` | `is_binding(court_code, forum)` via transitive BINDS closure |
| `seed/scc_caselaw_graph.json` | Seed graph: 149 nodes, 269 edges (seed-0.1) |
| `seed/case_identifier_overrides.yaml` | Manual Chroma ID overrides (empty for seed-0.1) |

## Conventions

**Canonical keys** (`keys.py`)
- `case_key("[2024] KEHC 15763 (KLR)") → "KEHC-2024-15763"`
- `provision_key("Evidence Act", "s.106B(4)(d)") → "evidence-act/s106B/4/d"`
- `para_key("KEHC-2024-15763", 44) → "KEHC-2024-15763#p44"`

**`aliases_text`** — every `Case` node has `aliases: list[str]` for application code and
`aliases_text: str` (space-joined) for the full-text index `case_names_ft`. The repository
upsert keeps them in sync.

**Migrations** — never edit an applied migration; write `0002_*.cypher` instead. The runner
refuses if a file's checksum has changed.

**Security** — `client.py` uses only parameterised Cypher (`$param`). Never format user
input into query strings.

## Fast path (seed graph)

The fast-path is active when `GRAPH_ENABLED=true`. It adds graph-confirmed legal
principles to every `/api/qretrieve` response under `retrieval.graph_authorities`.

### How it works

1. `/api/qretrieve` calls `expand_for_answer(case_identifiers, query_text, forum="SCC", cap=5)`
   after case-law retrieval.
2. `expand_for_answer` anchors on the retrieved `chroma_case_identifier` values, follows
   `HOLDS` / `STATES_OBITER` edges to `Principle` nodes, pulls `QUALIFIES` / `CONFLICTS_WITH`
   relations, and ranks by authority (binding first, ratio > obiter).
3. `/api/generate-legal-answer` injects the results as G-alias source blocks (`[G1]` / `[G1.1]`)
   alongside case (`[C#]`) and statute (`[S#]`) sources.
4. The LLM cites `[G1.1]` for the principle text; the citation verifier validates G-aliases.

### Load the seed graph

```bash
# From backend/
GRAPH_ENABLED=true NEO4J_PASSWORD=<password> python -m graph.seed_loader load [--dry-run]
```

Outputs `docs/graph/seed_load_report.md` with node counts and Chroma mappings.

### On/off evaluation

```bash
cd backend
GRAPH_ENABLED=true NEO4J_PASSWORD=<password> venv/bin/python ../eval/run_graph_eval.py
```

Results: `eval/results/graph_eval_summary.md`

## Roadmap (do not build yet)

1. **Case-law ingestion** — text repair, paragraph segmentation, regex citation/provision
   extraction, entity resolution, 2 LLM passes, validation against seed graph.

2. **Statutes layer** — `Statute`/`Provision` nodes from the `statutes` Chroma collection;
   `PART_OF` hierarchy; `INTERPRETS` links to cases.

3. **Master rules / Constitution layer.**

4. **Concept layer** — port the Black's dictionary (`Term` + `SEE`/`CF`/`ALSO_TERMED`) with
   a Python loader from `Const_IntepretationLaws/blacklaw_export_for_neo4j.json`. Build
   this after case/statute layers so term expansion is anchored to Kenyan sources.
   See `legacy/neo4j-java/LEGACY_README.md` for the original Java approach.

5. **Query-time integration** — a `GraphRetriever` alongside `Retriever.py`, behind
   `GRAPH_ENABLED`, with A/B evaluation before switching over.

**Open architecture question:** whether paragraph embeddings live in Neo4j (using the
vector index in `0001_schema_v1.cypher`) or stay in ChromaDB (`ChunkDB`), or both.
Decide before the case-law ingestion prompt.
