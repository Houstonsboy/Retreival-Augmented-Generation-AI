# backend/graph — Python-native Neo4j layer

**Status: schema installed, ingestion not yet wired.**
**Default: `GRAPH_ENABLED=false` — the RAG pipeline is unchanged.**

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
