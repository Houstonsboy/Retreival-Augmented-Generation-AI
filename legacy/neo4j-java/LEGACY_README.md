# Legacy: Neo4j Java Graph Builder

**Status: Archived 2026-11-23. Superseded by `backend/graph/` (Python).**

This directory contains the original two-step pipeline for loading Black's Law Dictionary
entries into Neo4j:

1. `export_blacklaw_to_json.py` (backend root) — exports ChromaDB entries to JSON
2. `DictionaryKnowledgeGraph.java` — reads the JSON and creates `(:Term)` nodes with
   `SEE`, `CF`, `ALSO_TERMED` relationships

## Why it was replaced

The new `backend/graph/` package is Python-native, uses Docker-managed Neo4j, and has a
versioned migration runner. The concept layer (porting the dictionary) will be a Python
loader that reads `Const_IntepretationLaws/blacklaw_export_for_neo4j.json` directly —
no Java required.

## Kept for reference

`DictionaryKnowledgeGraph.java` documents the original Cypher patterns and the
`SEE`/`CF`/`ALSO_TERMED` relationship model. Consult it when building the dictionary
loader in Phase C4 of the graph roadmap.

Do not build or run this code. It requires a manually-started Neo4j instance and the
old `mvn exec:java` toolchain.
