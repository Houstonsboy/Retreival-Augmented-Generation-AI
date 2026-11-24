#!/usr/bin/env python3
"""
Export Black's Law Dictionary from ChromaDB to JSON for the Neo4j knowledge graph.

ChromaDB is Python-only; Neo4j graph building is done in Java (neo4j-java).
This script runs in the backend env and writes a JSON file that DictionaryKnowledgeGraph
reads to create/update the knowledge graph.

Usage:
  From repo root:  python backend/export_blacklaw_to_json.py [--output path]
  From backend:    python export_blacklaw_to_json.py [--output path]

Output JSON format (expected by neo4j-java DictionaryKnowledgeGraph):
  { "entries": [ { "term", "definition", "page", "metadata": { "see": [], "cf": [], "also_termed": [] } } ] }
"""

import argparse
import json
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings


# ChromaDB lives next to reader.py
BACKEND_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BACKEND_DIR / "Const_IntepretationLaws" / "BlackLawDictionaryDB"
CHROMA_COLLECTION_NAME = "black_law_dictionary"

DEFAULT_OUTPUT = BACKEND_DIR / "Const_IntepretationLaws" / "blacklaw_export_for_neo4j.json"


def _extract_definition_from_document(doc: str) -> str:
    """Extract definition text from ChromaDB document (searchable text format)."""
    if not doc or "Definition:" not in doc:
        return ""
    # Format: "Term: ...\nDefinition: ...\nSee: ..." etc.
    match = re.search(r"Definition:\s*(.+?)(?=\n(?:See:|Compare:|Also termed:)|$)", doc, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _split_refs(s: str) -> list[str]:
    """Split comma-separated reference string into list of trimmed strings."""
    if not s or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def export_chromadb_to_json(
    chroma_path: Path,
    collection_name: str,
    output_path: Path,
) -> int:
    """
    Read all entries from ChromaDB and write JSON for Neo4j DictionaryKnowledgeGraph.
    Returns number of entries exported.
    """
    if not chroma_path.exists():
        raise FileNotFoundError(f"ChromaDB path not found: {chroma_path}")

    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name=collection_name)
    total = collection.count()
    if total == 0:
        entries_out = []
        with open(output_path, "w") as f:
            json.dump({"entries": entries_out}, f, indent=2)
        print(f"Collection '{collection_name}' is empty. Wrote empty entries to {output_path}")
        return 0

    # Fetch in batches to avoid memory issues
    batch_size = 500
    entries_out = []
    offset = 0
    while offset < total:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = result["ids"]
        documents = result["documents"]
        metadatas = result["metadatas"]
        for i in range(len(ids)):
            meta = metadatas[i] or {}
            doc = documents[i] if documents else ""
            term = meta.get("term") or ""
            page = meta.get("page")
            if page is None:
                page = 0
            definition = _extract_definition_from_document(doc)
            if not definition and doc:
                definition = doc[:2000]  # fallback: first 2000 chars
            see = _split_refs(meta.get("see") or "")
            cf = _split_refs(meta.get("cf") or "")
            also_termed = _split_refs(meta.get("also_termed") or "")
            entries_out.append({
                "term": term,
                "definition": definition,
                "page": int(page),
                "metadata": {
                    "see": see,
                    "cf": cf,
                    "also_termed": also_termed,
                },
            })
        offset += len(ids)
        print(f"  Exported {min(offset, total)}/{total} entries...")

    with open(output_path, "w") as f:
        json.dump({"entries": entries_out}, f, indent=2)

    print(f"Wrote {len(entries_out)} entries to {output_path}")
    return len(entries_out)


def main():
    parser = argparse.ArgumentParser(
        description="Export Black's Law Dictionary from ChromaDB to JSON for Neo4j knowledge graph."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=CHROMA_DB_DIR,
        help=f"ChromaDB directory (default: {CHROMA_DB_DIR})",
    )
    args = parser.parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Black's Law Dictionary → JSON export for Neo4j knowledge graph")
    print(f"ChromaDB: {args.chroma_dir}")
    print(f"Collection: {CHROMA_COLLECTION_NAME}")
    print(f"Output: {output_path}\n")

    try:
        n = export_chromadb_to_json(
            chroma_path=args.chroma_dir,
            collection_name=CHROMA_COLLECTION_NAME,
            output_path=output_path,
        )
        print(f"\nDone. Run the Neo4j graph builder with this file:")
        print(f"  cd neo4j-java && mvn compile exec:java -Dexec.args='{output_path}'")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
