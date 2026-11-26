#!/usr/bin/env python3
"""
Step 5 — Embed statute and case chunks into ChromaDB.

Resumable: skip unchanged content/model pairs and update changed records.

Usage:
    # Embed statutes only
    python embed_knowledge.py --source statutes

    # Embed cases only
    python embed_knowledge.py --source cases

    # Embed both (default)
    python embed_knowledge.py --source both

    # Smoke-test: embed first 20 statute chunks into a temporary collection
    python embed_knowledge.py --source statutes --limit 20 --suffix _test

    # Dry-run: print plan without embedding
    python embed_knowledge.py --source both --dry-run
"""

import argparse
from hashlib import sha256
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    STATUTES_DB_DIR,
    STATUTES_COLLECTION_NAME,
    CASELAW_DB_DIR,
    CASELAW_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)

DATA_DIR = BASE_DIR.parent / "data"
STATUTES_JSONL = DATA_DIR / "processed" / "statutes" / "sections_embed.jsonl"
CASES_JSONL = DATA_DIR / "processed" / "cases" / "cases_embed.jsonl"
LOG_DIR = BASE_DIR / "logs"

EMBED_BATCH = 32

# Fields that must NOT be stored as metadata (stored elsewhere or not serialisable)
_SKIP_META = {"text", "chunk_id"}


def load_jsonl(path: Path):
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def to_chroma_meta(chunk: dict) -> dict:
    """
    Return a dict of metadata safe for ChromaDB: values must be
    str, int, float, or bool. Lists and None are converted.
    """
    out = {}
    for k, v in chunk.items():
        if k in _SKIP_META:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif v is None:
            out[k] = ""
        elif isinstance(v, list):
            out[k] = ",".join(str(x) for x in v)
        else:
            out[k] = str(v)
    return out


def embed_chunks(chunks, collection, model, source_label: str, limit=None, verbose=True) -> int:
    """
    Embed chunks into collection; skip IDs already present (resumable).
    Returns number of newly embedded chunks.
    """
    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk ID in staging data")
    stored = collection.get(include=["metadatas"])
    existing = dict(zip(stored["ids"], stored["metadatas"]))
    prepared = []
    for chunk in chunks:
        meta = to_chroma_meta(chunk)
        meta["embedding_hash"] = sha256(
            f"{EMBEDDING_MODEL_NAME}\0passage: {chunk['text']}".encode()
        ).hexdigest()
        old = existing.get(chunk["chunk_id"]) or {}
        if old.get("embedding_hash") != meta["embedding_hash"]:
            prepared.append((chunk, meta))
        elif old != meta:
            collection.update(ids=[chunk["chunk_id"]], metadatas=[meta])
    todo = prepared
    if limit:
        todo = todo[:limit]

    if not todo:
        print(f"  Nothing new to embed ({len(chunks)} chunks already present)")
        return 0

    skipped = len(chunks) - len(todo)
    if limit:
        print(f"  Embedding {len(todo)} chunks (limit={limit}, {skipped} already present)")
    else:
        print(f"  Embedding {len(todo)} chunks ({skipped} already present, resuming if any)")

    embedded = 0
    for i in range(0, len(todo), EMBED_BATCH):
        batch = todo[i : i + EMBED_BATCH]
        texts = ["passage: " + c["text"] for c, _ in batch]
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=EMBED_BATCH)

        collection.upsert(
            ids=[c["chunk_id"] for c, _ in batch],
            embeddings=embeddings.tolist(),
            documents=[c["text"] for c, _ in batch],
            metadatas=[meta for _, meta in batch],
        )
        embedded += len(batch)
        if verbose:
            print(f"    {embedded}/{len(todo)} embedded", end="\r", flush=True)

    print(f"\n  Done: {embedded} new chunks embedded into '{collection.name}'")
    return embedded


def main():
    parser = argparse.ArgumentParser(description="Embed statute/case knowledge into ChromaDB")
    parser.add_argument("--source", choices=["statutes", "cases", "both"], default="both",
                        help="Which knowledge source to embed")
    parser.add_argument("--limit", type=int, default=None,
                        help="Embed at most N new chunks (useful for smoke tests)")
    parser.add_argument("--suffix", default="",
                        help="Append suffix to collection name, e.g. _test for a temp collection")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load JSONL and print plan without embedding anything")
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--cases-jsonl", type=Path, default=CASES_JSONL)
    parser.add_argument("--statutes-jsonl", type=Path, default=STATUTES_JSONL)
    parser.add_argument("--sync", action="store_true", help="Remove IDs absent from a complete staging collection")
    args = parser.parse_args()
    if args.sync and (args.limit is not None or not args.db_root):
        parser.error("--sync requires an explicit --db-root and a complete run")


    sources = []
    if args.source in ("statutes", "both"):
        sources.append({
            "label": "statutes",
            "jsonl": args.statutes_jsonl,
            "db_dir": STATUTES_DB_DIR,
            "collection": STATUTES_COLLECTION_NAME + args.suffix,
        })
    if args.source in ("cases", "both"):
        sources.append({
            "label": "cases",
            "jsonl": args.cases_jsonl,
            "db_dir": CASELAW_DB_DIR,
            "collection": CASELAW_COLLECTION_NAME + args.suffix,
        })

    if not args.dry_run:
        import chromadb
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    else:
        model = None

    log_dir = args.db_root if args.db_root else LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"embed_{int(time.time())}.log"
    total_embedded = 0

    for src in sources:
        if args.db_root:
            src["db_dir"] = args.db_root / src["db_dir"].name
        print(f"\n{'='*64}")
        print(f"Source   : {src['label']}")
        print(f"JSONL    : {src['jsonl']}")
        print(f"DB dir   : {src['db_dir']}")
        print(f"Collection: {src['collection']}")
        print(f"{'='*64}")

        if not src["jsonl"].exists():
            script = "build_statute_chunks.py" if src["label"] == "statutes" else "build_case_chunks.py"
            print(f"ERROR: {src['jsonl']} not found. Run {script} first.", file=sys.stderr)
            sys.exit(1)

        chunks = load_jsonl(src["jsonl"])
        print(f"Loaded {len(chunks)} chunks from {src['jsonl'].name}")

        if args.dry_run:
            limit_note = f" (limit {args.limit})" if args.limit else ""
            print(f"[dry-run] Would embed up to {len(chunks)} chunks{limit_note} → '{src['collection']}'")
            continue

        src["db_dir"].mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(src["db_dir"]))
        collection = client.get_or_create_collection(
            name=src["collection"],
            metadata={"hnsw:space": "cosine"},
        )

        n = embed_chunks(chunks, collection, model, src["label"], limit=args.limit)
        total_embedded += n
        if args.sync:
            stale = set(collection.get(include=[])["ids"]) - {c["chunk_id"] for c in chunks}
            if stale:
                collection.delete(ids=sorted(stale))

        with open(log_path, "a") as lf:
            lf.write(json.dumps({
                "source": src["label"],
                "collection": src["collection"],
                "embedded": n,
                "ts": int(time.time()),
            }) + "\n")

    if not args.dry_run:
        print(f"\nTotal embedded this run: {total_embedded}")
        print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
