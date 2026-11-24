"""Capture baseline retrieval results (chunk IDs) for 5 advocate queries.

Usage: cd backend && PYTHONPATH=. python ../eval/capture_baseline.py
"""
import json
import sys
import os
import time
from pathlib import Path

# Backend is the working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from Retriever import retrieve_relevant_cases

QUERIES_FILE = Path(__file__).resolve().parent / "baseline_queries.json"
BASELINE_DIR = Path(__file__).resolve().parent / "baseline"
BASELINE_DIR.mkdir(exist_ok=True)

with open(QUERIES_FILE) as f:
    queries = json.load(f)

summary = []
for q in queries:
    qid = q["id"]
    print(f"\n--- {qid}: {q['query'][:60]} ---")
    t0 = time.time()
    result = retrieve_relevant_cases(q["classifier"], verbose=False, skip_summaries=True)
    latency_ms = int((time.time() - t0) * 1000)

    # Collect chunk IDs from returned results
    chunk_ids = []
    case_identifiers = []
    for r in result.get("results", []):
        case_identifiers.append(r.get("case_name", r.get("parent_document", "")))
        for ch in r.get("chunks", []):
            cid = ch.get("chunk_id", ch.get("id", ""))
            if cid:
                chunk_ids.append(cid)

    record = {
        "id": qid,
        "query": q["query"],
        "chunk_ids": chunk_ids,
        "case_identifiers": case_identifiers,
        "latency_ms": latency_ms,
        "results_count": len(result.get("results", [])),
    }
    out_path = BASELINE_DIR / f"{qid}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    summary.append(record)
    print(f"  cases: {case_identifiers}")
    print(f"  chunks: {len(chunk_ids)}, latency: {latency_ms}ms")

with open(BASELINE_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nBaseline captured to eval/baseline/")
