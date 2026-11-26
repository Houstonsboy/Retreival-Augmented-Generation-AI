"""A5 — Retrieval regression check.

Runs the existing hybrid retriever against case_law and case_law_v2,
computes Recall@5 and MRR per question, and decides whether to switch.

Usage:
    cd backend && python ../scripts/eval_caselaw_retrieval.py

Output:
    eval/retrieval_report.json
    Console table with per-question and aggregate metrics.

Switch condition (from spec):
    v2 overall recall ≥ v1 AND no question drops from a hit to a miss.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_backend))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_backend / ".env", override=True)

import yaml
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "eval" / "caselaw_questions.yaml"
_REPORT_PATH = Path(__file__).resolve().parent.parent / "eval" / "retrieval_report.json"
_EMBED_MODEL = "intfloat/e5-base-v2"
_EMBED_PREFIX = "query: "
_TOP_K = 5


def load_questions() -> list[dict]:
    with _QUESTIONS_PATH.open() as f:
        return yaml.safe_load(f)["questions"]


def _hit_in_range(meta: dict, expected_hits: list[dict]) -> bool:
    case_key = meta.get("case_key", "")
    para_no = int(meta.get("para_no", 0))
    for exp in expected_hits:
        if exp["case_key"] != case_key:
            continue
        lo, hi = exp.get("para_range", [0, 999])
        if lo <= para_no <= hi:
            return True
    return False


def evaluate_collection(
    db_dir: Path,
    collection_name: str,
    questions: list[dict],
    embedder: SentenceTransformer,
) -> list[dict]:
    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        col = client.get_collection(collection_name)
    except Exception as exc:
        print(f"  Collection '{collection_name}' not found in {db_dir}: {exc}")
        return []

    all_data = col.get(include=["documents", "metadatas"])
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]
    all_ids = all_data["ids"]

    bm25 = BM25Okapi([d.split() for d in all_docs])

    results = []
    for q in questions:
        query = q["question"]
        expected = q["expected_hits"]

        # Vector search
        q_emb = embedder.encode([_EMBED_PREFIX + query], show_progress_bar=False).tolist()
        vec_res = col.query(
            query_embeddings=q_emb,
            n_results=min(_TOP_K * 2, col.count()),
            include=["metadatas", "distances"],
        )
        vec_metas = vec_res["metadatas"][0]
        vec_dists = vec_res["distances"][0]

        # BM25 search
        bm25_scores = bm25.get_scores(query.split())
        top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:_TOP_K * 2]

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        for rank, meta in enumerate(vec_metas):
            pk = meta.get("case_key", "") + "#" + str(meta.get("para_no", ""))
            rrf_scores[pk] = rrf_scores.get(pk, 0) + 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(top_bm25_idx):
            pk = all_metas[idx].get("case_key", "") + "#" + str(all_metas[idx].get("para_no", ""))
            rrf_scores[pk] = rrf_scores.get(pk, 0) + 1.0 / (60 + rank + 1)

        # Build top-K by RRF score with metadata
        pk_to_meta: dict[str, dict] = {}
        for meta in vec_metas:
            pk = meta.get("case_key", "") + "#" + str(meta.get("para_no", ""))
            pk_to_meta[pk] = meta
        for idx in top_bm25_idx:
            m = all_metas[idx]
            pk = m.get("case_key", "") + "#" + str(m.get("para_no", ""))
            pk_to_meta[pk] = m

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:_TOP_K]

        # Compute metrics
        hit = False
        rr = 0.0
        for rank, (pk, _) in enumerate(ranked):
            meta = pk_to_meta.get(pk, {})
            if _hit_in_range(meta, expected):
                if not hit:
                    hit = True
                    rr = 1.0 / (rank + 1)
                break

        results.append({
            "question": query,
            "hit_at_5": hit,
            "rr": rr,
            "top5_case_keys": [pk_to_meta.get(pk, {}).get("case_key", pk) for pk, _ in ranked],
        })

    return results


def print_table(label: str, results: list[dict]) -> None:
    hits = sum(1 for r in results if r["hit_at_5"])
    mrr = sum(r["rr"] for r in results) / len(results) if results else 0
    print(f"\n{'='*60}")
    print(f"Collection: {label}  |  Recall@5: {hits}/{len(results)} = {hits/len(results):.2%}  |  MRR: {mrr:.3f}")
    print(f"{'='*60}")
    for r in results:
        icon = "✓" if r["hit_at_5"] else "✗"
        print(f"  {icon} [{r['rr']:.2f}] {r['question'][:65]}")
        print(f"      top5: {r['top5_case_keys']}")


def main() -> None:
    questions = load_questions()
    embedder = SentenceTransformer(_EMBED_MODEL)

    from config import CASELAW_DB_DIR, CASELAW_COLLECTION_NAME
    db_v2_dir = Path(str(CASELAW_DB_DIR) + "_v2")

    print(f"Evaluating {len(questions)} questions against two collections …")

    v1_results = evaluate_collection(CASELAW_DB_DIR, CASELAW_COLLECTION_NAME, questions, embedder)
    v2_results = evaluate_collection(db_v2_dir, "case_law_v2", questions, embedder)

    if not v1_results:
        print("WARNING: v1 (case_law) is empty or not found — skipping comparison.")
    if not v2_results:
        print("WARNING: v2 (case_law_v2) is empty or not found — cannot switch.")
        return

    print_table("case_law (v1)", v1_results)
    print_table("case_law_v2", v2_results)

    # Switch decision
    v1_hits = sum(1 for r in v1_results if r["hit_at_5"])
    v2_hits = sum(1 for r in v2_results if r["hit_at_5"])
    regressions = []
    for v1r, v2r in zip(v1_results, v2_results):
        if v1r["hit_at_5"] and not v2r["hit_at_5"]:
            regressions.append(v2r["question"])

    should_switch = (v2_hits >= v1_hits) and len(regressions) == 0
    print(f"\n{'='*60}")
    if should_switch:
        print("✓ SWITCH RECOMMENDED: v2 ≥ v1 and no regressions.")
        print("  Run: python scripts/eval_caselaw_retrieval.py --switch")
    else:
        print("✗ SWITCH NOT recommended.")
        if regressions:
            print(f"  REGRESSIONS ({len(regressions)}):")
            for r in regressions:
                print(f"    - {r}")

    # Save report
    report = {
        "v1": {"hits": v1_hits, "total": len(v1_results),
               "mrr": sum(r["rr"] for r in v1_results) / max(1, len(v1_results)),
               "per_question": v1_results},
        "v2": {"hits": v2_hits, "total": len(v2_results),
               "mrr": sum(r["rr"] for r in v2_results) / max(1, len(v2_results)),
               "per_question": v2_results},
        "regressions": regressions,
        "should_switch": should_switch,
    }
    _REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved → {_REPORT_PATH}")

    # Switch flag
    if "--switch" in sys.argv and should_switch:
        from ingest.caselaw.chroma_index import switch_collection
        switch_collection(_backend / "config.py", "case_law", "case_law_v2")
        print("Collection switched in config.py.")


if __name__ == "__main__":
    main()
