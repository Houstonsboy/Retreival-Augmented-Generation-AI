"""
eval/run_retrieval_eval.py  —  Retrieval baseline evaluation harness
Branch: feat/fr04-s106b-certificate
Generated: 2026-11-17

Reads eval/retrieval_queries.csv, drives both RulesRetriever (statutes) and
Retriever (case law) for every query, then scores each result against the
expected_statute_sections and expected_cases columns.

Metric
------
Hit@K  (K = 3 for statutes, K = 3 for cases)
  A query is a "hit" if at least one expected item appears in the top-K
  results returned by the retriever.

Precision@K
  |{returned ∩ expected}| / K  (averaged across all queries that have
  expected items in that column).

Usage
-----
  cd /home/gift/Retreival-Augmented-Generation-AI
  python eval/run_retrieval_eval.py [--verbose] [--csv-out eval/logs/retrieval_results.csv]

Requirements
------------
  - backend venv must be activated (or the packages installed system-wide)
  - GROQ_API_KEY in backend/.env   (Retriever uses it; skip_summaries=True avoids LLM calls)
  - StatutesDB and CaseLawDB must exist
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — make backend importable from the eval directory
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

QUERIES_CSV = Path(__file__).resolve().parent / "retrieval_queries.csv"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

TOP_K_STATUTES = 3
TOP_K_CASES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip whitespace — used for fuzzy containment checks."""
    return text.strip().lower()


def _expected_items(cell: str) -> List[str]:
    """Split a CSV cell like 'Evidence Act Cap 80 s.106B; s.78A' into items."""
    if not cell or cell.strip() == "":
        return []
    return [i.strip() for i in cell.split(";") if i.strip()]


def _hit(returned_texts: List[str], expected_items: List[str]) -> bool:
    """
    Return True if any expected item is a substring of any returned text
    (case-insensitive).  This is intentionally lenient because section
    identifiers appear in various formats in chunk text.
    """
    for exp in expected_items:
        norm_exp = _normalise(exp)
        for text in returned_texts:
            if norm_exp in _normalise(text):
                return True
    return False


def _precision_at_k(returned_texts: List[str], expected_items: List[str], k: int) -> float:
    if not expected_items:
        return float("nan")
    hits = sum(
        1 for exp in expected_items
        if any(_normalise(exp) in _normalise(t) for t in returned_texts[:k])
    )
    return hits / k


def _build_classifier_input(query: str, expected_statutes: List[str]) -> Dict:
    """
    Build a minimal Qclassifier-shaped dict for the retrievers.
    For the eval harness we bypass the actual classifier and hand-craft
    the classifier output from the CSV expected columns.
    """
    # Infer statute names from the expected_statute_sections column
    statutes_primary = list({s.split(" s.")[0].split(" s ")[0].strip() for s in expected_statutes if s})

    return {
        "intents": ["RULE_SEARCH", "OUTCOME_ANALYSIS"],
        "target_components": ["RULES", "APPLICATION", "CONCLUSION"],
        "legal_domains": [{"domain": "Civil Law", "confidence_score": 0.7}],
        "entities": {
            "statutes": {"primary": statutes_primary, "contextual": []},
            "source_documents": statutes_primary,
            "cases": [],
            "section_numbers": [],
            "article_numbers": [],
        },
        "vector_query": query,
        "vector_query_statutory": query,
        "vector_query_caselaw": query,
        "retrieval_mode": "hybrid",
        "retrieval_strategy": "balanced",
    }


# ---------------------------------------------------------------------------
# Single-query evaluation
# ---------------------------------------------------------------------------

def eval_one_query(
    query_id: str,
    query: str,
    expected_statutes: List[str],
    expected_cases: List[str],
    verbose: bool = False,
) -> Dict:
    """
    Run one query through both retrievers, return a result dict.
    Errors are caught and stored in the result — the harness continues.
    """
    result = {
        "query_id": query_id,
        "query": query,
        "statute_hit": None,
        "statute_precision": None,
        "case_hit": None,
        "case_precision": None,
        "statute_error": None,
        "case_error": None,
        "statute_returned": [],
        "case_returned": [],
    }

    classifier_input = _build_classifier_input(query, expected_statutes)

    # ---- Statute retrieval (RulesRetriever) --------------------------------
    if expected_statutes:
        try:
            from RulesRetriever import retrieve_relevant_rules  # noqa: F401
            rules_out = retrieve_relevant_rules(
                classifier_input,
                verbose=verbose,
                max_results=TOP_K_STATUTES,
            )
            # Each result has 'full_text', 'article_header', 'document_label', 'article_number'
            statute_texts = []
            for r in rules_out.get("results", []):
                label = f"{r.get('document_label', '')} s.{r.get('article_number', '')}"
                statute_texts.append(label)
                statute_texts.append(r.get("full_text", ""))
            result["statute_returned"] = [
                f"{r.get('document_label','')} s.{r.get('article_number','')}"
                for r in rules_out.get("results", [])
            ]
            result["statute_hit"] = _hit(statute_texts, expected_statutes)
            result["statute_precision"] = _precision_at_k(statute_texts, expected_statutes, TOP_K_STATUTES)
        except Exception as exc:
            result["statute_error"] = str(exc)
            result["statute_hit"] = False
            result["statute_precision"] = 0.0

    # ---- Case-law retrieval (Retriever) ------------------------------------
    if expected_cases:
        try:
            from Retriever import retrieve_relevant_cases  # noqa: F401
            cases_out = retrieve_relevant_cases(
                classifier_input,
                verbose=verbose,
                skip_summaries=True,
                top_n=TOP_K_CASES,
            )
            case_texts = []
            for r in cases_out.get("results", []):
                case_texts.append(r.get("case_name", ""))
                case_texts.append(r.get("best_chunk", ""))
            result["case_returned"] = [r.get("case_name", "") for r in cases_out.get("results", [])]
            result["case_hit"] = _hit(case_texts, expected_cases)
            result["case_precision"] = _precision_at_k(case_texts, expected_cases, TOP_K_CASES)
        except Exception as exc:
            result["case_error"] = str(exc)
            result["case_hit"] = False
            result["case_precision"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run retrieval baseline evaluation")
    parser.add_argument("--verbose", action="store_true", help="Verbose retriever output")
    parser.add_argument(
        "--csv-out",
        default=str(LOG_DIR / "retrieval_results.csv"),
        help="Path to write per-query result CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N queries (useful for quick smoke-tests)",
    )
    args = parser.parse_args()

    if not QUERIES_CSV.exists():
        print(f"ERROR: {QUERIES_CSV} not found", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(QUERIES_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    if args.limit:
        rows = rows[: args.limit]

    all_results = []
    for i, row in enumerate(rows, 1):
        qid = row.get("query_id", f"Q{i:03d}")
        query = row.get("query", "")
        exp_statutes = _expected_items(row.get("expected_statute_sections", ""))
        exp_cases = _expected_items(row.get("expected_cases", ""))

        print(f"[{i}/{len(rows)}] {qid}: {query[:60]}...", flush=True)
        res = eval_one_query(qid, query, exp_statutes, exp_cases, verbose=args.verbose)
        all_results.append(res)

        if res["statute_error"]:
            print(f"  STATUTE ERROR: {res['statute_error']}")
        else:
            print(f"  statutes returned: {res['statute_returned']}  hit={res['statute_hit']}")

        if res["case_error"]:
            print(f"  CASE ERROR: {res['case_error']}")
        else:
            print(f"  cases returned: {res['case_returned']}  hit={res['case_hit']}")

    # ---- Aggregate metrics -------------------------------------------------
    statute_hits = [r["statute_hit"] for r in all_results if r["statute_hit"] is not None]
    case_hits = [r["case_hit"] for r in all_results if r["case_hit"] is not None]
    statute_precs = [r["statute_precision"] for r in all_results
                     if r["statute_precision"] is not None and not (r["statute_precision"] != r["statute_precision"])]
    case_precs = [r["case_precision"] for r in all_results
                  if r["case_precision"] is not None and not (r["case_precision"] != r["case_precision"])]

    def _pct(lst):
        return f"{100 * sum(lst) / len(lst):.1f}%" if lst else "N/A"

    print("\n" + "=" * 60)
    print("RETRIEVAL BASELINE RESULTS")
    print("=" * 60)
    print(f"Queries evaluated       : {len(all_results)}")
    print(f"Statute Hit@{TOP_K_STATUTES}           : {_pct(statute_hits)} ({sum(statute_hits)}/{len(statute_hits)})")
    print(f"Case Hit@{TOP_K_CASES}              : {_pct(case_hits)} ({sum(case_hits)}/{len(case_hits)})")
    print(f"Statute Precision@{TOP_K_STATUTES}     : {_pct(statute_precs)}")
    print(f"Case Precision@{TOP_K_CASES}         : {_pct(case_precs)}")
    print("=" * 60)

    # ---- Write CSV ---------------------------------------------------------
    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id", "query",
        "statute_hit", "statute_precision", "statute_returned", "statute_error",
        "case_hit", "case_precision", "case_returned", "case_error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            r["statute_returned"] = "; ".join(r.get("statute_returned", []))
            r["case_returned"] = "; ".join(r.get("case_returned", []))
            writer.writerow(r)
    print(f"\nPer-query results written to: {out_path}")


if __name__ == "__main__":
    main()
