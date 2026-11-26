"""Run isolated retrieval comparisons. See the implementation report for setup."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--statute-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=list("ABCD"), required=True)
    parser.add_argument("--baseline-retriever", type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--query-id", action="append", help="Run only selected fixture IDs")
    args = parser.parse_args()

    import config
    config.CASELAW_DB_DIR = args.case_root / "CaseLawDB"
    config.STATUTES_DB_DIR = args.statute_root / "StatutesDB"
    config.MASTER_RULES_DB_DIR = args.statute_root / "MasterRulesDB"
    from direct_answer import Passage, assemble, respond
    from direct_retrieval import normal, resolve, retrieve_resolved
    from groq_budget import MemoryLedger, research_budget
    from groq_client import call_groq
    from graph.config import GRAPH_ENABLED
    from Retriever import retrieve_relevant_cases
    from RulesRetriever import retrieve_relevant_rules
    if args.mode == "D" and not GRAPH_ENABLED:
        parser.error("D requires a development graph connection")
    if args.mode == "A":
        if not args.baseline_retriever:
            parser.error("A requires the preserved baseline Retriever.py")
        spec = importlib.util.spec_from_file_location("baseline_retriever", args.baseline_retriever)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        retrieve_relevant_cases = baseline.retrieve_relevant_cases

    ledger = MemoryLedger()
    output = []
    sets = ["diagnostics"] if args.diagnostics_only else ["diagnostics", "benchmark"]
    for suite in sets:
        fixtures = json.loads((HERE / f"level1_direct_{suite}.json").read_text())
        for gold in fixtures:
            key, query = gold["id"], gold["query"]
            if args.query_id and key not in args.query_id:
                continue
            resolution = resolve(query)
            start = time.monotonic()
            graph = args.mode == "D"
            if args.mode in "AB" and gold.get("case"):
                retrieval = retrieve_relevant_cases(
                    {"vector_query": query, "target_components": [], "legal_domains": []},
                    verbose=False, skip_summaries=True,
                )
                retrieval["resolution"] = resolution
            elif args.mode in "AB":
                reference = resolution["provision"]
                rules = retrieve_relevant_rules({
                    "vector_query": query, "entities": {
                        "section_numbers": [reference["section"]],
                        "source_documents": [reference["title"]],
                    },
                }, verbose=False)
                retrieval = {"resolution": resolution, "results": [], "rules": rules["results"]}
            else:
                retrieval = retrieve_resolved(query, resolution, graph_enabled=graph)
            if args.mode == "A":
                # Reproduce historical clipping for the retrieval-only comparison.
                passages = []
                for kind, letter, rows, size in [
                    ("case", "C", retrieval.get("results", []), 500),
                    ("statute", "S", retrieval.get("rules", []), 600),
                ]:
                    for i, row in enumerate(rows, 1):
                        for j, ch in enumerate(row.get("chunks", []), 1):
                            passages.append(Passage(f"{letter}{i}.{j}", ch["chunk_id"], "", ch["text"][:size], kind,
                                kind == "statute" or row.get("case_name") == gold.get("case")))
            else:
                passages = assemble(retrieval)
            mode = resolution.get("mode")
            ranks = [row["case_name"] for row in retrieval.get("results", [])]
            if mode == "DIRECT_CASE_LOOKUP" and retrieval.get("authority_status") == "indexed_lookup":
                ranks.insert(0, resolution["case"]["case_identifier"])
            intended_id = resolution.get("case", {}).get("case_identifier")
            is_primary = lambda p: p.primary and bool(p.chunk_id)
            text = normal(" ".join(p.text for p in passages if is_primary(p)))
            found = [g for g in gold["gold"] if normal(g) in text]
            row = {"suite": suite, "id": key, "query": query, "mode": mode,
                   "case_id": intended_id, "expected_case": gold.get("case"), "category": gold["category"],
                   "ranks": ranks, "gold_anchors": gold["gold"], "found_primary_anchors": found,
                   "expected_statute": gold.get("statute"), "statute_matches": [
                       r["article_number"] for r in retrieval.get("rules", [])
                       if r["source_document"] == gold.get("statute") and r["article_number"] == gold.get("section")
                   ], "graph_case": retrieval.get("graph_case", {}).get("case_key"),
                   "authority_status": retrieval.get("authority_status"), "origin": retrieval.get("candidate_origin"),
                   "selected": [p.chunk_id for p in passages],
                   "retrieval_ms": round((time.monotonic() - start) * 1000)}
            if args.generate:
                with research_budget(lambda stage, *_: ledger.reserve(key, stage)) as budget:
                    start = time.monotonic()
                    answer, claims, status = respond(query, retrieval, passages, call_groq)
                    row.update(answer=answer, claims=claims, verification=status, groq=budget.outcomes,
                               generation_ms=round((time.monotonic() - start) * 1000))
            output.append(row)
            print(args.mode, suite, key, row["ranks"][:3], len(found), row.get("verification", ""), flush=True)
            args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
