"""Phase 5 — on/off graph evaluation.

Runs expand_for_answer() for 12 baseline questions and compares:
  OFF: graph disabled (empty list)
  ON:  graph enabled (live graph)

Reports per-question counts and writes eval/results/graph_eval_summary.md.

Usage (from backend/):
    GRAPH_ENABLED=true NEO4J_PASSWORD=legalai_dev_2026 \\
        venv/bin/python ../eval/run_graph_eval.py

Requires Neo4j running on bolt://localhost:7687 (dev instance).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# --- path setup (run from backend/ or repo root) ----
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# 12 questions selected from caselaw_questions.yaml (cover all seed cases)
QUESTIONS = [
    {"id": "EQ01", "query": "Is a s.106B certificate required for WhatsApp evidence?",
     "case_identifiers": ["2014KEHC6468(KLR)", "2024KEHC15763(KLR)", "2020KEHC8905(KLR)"]},
    {"id": "EQ02", "query": "Who must sign a s.106B certificate?",
     "case_identifiers": ["2024KEHC17038(KLR)", "2014KEHC6468(KLR)"]},
    {"id": "EQ03", "query": "Can a s.106B certificate be filed late or after the trial begins?",
     "case_identifiers": ["2014KEHC6468(KLR)", "2024KEHC15763(KLR)"]},
    {"id": "EQ04", "query": "When will the court set aside a regular default judgment?",
     "case_identifiers": ["2026SCC241(KLR)", "2001KECA154(KLR)"]},
    {"id": "EQ05", "query": "Is service via WhatsApp valid service of court process?",
     "case_identifiers": ["2026SCC241(KLR)"]},
    {"id": "EQ06", "query": "Adverse inference for failing to call a witness",
     "case_identifiers": ["2026SCC242(KLR)"]},
    {"id": "EQ07", "query": "Is the Small Claims Court bound by strict rules of evidence?",
     "case_identifiers": ["2026SCC242(KLR)", "2024KEHC15763(KLR)"]},
    {"id": "EQ08", "query": "What are the conditions for admitting M-Pesa statements as evidence?",
     "case_identifiers": ["2026SCC241(KLR)"]},
    {"id": "EQ09", "query": "Does the Small Claims Court have jurisdiction over employment claims?",
     "case_identifiers": ["2026SCC235(KLR)"]},
    {"id": "EQ10", "query": "What is the jurisdiction of the Small Claims Court in commercial cases?",
     "case_identifiers": ["2026SCC235(KLR)", "2026SCC242(KLR)"]},
    {"id": "EQ11", "query": "What must a defamation claimant prove regarding publication?",
     "case_identifiers": ["2020KEHC8905(KLR)"]},
    {"id": "EQ12", "query": "What is the standard for granting a stay of arbitral proceedings?",
     "case_identifiers": ["2010KECA346(KLR)"]},
]


def run_on(query: str, case_identifiers: list[str]) -> list[dict]:
    """Run expand_for_answer() (GRAPH_ENABLED=true path)."""
    from graph.queries import expand_for_answer
    return expand_for_answer(case_identifiers, query, forum="SCC", cap=5)


def run_off(query: str, case_identifiers: list[str]) -> list[dict]:
    """Return [] (baseline — no graph)."""
    return []


def _fmt_principles(results: list[dict]) -> str:
    if not results:
        return "(none)"
    return "; ".join(
        f"{r['principle_key']} [{r.get('kind', '?')}, binding={r.get('is_binding', False)}]"
        for r in results
    )


def main():
    import graph.config as cfg
    if not cfg.GRAPH_ENABLED:
        print("ERROR: set GRAPH_ENABLED=true to run this script.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for q in QUESTIONS:
        on_result  = run_on(q["query"], q["case_identifiers"])
        off_result = run_off(q["query"], q["case_identifiers"])

        binding_count = sum(1 for r in on_result if r.get("is_binding"))
        rows.append({
            "id": q["id"],
            "query": q["query"],
            "case_count": len(q["case_identifiers"]),
            "off_count": len(off_result),
            "on_count": len(on_result),
            "binding_added": binding_count,
            "principles": _fmt_principles(on_result),
        })
        print(f"{q['id']}: OFF={len(off_result):2d}  ON={len(on_result):2d}  binding={binding_count}  "
              f"q={q['query'][:60]}")

    total_on  = sum(r["on_count"]  for r in rows)
    total_off = sum(r["off_count"] for r in rows)
    total_bind = sum(r["binding_added"] for r in rows)

    print(f"\nTotal OFF: {total_off}  ON: {total_on}  binding: {total_bind}")

    # Write summary
    out_dir = _HERE / "results"
    out_dir.mkdir(exist_ok=True)
    summary_path = out_dir / "graph_eval_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Graph Eval Summary — Phase 5\n\n")
        f.write(f"Date: 2026-11-24  Branch: feat/graph-fastpath\n\n")
        f.write("## Per-question results\n\n")
        f.write("| ID | Query (truncated) | Graph OFF | Graph ON | Binding added | Principles |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            q60 = r["query"][:58].replace("|", "/")
            f.write(f"| {r['id']} | {q60} | {r['off_count']} | {r['on_count']} | {r['binding_added']} | {r['principles'][:80]} |\n")
        f.write(f"\n**Total**: OFF={total_off}, ON={total_on}, binding added={total_bind}\n\n")
        f.write("## Interpretation\n\n")
        f.write("- **Graph ON** adds principles from court-confirmed HOLDS/STATES_OBITER edges.\n")
        f.write("- **Binding** = the holding court sits above the SCC forum in the BINDS hierarchy.\n")
        f.write("- OFF baseline is always 0 (no graph).\n")
        f.write("- Principle text is passed to the LLM as a [G#.1] source block.\n")
    print(f"\nReport written to {summary_path}")


if __name__ == "__main__":
    main()
