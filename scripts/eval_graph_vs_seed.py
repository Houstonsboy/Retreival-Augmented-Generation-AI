"""B7 — Graph validation against scc_caselaw_seed_graph.json.

Computes precision/recall per edge type and reports specific named assertions.

Usage:
    cd backend && python ../scripts/eval_graph_vs_seed.py

Requires:
    GRAPH_ENABLED=true  NEO4J_PASSWORD=<password>
    scc_caselaw_seed_graph.json in project root (or --seed-path override)

If the seed graph file is missing, the script generates a skeleton template
and exits with a warning (does not fail).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_backend))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_backend / ".env", override=True)

_SEED_PATH = Path(__file__).resolve().parent.parent / "scc_caselaw_seed_graph.json"
_REPORT_PATH = Path(__file__).resolve().parent.parent / "eval" / "graph_vs_seed_report.json"


def _client():
    from graph.client import get_client
    return get_client()


def _graph_enabled() -> bool:
    import os
    return os.getenv("GRAPH_ENABLED", "false").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------

def load_seed(path: Path) -> dict | None:
    if not path.exists():
        print(f"WARNING: seed graph not found at {path}")
        print("  Generating skeleton template → scc_caselaw_seed_graph.json.template")
        skeleton = {
            "nodes": [],
            "edges": [
                {"type": "CITES", "src": "CASE_KEY", "dst": "CASE_KEY", "treatment": "FOLLOWED"},
                {"type": "INTERPRETS", "src": "CASE_KEY", "provision": "PROVISION_KEY"},
                {"type": "RULED_ON_EVIDENCE", "case_key": "CASE_KEY", "para": "pN", "evidence_type": "whatsapp", "result": "ADMITTED"},
                {"type": "STATES", "para": "PARA_KEY", "principle": "PRINCIPLE_KEY"},
                {"type": "QUALIFIES", "src_principle": "PRINCIPLE_KEY", "dst_principle": "PRINCIPLE_KEY"},
            ],
        }
        tmpl_path = path.parent / "scc_caselaw_seed_graph.json.template"
        tmpl_path.write_text(json.dumps(skeleton, indent=2))
        return None
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Graph queries
# ---------------------------------------------------------------------------

def get_all_cites(client) -> list[dict]:
    return client.read(
        """
        MATCH (para:Paragraph)-[r:CITES]->(cited:Case)
        RETURN para.case_key AS src, cited.case_key AS dst, r.treatment AS treatment
        """
    )


def get_all_interprets(client) -> list[dict]:
    return client.read(
        """
        MATCH (para:Paragraph)-[r:INTERPRETS|APPLIES]->(prov:Provision)
        RETURN para.case_key AS case_key, prov.provision_key AS provision_key, type(r) AS mode
        """
    )


def get_all_evidence_rulings(client) -> list[dict]:
    return client.read(
        """
        MATCH (para:Paragraph)-[r:RULED_ON_EVIDENCE]->(et:EvidenceType)
        RETURN para.case_key AS case_key, para.para_key AS para_key,
               et.code AS evidence_type, r.result AS result
        """
    )


def get_all_principles(client) -> list[dict]:
    return client.read(
        """
        MATCH (para:Paragraph)-[r:STATES]->(p:Principle)
        RETURN para.case_key AS case_key, para.para_key AS para_key,
               p.principle_key AS principle_key, p.text AS text
        """
    )


def get_principle_relations(client) -> list[dict]:
    return client.read(
        """
        MATCH (p1:Principle)-[r:QUALIFIES|CONFLICTS_WITH|SUPPORTS]->(p2:Principle)
        RETURN p1.principle_key AS src, type(r) AS rel_type, p2.principle_key AS dst
        """
    )


def get_case_in_degree(client, case_key: str) -> int:
    rows = client.read(
        "MATCH (para:Paragraph)-[:CITES]->(c:Case {case_key: $ck}) "
        "RETURN count(DISTINCT para.case_key) AS n",
        ck=case_key,
    )
    return rows[0]["n"] if rows else 0


def get_cites_treatment(client, src_case: str, dst_case: str) -> str | None:
    rows = client.read(
        "MATCH (para:Paragraph {case_key: $src})-[r:CITES]->(c:Case {case_key: $dst}) "
        "RETURN r.treatment AS treatment LIMIT 1",
        src=src_case, dst=dst_case,
    )
    return rows[0]["treatment"] if rows else None


# ---------------------------------------------------------------------------
# Precision / Recall helpers
# ---------------------------------------------------------------------------

def _pr(tp: int, found: int, expected: int) -> tuple[float, float]:
    precision = tp / found if found else 0.0
    recall = tp / expected if expected else 0.0
    return precision, recall


def eval_cites(seed_edges: list[dict], graph_cites: list[dict]) -> dict:
    seed_pairs = {(e["src"], e["dst"]) for e in seed_edges if e.get("type") == "CITES"}
    graph_pairs = {(r["src"], r["dst"]) for r in graph_cites}

    tp_target = len(seed_pairs & graph_pairs)
    p, r = _pr(tp_target, len(graph_pairs), len(seed_pairs))

    # Treatment accuracy
    treatment_correct = 0
    treatment_total = 0
    seed_treatment = {(e["src"], e["dst"]): e.get("treatment") for e in seed_edges if e.get("type") == "CITES"}
    graph_treatment = {(r["src"], r["dst"]): r.get("treatment") for r in graph_cites}
    for pair, expected_t in seed_treatment.items():
        if pair in graph_treatment:
            treatment_total += 1
            if graph_treatment[pair] == expected_t:
                treatment_correct += 1

    treatment_acc = treatment_correct / treatment_total if treatment_total else 0.0
    return {"precision": p, "recall": r, "treatment_accuracy": treatment_acc,
            "seed_count": len(seed_pairs), "graph_count": len(graph_pairs)}


def eval_provisions(seed_edges: list[dict], graph_interps: list[dict]) -> dict:
    seed_pairs = {(e["src"], e.get("provision")) for e in seed_edges
                  if e.get("type") in ("INTERPRETS", "APPLIES")}
    graph_pairs = {(r["case_key"], r["provision_key"]) for r in graph_interps}
    tp = len(seed_pairs & graph_pairs)
    p, r = _pr(tp, len(graph_pairs), len(seed_pairs))
    return {"precision": p, "recall": r,
            "seed_count": len(seed_pairs), "graph_count": len(graph_pairs)}


def eval_principles(seed_edges: list[dict], graph_principles: list[dict]) -> dict:
    # Semantic match — allow any principle from the same case
    seed_pairs = {(e["case_key"],) for e in seed_edges
                  if e.get("type") in ("STATES", "HOLDS")}
    graph_pairs = {(r["case_key"],) for r in graph_principles}
    tp = len(seed_pairs & graph_pairs)
    p, r = _pr(tp, len(graph_pairs), len(seed_pairs))
    return {"precision": p, "recall": r,
            "seed_count": len(seed_pairs), "graph_count": len(graph_pairs)}


def eval_qualifies(seed_edges: list[dict], graph_relations: list[dict]) -> dict:
    seed_rels = {(e.get("src_principle", ""), e.get("dst_principle", ""))
                 for e in seed_edges if e.get("type") in ("QUALIFIES", "CONFLICTS_WITH")}
    graph_rels = {(r["src"], r["dst"]) for r in graph_relations}
    tp = len(seed_rels & graph_rels)
    p, r = _pr(tp, len(graph_rels), len(seed_rels))
    return {"precision": p, "recall": r,
            "seed_count": len(seed_rels), "graph_count": len(graph_rels)}


def eval_evidence_rulings(seed_edges: list[dict], graph_ev: list[dict]) -> dict:
    seed_tuples = {(e["case_key"], e.get("evidence_type"))
                   for e in seed_edges if e.get("type") == "RULED_ON_EVIDENCE"}
    graph_tuples = {(r["case_key"], r["evidence_type"]) for r in graph_ev}
    tp = len(seed_tuples & graph_tuples)
    p, r = _pr(tp, len(graph_tuples), len(seed_tuples))
    return {"precision": p, "recall": r,
            "seed_count": len(seed_tuples), "graph_count": len(graph_tuples)}


# ---------------------------------------------------------------------------
# Named assertions (B7 specific checks)
# ---------------------------------------------------------------------------

def run_named_assertions(client) -> list[dict]:
    checks: list[dict] = []

    def chk(name: str, passed: bool, detail: str = "") -> None:
        icon = "✓" if passed else "✗"
        checks.append({"name": name, "passed": passed, "detail": detail})
        print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))

    # Barisa in-degree ≥ 3 (Nonny, PNN via MNN, Ogembo cite it)
    barisa_indegree = get_case_in_degree(client, "KEHC-2011-1481")
    chk("Barisa in-degree ≥ 3", barisa_indegree >= 3, f"actual={barisa_indegree}")

    # Ogembo → Nonny is DISTINGUISHED at p44
    treatment = get_cites_treatment(client, "KEHC-2024-15763", "KEHC-2014-6468")
    chk("Ogembo→Nonny treatment=DISTINGUISHED", treatment == "DISTINGUISHED", f"actual={treatment}")

    # New Titanic → Maisha Mapya is DEPARTED_FROM
    rows = client.read(
        "MATCH (para:Paragraph {case_key: 'SCC-2026-235'})-[r:CITES]->(c:Case) "
        "WHERE c.name CONTAINS 'Maisha' OR c.case_key CONTAINS 'Maisha' "
        "RETURN r.treatment AS t LIMIT 1"
    )
    nt_treatment = rows[0]["t"] if rows else None
    chk("New Titanic→Maisha Mapya DEPARTED_FROM", nt_treatment == "DEPARTED_FROM",
        f"actual={nt_treatment}")

    # Umoja "not bound by strict rules" QUALIFIES "certificate mandatory"
    rows = client.read(
        "MATCH (p1:Principle)-[:QUALIFIES]->(p2:Principle) "
        "WHERE (toLower(p1.text) CONTAINS 'strict' OR toLower(p1.text) CONTAINS 'bound') "
        "AND (toLower(p2.text) CONTAINS 'certificate') "
        "RETURN p1.principle_key LIMIT 1"
    )
    chk("Umoja strict-evidence QUALIFIES certificate", len(rows) > 0)

    # Ndeti p17 RULED_ON_EVIDENCE mpesa ACCEPTED_TO_LINK_IDENTITY, BEARS_ON A
    rows = client.read(
        "MATCH (para:Paragraph {case_key: 'SCC-2026-241', para_no: 17})"
        "-[r:RULED_ON_EVIDENCE]->(et:EvidenceType {code: 'mpesa'}) "
        "RETURN r.result AS result"
    )
    ndeti_ev = rows[0]["result"] if rows else None
    chk("Ndeti p17 mpesa ACCEPTED_TO_LINK_IDENTITY",
        ndeti_ev == "ACCEPTED_TO_LINK_IDENTITY", f"actual={ndeti_ev}")

    rows = client.read(
        "MATCH (para:Paragraph {case_key: 'SCC-2026-241', para_no: 17})"
        "-[:STATES]->(p:Principle)-[:BEARS_ON]->(ce:ContractElement {code: 'A'}) "
        "RETURN p.principle_key LIMIT 1"
    )
    chk("Ndeti p17 principle BEARS_ON ContractElement A", len(rows) > 0)

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _graph_enabled():
        print("GRAPH_ENABLED=false — cannot evaluate graph. Set GRAPH_ENABLED=true.")
        sys.exit(1)

    seed = load_seed(_SEED_PATH)
    if seed is None:
        print("Seed graph missing — skipping recall checks. See template above.")
        sys.exit(0)

    seed_edges = seed.get("edges", [])
    client = _client()

    print("Running named assertions …")
    assertion_results = run_named_assertions(client)

    print("\nFetching graph data …")
    graph_cites = get_all_cites(client)
    graph_interps = get_all_interprets(client)
    graph_ev = get_all_evidence_rulings(client)
    graph_principles = get_all_principles(client)
    graph_relations = get_principle_relations(client)

    print("\n=== Precision/Recall by Edge Type ===")
    metrics: dict[str, dict] = {}

    cites_m = eval_cites(seed_edges, graph_cites)
    metrics["CITES"] = cites_m
    print(f"  CITES target recall:     {cites_m['recall']:.2%} (target ≥ 0.95)  {'✓' if cites_m['recall'] >= 0.95 else '✗'}")
    print(f"  treatment accuracy:      {cites_m['treatment_accuracy']:.2%} (target ≥ 0.85)  {'✓' if cites_m['treatment_accuracy'] >= 0.85 else '✗'}")

    prov_m = eval_provisions(seed_edges, graph_interps)
    metrics["PROVISIONS"] = prov_m
    print(f"  PROVISIONS recall:       {prov_m['recall']:.2%} (target ≥ 0.90)  {'✓' if prov_m['recall'] >= 0.90 else '✗'}")

    prin_m = eval_principles(seed_edges, graph_principles)
    metrics["PRINCIPLES"] = prin_m
    print(f"  PRINCIPLES recall:       {prin_m['recall']:.2%} (target ≥ 0.80)  {'✓' if prin_m['recall'] >= 0.80 else '✗'}")

    qual_m = eval_qualifies(seed_edges, graph_relations)
    metrics["QUALIFIES_CONFLICTS"] = qual_m
    seed_qual = qual_m.get("seed_count", 0)
    print(f"  QUALIFIES/CONFLICTS:     {qual_m['recall']:.2%} (must find all {seed_qual})  {'✓' if qual_m['recall'] >= 1.0 else '✗'}")

    ev_m = eval_evidence_rulings(seed_edges, graph_ev)
    metrics["RULED_ON_EVIDENCE"] = ev_m
    print(f"  RULED_ON_EVIDENCE recall:{ev_m['recall']:.2%}")

    overall_pass = (
        cites_m["recall"] >= 0.95
        and cites_m["treatment_accuracy"] >= 0.85
        and prov_m["recall"] >= 0.90
        and prin_m["recall"] >= 0.80
        and qual_m["recall"] >= 1.0
    )

    assertions_pass = all(a["passed"] for a in assertion_results)
    print(f"\n{'='*50}")
    print(f"Overall metric targets: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Named assertions:       {'PASS' if assertions_pass else 'FAIL'}")

    report = {
        "metrics": metrics,
        "assertions": assertion_results,
        "overall_pass": overall_pass and assertions_pass,
    }
    _REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved → {_REPORT_PATH}")


if __name__ == "__main__":
    main()
