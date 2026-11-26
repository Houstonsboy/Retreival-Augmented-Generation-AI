"""Seed graph loader (Phase 2 fast-path).

Usage:
    cd backend
    python -m graph.seed_loader load [--dry-run] [--file PATH]
    python -m graph.seed_loader status

Loads backend/graph/seed/scc_caselaw_graph.json into Neo4j.  Idempotent: safe
to rerun — all writes use MERGE.  No LLM calls.

The loader requires GRAPH_ENABLED=true and NEO4J_PASSWORD to be set.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).resolve().parent / "seed" / "scc_caselaw_graph.json"
_OVERRIDES_PATH = Path(__file__).resolve().parent / "seed" / "case_identifier_overrides.yaml"

# ── Seed id prefix → (Neo4j label, key_field)
_LABEL_MAP = {
    "court:": ("Court", "code"),
    "case:":  ("Case",  "case_key"),
    "ext:":   ("Case",  "case_key"),
    "lower:": ("LowerCourtDecision", "lcd_key"),
    "prov:":  ("Provision", "provision_key"),
    "issue:": ("Issue", "issue_key"),
    "pr:":    ("Principle", "principle_key"),
    "ev:":    ("EvidenceType", "code"),
    "con:":   ("Concept", "concept_key"),
}

# Regex to extract YEAR/COURT/NUM from seed ids like case:2024KEHC15763
_CASE_ID_RE = re.compile(r"^(?:case|ext):(\d{4})([A-Z]+)(\d+)$")


def _load_seed(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {k: v for k, v in data.items() if isinstance(v, str)}


def _chroma_id_from_seed(seed_id: str, overrides: dict[str, str]) -> str | None:
    """Return Chroma case_identifier for a seed Case id, or None."""
    if seed_id in overrides:
        return overrides[seed_id]
    m = _CASE_ID_RE.match(seed_id)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}(KLR)"
    return None


def _node_key(seed_id: str, node: dict, overrides: dict) -> tuple[str, str, str]:
    """Return (label, key_field, key_value) for a node."""
    from .keys import case_key as make_case_key, provision_key as make_provision_key

    for prefix, (label, key_field) in _LABEL_MAP.items():
        if seed_id.startswith(prefix):
            bare = seed_id[len(prefix):]
            if prefix in ("case:", "ext:"):
                citation = node.get("citation", "")
                ck = make_case_key(citation, node.get("name", "")) if citation else f"UNKNOWN-{bare}"
                return label, key_field, ck
            elif prefix == "court:":
                return label, key_field, bare.upper()
            elif prefix == "lower:":
                return label, key_field, bare
            elif prefix == "prov:":
                act = node.get("act", "")
                section = node.get("section", bare)
                pk = make_provision_key(act, section) if act else f"UNKNOWN/{bare}"
                return label, key_field, pk
            elif prefix == "issue:":
                return label, key_field, bare
            elif prefix == "pr:":
                return label, key_field, bare
            elif prefix == "ev:":
                return label, key_field, bare
            elif prefix == "con:":
                return label, key_field, bare
    raise ValueError(f"Unrecognised seed id prefix: {seed_id!r}")


def _build_node_props(seed_id: str, node: dict, overrides: dict, ingest_run_id: str, label: str, key_field: str, key_value: str) -> dict:
    """Build property dict for a node MERGE."""
    props: dict[str, Any] = {
        key_field: key_value,
        "seed_id": seed_id,
        "extractor": "seed",
        "ingest_run_id": ingest_run_id,
        "source": "scc_caselaw_graph.json@seed-0.1",
    }

    # Copy seed properties (excluding type/id which we handle separately)
    for k, v in node.items():
        if k in ("id", "type"):
            continue
        if isinstance(v, list):
            props[k] = v
            # aliases_text for full-text index
            if k == "aliases":
                props["aliases_text"] = " ".join(v)
        elif v is not None:
            props[k] = v

    # Case-specific: in_corpus flag and chroma link
    if label == "Case":
        is_in_corpus = node.get("in_corpus", False) or node.get("in_vector_index", False)
        props["in_corpus"] = is_in_corpus
        chroma_id = _chroma_id_from_seed(seed_id, overrides)
        if chroma_id and node.get("in_vector_index", is_in_corpus):
            props["chroma_case_identifier"] = chroma_id

    return props


def _load_nodes(client, nodes: list[dict], overrides: dict, ingest_run_id: str, dry_run: bool) -> dict[str, tuple[str, str, str]]:
    """MERGE all seed nodes. Returns {seed_id: (label, key_field, key_value)}."""
    seed_to_key: dict[str, tuple[str, str, str]] = {}

    for node in nodes:
        seed_id = node["id"]
        try:
            label, key_field, key_value = _node_key(seed_id, node, overrides)
        except ValueError as e:
            logger.warning("Skipping node %s: %s", seed_id, e)
            continue

        props = _build_node_props(seed_id, node, overrides, ingest_run_id, label, key_field, key_value)
        seed_to_key[seed_id] = (label, key_field, key_value)

        if dry_run:
            logger.info("DRY-RUN: MERGE (%s {%s: %r}) SET props=%r", label, key_field, key_value, list(props.keys()))
            continue

        # Build SET clause for all props
        set_parts = ", ".join(f"n.{k} = ${k}" for k in props)
        query = f"MERGE (n:{label} {{{key_field}: ${key_field}}}) SET {set_parts}"
        try:
            client.write(query, **props)
        except Exception as exc:
            logger.error("Failed to upsert node %s (%s=%s): %s", seed_id, key_field, key_value, exc)

    return seed_to_key


def _load_edges(client, edges: list[dict], seed_to_key: dict, ingest_run_id: str, dry_run: bool) -> int:
    """MERGE all seed edges. Returns count of successfully merged edges."""
    merged = 0
    for edge in edges:
        src_id = edge["src"]
        dst_id = edge["dst"]
        rel = edge["rel"]

        if src_id not in seed_to_key or dst_id not in seed_to_key:
            logger.warning("Edge skip: missing node(s) %s or %s", src_id, dst_id)
            continue

        s_label, s_key_field, s_key_val = seed_to_key[src_id]
        d_label, d_key_field, d_key_val = seed_to_key[dst_id]

        # Edge properties (excluding src/dst/rel)
        edge_props: dict[str, Any] = {}
        for k, v in edge.items():
            if k in ("src", "dst", "rel") or v is None:
                continue
            edge_props[k] = v

        if dry_run:
            logger.info("DRY-RUN: MERGE (%s {%s:%r}) -[:%s]-> (%s {%s:%r})",
                        s_label, s_key_field, s_key_val, rel, d_label, d_key_field, d_key_val)
            merged += 1
            continue

        # Build edge MERGE query
        edge_prop_str = ""
        if edge_props:
            edge_prop_str = " {" + ", ".join(f"{k}: ${k}" for k in edge_props) + "}"

        query = (
            f"MATCH (src:{s_label} {{{s_key_field}: $src_key}}) "
            f"MATCH (dst:{d_label} {{{d_key_field}: $dst_key}}) "
            f"MERGE (src)-[r:{rel}{edge_prop_str}]->(dst)"
        )
        params = {"src_key": s_key_val, "dst_key": d_key_val, **edge_props}
        try:
            client.write(query, **params)
            merged += 1
        except Exception as exc:
            logger.error("Failed to merge edge %s -[%s]-> %s: %s", src_id, rel, dst_id, exc)

    return merged


def _validate(client, seed_to_key: dict) -> list[str]:
    """Run post-load validation checks. Returns list of error messages."""
    errors = []

    # Count seed nodes
    seed_count = client.read(
        "MATCH (n) WHERE n.source = 'scc_caselaw_graph.json@seed-0.1' RETURN count(n) AS c"
    )[0]["c"]
    if seed_count != 149:
        errors.append(f"Expected 149 seed nodes, found {seed_count}")

    # Count seed edges (approximate — edges don't carry source)
    ogembo_key = seed_to_key.get("case:2024KEHC15763", ("Case", "case_key", "KEHC-2024-15763"))[2]
    nonny_key = seed_to_key.get("case:2014KEHC6468", ("Case", "case_key", "KEHC-2014-6468"))[2]
    rows = client.read(
        "MATCH (a:Case {case_key: $ck})-[r:CITES]->(b:Case {case_key: $dk}) "
        "RETURN r.treatment AS t, r.para AS p",
        ck=ogembo_key, dk=nonny_key,
    )
    if not any(r.get("t") == "DISTINGUISHED" and r.get("p") == "44" for r in rows):
        errors.append("Missing: Ogembo CITES Nonny DISTINGUISHED para:44")

    rows = client.read(
        "MATCH (a:Principle {principle_key: 'scc-not-strict-evidence'})-[r:QUALIFIES]->"
        "(b:Principle {principle_key: 'cert-mandatory'}) RETURN count(r) AS c"
    )
    if not rows or rows[0]["c"] == 0:
        errors.append("Missing: pr:scc-not-strict-evidence QUALIFIES pr:cert-mandatory")

    rows = client.read(
        "MATCH (a:Principle {principle_key: 'cert-late-not-after-trial'})-[r:CONFLICTS_WITH]->"
        "(b:Principle {principle_key: 'cert-late-allowed'}) RETURN count(r) AS c"
    )
    if not rows or rows[0]["c"] == 0:
        errors.append("Missing: pr:cert-late-not-after-trial CONFLICTS_WITH pr:cert-late-allowed")

    # Barisa has >= 3 incoming CITES
    barisa_key = seed_to_key.get("case:2011KEHC1481", ("Case", "case_key", "KEHC-2011-1481"))[2]
    rows = client.read(
        "MATCH ()-[:CITES]->(b:Case {case_key: $k}) RETURN count(*) AS c",
        k=barisa_key,
    )
    barisa_cites = rows[0]["c"] if rows else 0
    if barisa_cites < 3:
        errors.append(f"Barisa should have >= 3 incoming CITES, found {barisa_cites}")

    return errors


def _count_summary(client) -> dict:
    rows = client.read(
        "MATCH (n) WHERE NOT n:Migration RETURN labels(n)[0] AS label, count(n) AS c"
    )
    return {r["label"]: r["c"] for r in rows if r["label"]}


def cmd_load(args):
    from . import config
    if not config.GRAPH_ENABLED:
        print("ERROR: GRAPH_ENABLED must be true to load seed graph.", file=sys.stderr)
        sys.exit(1)
    from .client import get_client

    seed_path = Path(args.file) if args.file else _SEED_PATH
    if not seed_path.exists():
        print(f"ERROR: seed file not found at {seed_path}", file=sys.stderr)
        sys.exit(1)

    seed = _load_seed(seed_path)
    overrides = _load_overrides(_OVERRIDES_PATH)
    ingest_run_id = str(uuid.uuid4())

    print(f"Loading seed graph from {seed_path}")
    print(f"  version={seed.get('version')}, nodes={len(seed['nodes'])}, edges={len(seed['edges'])}")
    print(f"  ingest_run_id={ingest_run_id}")
    if args.dry_run:
        print("  DRY-RUN mode (no writes)")

    client = None if args.dry_run else get_client()

    # Load nodes
    seed_to_key = _load_nodes(client, seed["nodes"], overrides, ingest_run_id, args.dry_run)
    print(f"  Nodes processed: {len(seed_to_key)}")

    # Load edges
    merged_edges = _load_edges(client, seed["edges"], seed_to_key, ingest_run_id, args.dry_run)
    print(f"  Edges merged: {merged_edges}")

    if args.dry_run:
        print("DRY-RUN complete — no data written.")
        return

    # Validate
    errors = _validate(client, seed_to_key)
    if errors:
        print(f"\nVALIDATION ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("\nAll validation checks passed.")

    # Summary
    counts = _count_summary(client)
    seed_nodes = client.read(
        "MATCH (n) WHERE n.source = 'scc_caselaw_graph.json@seed-0.1' "
        "RETURN labels(n)[0] AS label, count(n) AS c"
    )
    seed_counts = {r["label"]: r["c"] for r in seed_nodes if r["label"]}

    print("\nNode counts (seed only):")
    for label, c in sorted(seed_counts.items()):
        print(f"  {label}: {c}")

    print("\nTotal node counts (including pre-existing from migrations):")
    for label, c in sorted(counts.items()):
        print(f"  {label}: {c}")

    # Chroma mapping table
    print("\nChroma case_identifier mapping:")
    chroma_rows = client.read(
        "MATCH (n:Case) WHERE n.chroma_case_identifier IS NOT NULL "
        "RETURN n.case_key AS ck, n.name AS name, n.chroma_case_identifier AS chroma_id"
    )
    no_chroma = client.read(
        "MATCH (n:Case) WHERE n.in_corpus = true AND n.chroma_case_identifier IS NULL "
        "RETURN n.case_key AS ck, n.name AS name, n.seed_id AS sid"
    )
    for r in sorted(chroma_rows, key=lambda x: x["ck"]):
        print(f"  {r['ck']:25s}  → {r['chroma_id']:30s}  ({r['name']})")
    if no_chroma:
        print(f"\nIn-corpus cases with no Chroma match ({len(no_chroma)}):")
        for r in no_chroma:
            print(f"  {r['ck']:25s}  ({r.get('name','')})  seed_id={r.get('sid','')}")


def cmd_status(args):
    from . import config
    if not config.GRAPH_ENABLED:
        print("GRAPH_ENABLED=false — no graph available.")
        return
    from .client import get_client
    client = get_client()
    counts = _count_summary(client)
    print("Node counts:")
    for label, c in sorted(counts.items()):
        print(f"  {label}: {c}")
    seed_rows = client.read(
        "MATCH (n) WHERE n.source = 'scc_caselaw_graph.json@seed-0.1' "
        "RETURN count(n) AS c"
    )
    seed_count = seed_rows[0]["c"] if seed_rows else 0
    print(f"\nSeed nodes loaded: {seed_count}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m graph.seed_loader")
    sub = parser.add_subparsers(dest="command")

    load_p = sub.add_parser("load", help="Load seed graph into Neo4j.")
    load_p.add_argument("--dry-run", action="store_true", help="Print actions without writing.")
    load_p.add_argument("--file", default=None, help=f"Path to seed JSON (default: {_SEED_PATH})")

    sub.add_parser("status", help="Show current node counts.")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {"load": cmd_load, "status": cmd_status}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
