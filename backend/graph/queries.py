"""Parameterised query helpers for the case-law graph (B8).

Each function returns a list of dicts with:
  - para_key: str
  - path_explanation: str  (human-readable chain of edges that produced this result)

These are NOT wired into Retriever.py yet — that is the next integration prompt.
Unit-tested in backend/graph/tests/test_queries.py.

All queries return an empty list (not an error) when GRAPH_ENABLED=false.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _client():
    from .client import get_client
    return get_client()


def _graph_enabled() -> bool:
    from . import config
    return config.GRAPH_ENABLED


# ---------------------------------------------------------------------------
# doctrinal(issue_ids | principle_ids, forum)
# ---------------------------------------------------------------------------

def doctrinal(
    ids: list[str],
    forum: str = "SCC",
    include_proposed: bool = False,
) -> list[dict]:
    """
    For a list of issue or principle IDs, return supporting paragraphs
    ranked by authority, with QUALIFIES/CONFLICTS relations always included.

    Parameters
    ----------
    ids : list of issue_key or principle_key
    forum : court code for binding check (default "SCC")
    include_proposed : include proposed (unreviewed) principles
    """
    if not _graph_enabled():
        return []
    client = _client()
    from .authority import is_binding

    results: list[dict] = []

    for uid in ids:
        is_issue = uid.startswith("issue:")
        node_label = "Issue" if is_issue else "Principle"
        node_key = "issue_key" if is_issue else "principle_key"

        # Walk: Issue → Principle → Paragraph (via ANSWERS / STATES)
        rows = client.read(
            f"""
            MATCH (target:{node_label} {{{node_key}: $uid}})
            OPTIONAL MATCH (target)<-[:ANSWERS]-(prin:Principle)
            WITH coalesce(prin, target) AS p
            WHERE ({'' if include_proposed else "p.status <> 'proposed' AND"} true)
            MATCH (para:Paragraph)-[:STATES]->(p)
            MATCH (para)<-[:HAS_PARAGRAPH]-(c:Case)-[:DECIDED_IN]->(court:Court)
            OPTIONAL MATCH (p)-[rel:QUALIFIES|CONFLICTS_WITH|SUPPORTS]->(p2:Principle)
            RETURN
              para.para_key AS para_key,
              p.principle_key AS principle_key,
              p.text AS principle_text,
              p.kind AS principle_kind,
              court.code AS court_code,
              c.case_key AS case_key,
              collect(DISTINCT type(rel) + ':' + p2.principle_key) AS relations
            ORDER BY court.rank ASC
            """,
            uid=uid,
        )

        for row in rows:
            court_code = row.get("court_code", "")
            binding = is_binding(court_code, forum)
            relations_str = " | ".join(row.get("relations", []))
            explanation = (
                f"{uid} → {row['principle_key']} ({row['principle_kind']}) "
                f"← {row['para_key']} [{court_code}"
                f"{' ✓binding' if binding else ''}]"
                + (f" | rels: {relations_str}" if relations_str else "")
            )
            results.append({
                "para_key": row["para_key"],
                "path_explanation": explanation,
                "principle_key": row.get("principle_key", ""),
                "principle_text": row.get("principle_text", ""),
                "court_code": court_code,
                "is_binding": binding,
                "relations": row.get("relations", []),
            })

    return results


# ---------------------------------------------------------------------------
# case_lookup(case_key)
# ---------------------------------------------------------------------------

def case_lookup(case_key: str) -> dict:
    """
    Return holdings, outgoing citations with treatment, and incoming citations
    for a case.
    """
    if not _graph_enabled():
        return {}
    client = _client()

    # Holdings / principles
    holdings = client.read(
        """
        MATCH (c:Case {case_key: $ck})
        OPTIONAL MATCH (para:Paragraph)-[:HAS_PARAGRAPH|NEXT*0..]->(para)
        WHERE para.case_key = $ck
        MATCH (para)-[:STATES]->(p:Principle)
        RETURN para.para_key AS para_key, p.text AS principle, p.kind AS kind
        LIMIT 20
        """,
        ck=case_key,
    )

    # Outgoing citations (this case cites others)
    out_cites = client.read(
        """
        MATCH (para:Paragraph {case_key: $ck})-[r:CITES]->(cited:Case)
        RETURN cited.case_key AS cited_key, cited.name AS cited_name,
               r.treatment AS treatment, para.para_key AS para_key
        ORDER BY cited.year ASC
        """,
        ck=case_key,
    )

    # Incoming citations (others cite this case)
    in_cites = client.read(
        """
        MATCH (para:Paragraph)-[r:CITES]->(c:Case {case_key: $ck})
        RETURN para.case_key AS citing_case, r.treatment AS treatment,
               para.para_key AS para_key
        ORDER BY r.treatment
        """,
        ck=case_key,
    )

    return {
        "case_key": case_key,
        "holdings": holdings,
        "outgoing_citations": out_cites,
        "incoming_citations": in_cites,
    }


# ---------------------------------------------------------------------------
# provision(provision_key)
# ---------------------------------------------------------------------------

def provision(provision_key: str) -> list[dict]:
    """
    Walk the PART_OF tree upward from a provision, then return cases that
    interpret/apply/mention it, ranked by mode importance and then authority.
    """
    if not _graph_enabled():
        return []
    client = _client()

    rows = client.read(
        """
        MATCH (prov:Provision {provision_key: $pk})
        OPTIONAL MATCH (prov)-[:PART_OF*0..3]->(parent:Provision|Statute)
        MATCH (para:Paragraph)-[r:INTERPRETS|APPLIES|MENTIONS]->(prov)
        MATCH (para)<-[:HAS_PARAGRAPH]-(c:Case)-[:DECIDED_IN]->(court:Court)
        RETURN
          para.para_key AS para_key,
          type(r) AS mode,
          c.case_key AS case_key,
          court.code AS court_code,
          court.rank AS court_rank,
          collect(DISTINCT parent.provision_key) AS parent_path
        ORDER BY
          CASE type(r) WHEN 'INTERPRETS' THEN 0 WHEN 'APPLIES' THEN 1 ELSE 2 END ASC,
          court.rank ASC
        """,
        pk=provision_key,
    )

    results = []
    for row in rows:
        path_str = " → ".join(row.get("parent_path", []))
        explanation = (
            f"{provision_key} ← {row['mode']} ← {row['para_key']} "
            f"[{row['court_code']}]"
            + (f" | ancestry: {path_str}" if path_str else "")
        )
        results.append({
            "para_key": row["para_key"],
            "path_explanation": explanation,
            "mode": row.get("mode", ""),
            "case_key": row.get("case_key", ""),
            "court_code": row.get("court_code", ""),
        })

    return results


# ---------------------------------------------------------------------------
# from_paragraphs(para_keys)
# ---------------------------------------------------------------------------

def from_paragraphs(para_keys: list[str], max_hops: int = 2, cap: int = 12) -> list[dict]:
    """
    Starting from Chroma hit paragraphs, expand up to max_hops hops in the
    graph, capped at `cap` total paragraphs.

    Expansion: STATES → Principle → ←STATES (sibling paragraphs supporting
    the same principle), and CITES edges from each paragraph.
    """
    if not _graph_enabled() or not para_keys:
        return []
    client = _client()

    rows = client.read(
        """
        UNWIND $pkeys AS pk
        MATCH (seed:Paragraph {para_key: pk})
        // Hop 1: same principle
        OPTIONAL MATCH (seed)-[:STATES]->(p:Principle)<-[:STATES]-(sibling:Paragraph)
        WHERE sibling.para_key <> pk
        // Hop 2: cited cases' paragraphs
        OPTIONAL MATCH (seed)-[:CITES]->(cited:Case)<-[:HAS_PARAGRAPH]-(cited_para:Paragraph)
        WITH pk, collect(DISTINCT sibling)[0..4] AS siblings,
             collect(DISTINCT cited_para)[0..4] AS cited_paras
        RETURN pk AS seed_para_key,
               [x IN siblings | x.para_key] AS sibling_keys,
               [x IN cited_paras | x.para_key] AS cited_keys
        LIMIT $cap
        """,
        pkeys=para_keys,
        cap=cap,
    )

    seen: set[str] = set(para_keys)
    expanded: list[dict] = []

    for row in rows:
        seed_key = row["seed_para_key"]
        expanded.append({
            "para_key": seed_key,
            "path_explanation": "seed",
        })
        for sib in row.get("sibling_keys", []):
            if sib not in seen:
                seen.add(sib)
                expanded.append({
                    "para_key": sib,
                    "path_explanation": f"{seed_key} → STATES→Principle→STATES",
                })
        for cit in row.get("cited_keys", []):
            if cit not in seen and len(expanded) < cap:
                seen.add(cit)
                expanded.append({
                    "para_key": cit,
                    "path_explanation": f"{seed_key} → CITES→cited_para",
                })

    return expanded[:cap]
