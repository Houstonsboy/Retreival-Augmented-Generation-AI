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


# ---------------------------------------------------------------------------
# expand_for_answer(case_identifiers, query_text, forum, cap)
# ---------------------------------------------------------------------------

def expand_for_answer(
    case_identifiers: list[str],
    query_text: str,
    forum: str = "SCC",
    cap: int = 5,
) -> list[dict]:
    """Return graph-authority dicts for anchor cases + query-time expansion.

    Parameters
    ----------
    case_identifiers : Chroma case_identifier strings from the retriever
        (e.g. ["2024KEHC15763(KLR)", "2014KEHC6468(KLR)"]).
    query_text : the user's query, used for full-text case-name expansion.
    forum : court code for authority-binding check (default "SCC").
    cap : maximum items to return.

    Returns
    -------
    [] when GRAPH_ENABLED=false.
    Otherwise a list (len ≤ cap) of dicts:
        case_key, case_name, chroma_case_identifier,
        principle_key, principle_text, kind,
        rel_type, court_code, is_binding,
        related_principles (list of {rel, key, text}),
        path_explanation
    Sorted: binding first, then ratio > obiter, then by court rank.
    """
    if not _graph_enabled():
        return []
    client = _client()
    from .authority import is_binding

    # -- Step 1: anchor by chroma_case_identifier (exact match) ---------------
    anchor_rows = client.read(
        """
        UNWIND $ids AS chroma_id
        MATCH (c:Case {chroma_case_identifier: chroma_id})
        RETURN c.case_key AS case_key
        """,
        ids=case_identifiers,
    )
    anchor_keys = [r["case_key"] for r in anchor_rows]

    # -- Step 2: full-text search for additional cases in query text ----------
    ft_keys: list[str] = []
    if query_text and query_text.strip():
        # Build a Lucene query from the text; use the first 128 chars
        ft_query = " ".join(
            w for w in query_text[:128].split() if len(w) >= 3
        )[:128]
        if ft_query:
            try:
                ft_rows = client.read(
                    "CALL db.index.fulltext.queryNodes('case_names_ft', $q) "
                    "YIELD node AS c, score "
                    "WHERE c.chroma_case_identifier IS NOT NULL "
                    "RETURN c.case_key AS case_key LIMIT 3",
                    q=ft_query,
                )
                ft_keys = [r["case_key"] for r in ft_rows]
            except Exception as exc:
                logger.debug("Full-text case search failed (non-fatal): %s", exc)

    all_keys = list(dict.fromkeys(anchor_keys + ft_keys))  # dedup, preserve order
    if not all_keys:
        return []

    # -- Step 3: collect principles (HOLDS / STATES_OBITER / via DECIDES+ANSWERS) --
    rows = client.read(
        """
        UNWIND $ck_list AS ck
        MATCH (c:Case {case_key: ck})
        MATCH (c)-[:DECIDED_IN]->(court:Court)
        // Direct principle assertions
        MATCH (c)-[r:HOLDS|STATES_OBITER]->(p:Principle)
        // Related principles (QUALIFIES / CONFLICTS_WITH)
        OPTIONAL MATCH (p)-[rel2:QUALIFIES|CONFLICTS_WITH]->(p2:Principle)
        WITH c, court, r, p,
             collect(DISTINCT {rel: type(rel2), key: p2.principle_key,
                               text: p2.text}) AS related
        RETURN
          c.case_key                   AS case_key,
          c.name                       AS case_name,
          c.chroma_case_identifier     AS chroma_case_identifier,
          court.code                   AS court_code,
          court.rank                   AS court_rank,
          type(r)                      AS rel_type,
          p.principle_key              AS principle_key,
          p.text                       AS principle_text,
          p.kind                       AS kind,
          related                      AS related_principles
        """,
        ck_list=all_keys,
    )

    # -- Step 4: also pull principles reached through Issue (DECIDES→ANSWERS) --
    via_issue_rows = client.read(
        """
        UNWIND $ck_list AS ck
        MATCH (c:Case {case_key: ck})
        MATCH (c)-[:DECIDED_IN]->(court:Court)
        MATCH (c)-[:DECIDES]->(iss:Issue)<-[:ANSWERS]-(p:Principle)
        OPTIONAL MATCH (p)-[rel2:QUALIFIES|CONFLICTS_WITH]->(p2:Principle)
        WITH c, court, p,
             collect(DISTINCT {rel: type(rel2), key: p2.principle_key,
                               text: p2.text}) AS related
        RETURN
          c.case_key                   AS case_key,
          c.name                       AS case_name,
          c.chroma_case_identifier     AS chroma_case_identifier,
          court.code                   AS court_code,
          court.rank                   AS court_rank,
          'ANSWERS'                    AS rel_type,
          p.principle_key              AS principle_key,
          p.text                       AS principle_text,
          p.kind                       AS kind,
          related                      AS related_principles
        """,
        ck_list=all_keys,
    )

    # -- Step 5: deduplicate on (case_key, principle_key) ---------------------
    seen: set[tuple] = set()
    combined: list[dict] = []
    for row in list(rows) + list(via_issue_rows):
        dedup_key = (row.get("case_key", ""), row.get("principle_key", ""))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        court_code = row.get("court_code", "")
        binding = is_binding(court_code, forum)
        kind = row.get("kind") or "obiter"
        related = [r for r in (row.get("related_principles") or []) if r.get("key")]

        combined.append({
            "case_key": row.get("case_key", ""),
            "case_name": row.get("case_name", ""),
            "chroma_case_identifier": row.get("chroma_case_identifier"),
            "principle_key": row.get("principle_key", ""),
            "principle_text": row.get("principle_text", ""),
            "kind": kind,
            "rel_type": row.get("rel_type", "HOLDS"),
            "court_code": court_code,
            "court_rank": row.get("court_rank", 99),
            "is_binding": binding,
            "related_principles": related,
            "path_explanation": (
                f"{row.get('case_key','')} -[{row.get('rel_type','HOLDS')}]-> "
                f"{row.get('principle_key','')} ({kind})"
                + (" [binding]" if binding else "")
            ),
        })

    # -- Step 6: rank and cap --------------------------------------------------
    def _rank_key(item: dict) -> tuple:
        return (
            0 if item["is_binding"] else 1,       # binding first
            0 if item["kind"] == "ratio" else 1,   # ratio before obiter
            item["court_rank"],                     # higher court (lower rank) first
        )

    combined.sort(key=_rank_key)
    return combined[:cap]
