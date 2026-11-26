"""Unit tests for expand_for_answer() — no Neo4j required.

Three tests:
  T1: returns [] when GRAPH_ENABLED=false
  T2: returns correctly-shaped dicts when graph is stubbed
  T3: QUALIFIES/CONFLICTS_WITH relations appear in related_principles
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# T1: returns [] when GRAPH_ENABLED is false
# ---------------------------------------------------------------------------

def test_expand_returns_empty_when_graph_disabled(monkeypatch):
    import graph.config as cfg
    monkeypatch.setattr(cfg, "GRAPH_ENABLED", False)

    from graph.queries import expand_for_answer
    result = expand_for_answer(["2024KEHC15763(KLR)"], "electronic evidence certificate")
    assert result == []


# ---------------------------------------------------------------------------
# Shared stub helpers
# ---------------------------------------------------------------------------

def _make_client_stub(anchor_rows, via_issue_rows=None, ft_rows=None):
    """Return a mock client where read() dispatches by query keyword."""
    client = MagicMock()
    via_issue_rows = via_issue_rows or []
    ft_rows = ft_rows or []

    def _read_side_effect(query, **params):
        q = query.strip()
        if "chroma_case_identifier: chroma_id" in q:
            return anchor_rows
        if "db.index.fulltext.queryNodes" in q:
            return ft_rows
        if "DECIDES" in q and "ANSWERS" in q:
            return via_issue_rows
        # HOLDS / STATES_OBITER query
        if "HOLDS|STATES_OBITER" in q or "HOLDS" in q:
            return []  # default: no principles unless overridden
        return []

    client.read.side_effect = _read_side_effect
    return client


# ---------------------------------------------------------------------------
# T2: returns correctly-shaped dicts
# ---------------------------------------------------------------------------

def test_expand_returns_shaped_dicts(monkeypatch):
    import graph.config as cfg
    monkeypatch.setattr(cfg, "GRAPH_ENABLED", True)

    _ANCHOR = [{"case_key": "KEHC-2024-15763"}]
    _HOLDS_ROWS = [
        {
            "case_key": "KEHC-2024-15763",
            "case_name": "Ogembo v Yongo",
            "chroma_case_identifier": "2024KEHC15763(KLR)",
            "court_code": "KEHC",
            "court_rank": 3,
            "rel_type": "HOLDS",
            "principle_key": "cert-mandatory",
            "principle_text": "Electronic evidence is inadmissible without a s.106B(4) certificate.",
            "kind": "ratio",
            "related_principles": [],
        }
    ]

    client = MagicMock()

    def _read(query, **params):
        q = query.strip()
        if "chroma_case_identifier: chroma_id" in q:
            return _ANCHOR
        if "db.index.fulltext.queryNodes" in q:
            return []
        if "DECIDES" in q:
            return []
        return _HOLDS_ROWS

    client.read.side_effect = _read

    with patch("graph.queries._client", return_value=client), \
         patch("graph.authority.is_binding", return_value=True):
        from graph.queries import expand_for_answer
        result = expand_for_answer(["2024KEHC15763(KLR)"], "certificate electronic evidence")

    assert len(result) == 1
    item = result[0]
    # Required keys
    for key in ("case_key", "case_name", "chroma_case_identifier",
                "principle_key", "principle_text", "kind",
                "rel_type", "court_code", "is_binding",
                "related_principles", "path_explanation"):
        assert key in item, f"Missing key: {key}"

    assert item["case_key"] == "KEHC-2024-15763"
    assert item["principle_key"] == "cert-mandatory"
    assert item["chroma_case_identifier"] == "2024KEHC15763(KLR)"
    assert item["kind"] == "ratio"
    assert item["is_binding"] is True
    assert "KEHC-2024-15763" in item["path_explanation"]


# ---------------------------------------------------------------------------
# T3: related_principles carries QUALIFIES/CONFLICTS_WITH entries
# ---------------------------------------------------------------------------

def test_expand_includes_related_principles(monkeypatch):
    import graph.config as cfg
    monkeypatch.setattr(cfg, "GRAPH_ENABLED", True)

    _ANCHOR = [{"case_key": "SCC-2026-235"}]
    _HOLDS_ROWS = [
        {
            "case_key": "SCC-2026-235",
            "case_name": "New Titanic Jacaranda General Spares v Omil",
            "chroma_case_identifier": "2026SCC235(KLR)",
            "court_code": "SCC",
            "court_rank": 1,
            "rel_type": "HOLDS",
            "principle_key": "scc-not-strict-evidence",
            "principle_text": "SCC is not strictly bound by evidence rules.",
            "kind": "ratio",
            "related_principles": [
                {"rel": "QUALIFIES", "key": "cert-mandatory",
                 "text": "Electronic evidence is inadmissible without a s.106B(4) certificate."},
            ],
        }
    ]

    client = MagicMock()

    def _read(query, **params):
        q = query.strip()
        if "chroma_case_identifier: chroma_id" in q:
            return _ANCHOR
        if "db.index.fulltext.queryNodes" in q:
            return []
        if "DECIDES" in q:
            return []
        return _HOLDS_ROWS

    client.read.side_effect = _read

    with patch("graph.queries._client", return_value=client), \
         patch("graph.authority.is_binding", return_value=True):
        from graph.queries import expand_for_answer
        result = expand_for_answer(["2026SCC235(KLR)"], "electronic evidence SCC")

    assert result, "Expected at least one result"
    item = result[0]
    related = item["related_principles"]
    assert len(related) >= 1
    quals = [r for r in related if r.get("rel") == "QUALIFIES"]
    assert quals, f"Expected a QUALIFIES entry in related_principles, got: {related}"
    assert quals[0]["key"] == "cert-mandatory"
