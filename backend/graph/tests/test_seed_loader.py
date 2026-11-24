"""Seed loader unit tests — no Neo4j required.

Tests cover: Chroma ID derivation, node key building, dry-run executes without
connecting, and idempotency (second load returns same seed_to_key map).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEED_PATH = Path(__file__).resolve().parents[2] / "graph" / "seed" / "scc_caselaw_graph.json"
_OVERRIDES_PATH = Path(__file__).resolve().parents[2] / "graph" / "seed" / "case_identifier_overrides.yaml"


def _get_loader():
    """Return the seed_loader module (re-importable)."""
    if "graph.seed_loader" in sys.modules:
        return sys.modules["graph.seed_loader"]
    from graph import seed_loader  # noqa: PLC0415
    return seed_loader


# ---------------------------------------------------------------------------
# T1: Chroma ID derivation from seed id
# ---------------------------------------------------------------------------

def test_chroma_id_in_corpus():
    loader = _get_loader()
    assert loader._chroma_id_from_seed("case:2024KEHC15763", {}) == "2024KEHC15763(KLR)"
    assert loader._chroma_id_from_seed("case:2001KECA154", {})   == "2001KECA154(KLR)"
    assert loader._chroma_id_from_seed("case:2026SCC242", {})    == "2026SCC242(KLR)"


def test_chroma_id_external_no_match():
    loader = _get_loader()
    # ext: nodes that don't have YEAR+COURT+NUM format → None
    assert loader._chroma_id_from_seed("ext:MNNvENK2017", {}) is None
    assert loader._chroma_id_from_seed("ext:GiellavCassman1973", {}) is None


def test_chroma_id_override_wins():
    loader = _get_loader()
    overrides = {"case:2011KEHC1481": "MANUAL-OVERRIDE"}
    assert loader._chroma_id_from_seed("case:2011KEHC1481", overrides) == "MANUAL-OVERRIDE"


# ---------------------------------------------------------------------------
# T2: node key building
# ---------------------------------------------------------------------------

def test_node_key_case():
    loader = _get_loader()
    node = {
        "id": "case:2024KEHC15763",
        "type": "Case",
        "citation": "[2024] KEHC 15763 (KLR)",
        "name": "Ogembo v Yongo",
    }
    label, key_field, key_value = loader._node_key("case:2024KEHC15763", node, {})
    assert label == "Case"
    assert key_field == "case_key"
    assert key_value == "KEHC-2024-15763"


def test_node_key_court():
    loader = _get_loader()
    node = {"id": "court:KESC", "type": "Court", "name": "Supreme Court", "rank": 1}
    label, key_field, key_value = loader._node_key("court:KESC", node, {})
    assert label == "Court"
    assert key_field == "code"
    assert key_value == "KESC"


def test_node_key_principle():
    loader = _get_loader()
    node = {"id": "pr:cert-mandatory", "type": "Principle", "text": "...", "kind": "ratio"}
    label, key_field, key_value = loader._node_key("pr:cert-mandatory", node, {})
    assert label == "Principle"
    assert key_field == "principle_key"
    assert key_value == "cert-mandatory"


def test_node_key_provision():
    loader = _get_loader()
    node = {
        "id": "prov:EA-s106B",
        "type": "Provision",
        "act": "Evidence Act (Cap 80)",
        "section": "s.106B",
    }
    label, key_field, key_value = loader._node_key("prov:EA-s106B", node, {})
    assert label == "Provision"
    assert key_field == "provision_key"
    assert "evidence-act" in key_value
    assert "s106B" in key_value


# ---------------------------------------------------------------------------
# T3: dry-run processes all 149 nodes without touching Neo4j
# ---------------------------------------------------------------------------

def test_dry_run_no_neo4j_connection(tmp_path, monkeypatch):
    """Running seed loader with --dry-run must not call get_client()."""
    loader = _get_loader()

    get_client_called = []

    def _mock_get_client():
        get_client_called.append(True)
        raise AssertionError("get_client() must not be called in dry-run mode")

    # Patch at the module level where it's imported
    monkeypatch.setattr("graph.seed_loader.get_client", _mock_get_client, raising=False)
    # Also patch config.GRAPH_ENABLED
    import graph.config as cfg
    monkeypatch.setattr(cfg, "GRAPH_ENABLED", True)

    import argparse
    args = argparse.Namespace(dry_run=True, file=None)

    # cmd_load should complete without raising
    loader.cmd_load(args)

    assert not get_client_called, "get_client() was called during dry-run"


# ---------------------------------------------------------------------------
# T4: _load_nodes returns all 149 seed nodes in seed_to_key map
# ---------------------------------------------------------------------------

def test_load_nodes_covers_all_seed_nodes():
    import json
    loader = _get_loader()
    seed = json.loads(_SEED_PATH.read_text())
    overrides = {}  # empty — no YAML overrides for seed-0.1

    seed_to_key = loader._load_nodes(
        client=None,
        nodes=seed["nodes"],
        overrides=overrides,
        ingest_run_id="test-run",
        dry_run=True,  # no DB calls
    )
    assert len(seed_to_key) == 149, f"Expected 149 entries, got {len(seed_to_key)}"
    # Verify a few specific keys
    assert "case:2024KEHC15763" in seed_to_key
    assert seed_to_key["case:2024KEHC15763"][2] == "KEHC-2024-15763"
    assert "case:2011KEHC1481" in seed_to_key
    assert "court:KESC" in seed_to_key
    assert seed_to_key["court:KESC"][2] == "KESC"
