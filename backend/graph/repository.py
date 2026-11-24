"""Graph repository — abstract interface and Neo4j implementation.

Upserts are implemented but NOT called from the RAG pipeline yet (GRAPH_ENABLED=false).
They are tested against the test container only.
"""
from __future__ import annotations

import abc
import logging
from typing import Any

from .models import GraphEdge

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500


class GraphRepository(abc.ABC):
    @abc.abstractmethod
    def upsert_node(self, label: str, key_field: str, node: dict) -> None:
        """MERGE on key_field, SET all other properties."""

    @abc.abstractmethod
    def upsert_edge(self, edge: GraphEdge) -> None:
        """MERGE the relationship between two nodes identified by their unique keys."""

    @abc.abstractmethod
    def get_case(self, case_key: str) -> dict | None:
        """Return the Case node dict or None."""

    @abc.abstractmethod
    def neighbors(
        self,
        key: str,
        rel_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict]:
        """Return neighboring nodes up to `depth` hops (max 2)."""

    @abc.abstractmethod
    def count_by_label(self) -> dict[str, int]:
        """Return {label: count} for all node labels (excluding :Migration)."""


class Neo4jRepository(GraphRepository):
    def __init__(self, client):
        self._c = client

    def upsert_node(self, label: str, key_field: str, node: dict) -> None:
        # Flatten nested provenance dict into top-level props with provenance_ prefix
        flat = _flatten(node)
        key_val = flat[key_field]

        self._c.write(
            f"MERGE (n:{label} {{{key_field}: $key_val}}) SET n += $props",
            key_val=key_val,
            props=flat,
        )

    def upsert_nodes_batch(
        self, label: str, key_field: str, nodes: list[dict]
    ) -> None:
        """Batch upsert using UNWIND for efficiency (batch_size=500)."""
        for i in range(0, len(nodes), _BATCH_SIZE):
            batch = [_flatten(n) for n in nodes[i : i + _BATCH_SIZE]]
            self._c.write(
                f"UNWIND $batch AS props "
                f"MERGE (n:{label} {{{key_field}: props.{key_field}}}) "
                f"SET n += props",
                batch=batch,
            )

    def upsert_edge(self, edge: GraphEdge) -> None:
        """MERGE a typed relationship. Requires that src and dst nodes exist first."""
        rel = edge.type.value
        props = edge.props or {}
        self._c.write(
            "MATCH (a {case_key: $src}) "
            "MATCH (b {case_key: $dst}) "
            f"MERGE (a)-[r:{rel}]->(b) "
            "SET r += $props",
            src=edge.src_key,
            dst=edge.dst_key,
            props=props,
        )

    def get_case(self, case_key: str) -> dict | None:
        rows = self._c.read(
            "MATCH (n:Case {case_key: $k}) RETURN n",
            k=case_key,
        )
        if not rows:
            return None
        return dict(rows[0]["n"])

    def neighbors(
        self,
        key: str,
        rel_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[dict]:
        if depth > 2:
            raise ValueError("neighbors() supports depth ≤ 2")
        depth = max(1, depth)
        rel_filter = "|".join(rel_types) if rel_types else ""
        rel_pattern = f"[:{rel_filter}*1..{depth}]" if rel_filter else f"[*1..{depth}]"
        rows = self._c.read(
            f"MATCH (src {{case_key: $k}})-{rel_pattern}-(n) "
            "RETURN DISTINCT n LIMIT 100",
            k=key,
        )
        return [dict(r["n"]) for r in rows]

    def count_by_label(self) -> dict[str, int]:
        rows = self._c.read(
            "CALL db.labels() YIELD label "
            "WHERE label <> 'Migration' "
            "CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) AS c', {}) "
            "YIELD value RETURN label, value.c AS cnt"
        )
        return {r["label"]: r["cnt"] for r in rows}

    def reset(self, confirm: bool = False) -> int:
        """Delete all non-Migration nodes. Dev/test only."""
        from . import config as cfg
        if cfg.GRAPH_ENV == "prod":
            raise RuntimeError("reset() is not allowed in production.")
        if not confirm:
            raise ValueError("Pass confirm=True to wipe the graph.")

        counts = self.count_by_label()
        total = sum(counts.values())
        logger.warning("Resetting graph: deleting ~%d data nodes.", total)

        self._c.write(
            "CALL { MATCH (n) WHERE NOT n:Migration DETACH DELETE n } "
            "IN TRANSACTIONS OF 10000 ROWS"
        )
        return total


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dicts; provenance sub-dict gets provenance_ prefix."""
    out = {}
    for k, v in d.items():
        full_key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, f"{full_key}_"))
        elif isinstance(v, list) and all(not isinstance(i, dict) for i in v):
            out[full_key] = v
        elif not isinstance(v, (dict, list)):
            out[full_key] = v
    return out
