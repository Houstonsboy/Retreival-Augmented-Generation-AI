"""Migration runner.

Applies numbered .cypher files in migrations/ exactly once, in order.
Applied versions are stored as :Migration nodes in the graph.

A migration whose checksum has changed since it was applied is refused —
write a new migration instead of editing an existing one.
"""
from __future__ import annotations

import hashlib
import logging
import pathlib
import re

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = pathlib.Path(__file__).parent
_FILENAME_RE = re.compile(r"^(\d{4})_(.+)\.cypher$")


def _load_files() -> list[tuple[int, str, pathlib.Path]]:
    """Return (version, name, path) sorted by version."""
    files = []
    for p in _MIGRATIONS_DIR.glob("*.cypher"):
        m = _FILENAME_RE.match(p.name)
        if m:
            files.append((int(m.group(1)), m.group(2), p))
    files.sort(key=lambda x: x[0])
    return files


def _checksum(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _statement_split(cypher: str) -> list[str]:
    """Split on statement-ending semicolons, skip blank/comment-only chunks.

    Semicolons that appear inside a // comment line are not statement
    terminators; strip comment lines first, then split on semicolons.
    """
    # Remove comment lines before splitting so that semicolons in comments
    # (e.g. "// foo[]; see bar") do not produce spurious empty statements.
    no_comments = "\n".join(
        line for line in cypher.splitlines() if not line.strip().startswith("//")
    )
    stmts = []
    for chunk in no_comments.split(";"):
        clean = chunk.strip()
        if clean:
            stmts.append(clean)
    return stmts


def run_pending(client) -> list[int]:
    """Apply all pending migrations; return list of newly applied version numbers."""
    _ensure_migration_node(client)
    applied = _applied_versions(client)
    new_versions = []

    for version, name, path in _load_files():
        checksum = _checksum(path)
        if version in applied:
            stored = applied[version]["checksum"]
            if stored != checksum:
                raise RuntimeError(
                    f"Migration {version:04d}_{name}.cypher has been modified "
                    f"(stored checksum {stored[:8]}… ≠ current {checksum[:8]}…). "
                    "Write a new migration instead of editing an applied one."
                )
            continue

        logger.info("Applying migration %04d: %s", version, name)
        for stmt in _statement_split(path.read_text(encoding="utf-8")):
            client.write(stmt)

        client.write(
            "CREATE (:Migration {version: $v, name: $n, checksum: $c, applied_at: datetime()})",
            v=version,
            n=name,
            c=checksum,
        )
        new_versions.append(version)
        logger.info("Migration %04d applied.", version)

    return new_versions


def status(client) -> dict:
    """Return {'applied': [...], 'pending': [...]}."""
    _ensure_migration_node(client)
    applied = _applied_versions(client)
    all_files = _load_files()

    applied_list = [
        {"version": v, "name": info["name"], "applied_at": info["applied_at"]}
        for v, info in sorted(applied.items())
    ]
    pending_list = [
        {"version": v, "name": n}
        for v, n, _ in all_files
        if v not in applied
    ]
    return {"applied": applied_list, "pending": pending_list}


def current_version(client) -> int | None:
    """Return the highest applied migration version, or None."""
    try:
        rows = client.read(
            "MATCH (m:Migration) RETURN max(m.version) AS v"
        )
        return rows[0]["v"] if rows else None
    except Exception:
        return None


def _ensure_migration_node(client):
    """Create the Migration uniqueness constraint if not yet present."""
    client.write(
        "CREATE CONSTRAINT migration_ver IF NOT EXISTS "
        "FOR (m:Migration) REQUIRE m.version IS UNIQUE"
    )


def _applied_versions(client) -> dict[int, dict]:
    rows = client.read(
        "MATCH (m:Migration) RETURN m.version AS v, m.name AS name, "
        "m.checksum AS checksum, toString(m.applied_at) AS applied_at"
    )
    return {r["v"]: r for r in rows}
