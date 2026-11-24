"""Graph admin CLI.

Usage:
    python -m graph.admin init
    python -m graph.admin status
    python -m graph.admin migrate
    python -m graph.admin reset --confirm-wipe
"""
from __future__ import annotations

import argparse
import json
import sys


def _client():
    from .client import get_client
    return get_client()


def cmd_init(args):
    c = _client()
    print("Verifying connectivity … ok")

    from .migrations.runner import run_pending, status as mig_status
    new = run_pending(c)
    if new:
        print(f"Applied {len(new)} migration(s): {new}")
    else:
        print("No new migrations.")

    cmd_status(args)


def cmd_status(args):
    c = _client()
    from .migrations.runner import status as mig_status

    # Server info
    rows = c.read(
        "CALL dbms.components() YIELD name, versions, edition "
        "RETURN name, versions[0] AS version, edition LIMIT 1"
    )
    if rows:
        print(f"Server: Neo4j {rows[0].get('version')} ({rows[0].get('edition')})")

    # Migrations
    s = mig_status(c)
    print(f"\nMigrations applied: {len(s['applied'])}")
    for m in s["applied"]:
        print(f"  [{m['version']:04d}] {m['name']}  applied_at={m['applied_at']}")
    if s["pending"]:
        print(f"Migrations pending: {len(s['pending'])}")
        for m in s["pending"]:
            print(f"  [{m['version']:04d}] {m['name']}")
    else:
        print("No pending migrations.")

    # Node counts
    counts = c.read(
        "CALL db.labels() YIELD label "
        "RETURN label ORDER BY label"
    )
    print("\nNode counts by label:")
    if not counts:
        print("  (no labels)")
    for row in counts:
        label = row["label"]
        cnt = c.read(f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"]
        print(f"  {label}: {cnt}")


def cmd_migrate(args):
    c = _client()
    from .migrations.runner import run_pending
    new = run_pending(c)
    if new:
        print(f"Applied: {new}")
    else:
        print("Already up to date.")


def cmd_reset(args):
    from . import config as cfg
    if cfg.GRAPH_ENV == "prod":
        print("ERROR: reset is not allowed in GRAPH_ENV=prod.", file=sys.stderr)
        sys.exit(1)
    if not args.confirm_wipe:
        print("Pass --confirm-wipe to delete all data nodes.", file=sys.stderr)
        sys.exit(1)

    c = _client()
    from .repository import Neo4jRepository
    repo = Neo4jRepository(c)
    counts = repo.count_by_label()
    total = sum(counts.values())
    print(f"Will delete ~{total} data node(s): {json.dumps(counts, indent=2)}")
    repo.reset(confirm=True)
    print("Reset complete.")


def main():
    parser = argparse.ArgumentParser(prog="python -m graph.admin")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Verify connectivity and run pending migrations.")
    sub.add_parser("status", help="Show server info, migrations and node counts.")
    sub.add_parser("migrate", help="Run pending migrations only.")
    reset_p = sub.add_parser("reset", help="Delete all data nodes (dev/test only).")
    reset_p.add_argument("--confirm-wipe", action="store_true")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "init": cmd_init,
        "status": cmd_status,
        "migrate": cmd_migrate,
        "reset": cmd_reset,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
