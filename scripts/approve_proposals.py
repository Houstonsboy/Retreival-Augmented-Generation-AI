"""Approve, merge, or reject proposed Issues and Principles in review_queue.yaml.

Usage:
    python scripts/approve_proposals.py list
    python scripts/approve_proposals.py approve <id>
    python scripts/approve_proposals.py merge <id> <existing_id>
    python scripts/approve_proposals.py reject <id>
    python scripts/approve_proposals.py approve-all

Proposed nodes are visible in queries only when GRAPH_INCLUDE_PROPOSED=true (default false).
Approving a node updates its status in the graph (GRAPH_ENABLED must be true).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_backend))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_backend / ".env", override=True)

import yaml

_QUEUE_PATH = Path(__file__).resolve().parent.parent / "backend" / "ingest" / "caselaw" / "review_queue.yaml"
_ISSUES_PATH = Path(__file__).resolve().parent.parent / "backend" / "ingest" / "caselaw" / "issues.yaml"


def load_queue() -> list[dict]:
    if not _QUEUE_PATH.exists():
        return []
    with _QUEUE_PATH.open() as f:
        return yaml.safe_load(f) or []


def save_queue(items: list[dict]) -> None:
    with _QUEUE_PATH.open("w") as f:
        yaml.dump(items, f, allow_unicode=True, sort_keys=False)


def _graph_enabled() -> bool:
    import os
    return os.getenv("GRAPH_ENABLED", "false").lower() in ("1", "true", "yes")


def _graph_client():
    from graph.client import get_client
    return get_client()


def _update_graph_status(node_id: str, item_type: str, status: str, merged_into: str | None = None) -> None:
    if not _graph_enabled():
        print("  (GRAPH_ENABLED=false — graph not updated)")
        return
    client = _graph_client()
    if item_type == "issue":
        label, key = "Issue", "issue_key"
    else:
        label, key = "Principle", "principle_key"

    if merged_into:
        # Rewire edges from old ID to merged_into, then delete old node
        client.write(
            f"MATCH (old:{label} {{{key}: $oid}}) "
            f"MATCH (new:{label} {{{key}: $nid}}) "
            "MATCH (old)-[r]-() "
            "CALL apoc.refactor.to(r, new) YIELD input "
            "RETURN count(input)",
            oid=node_id, nid=merged_into,
        )
        client.write(
            f"MATCH (n:{label} {{{key}: $oid}}) DETACH DELETE n",
            oid=node_id,
        )
    else:
        client.write(
            f"MATCH (n:{label} {{{key}: $id}}) SET n.status = $status",
            id=node_id, status=status,
        )


def cmd_list(args) -> None:
    queue = load_queue()
    if not queue:
        print("review_queue.yaml is empty.")
        return
    print(f"{'ID':<45} {'TYPE':<12} {'STATUS':<12} {'DETAIL'}")
    print("-" * 100)
    for item in queue:
        iid = item.get("issue_id") or item.get("principle_id") or "?"
        itype = item.get("type", "?")
        status = item.get("status", "proposed")
        detail = item.get("question") or item.get("text", "")
        print(f"{iid:<45} {itype:<12} {status:<12} {detail[:40]}")


def cmd_approve(args) -> None:
    queue = load_queue()
    target = args.id
    found = False
    for item in queue:
        iid = item.get("issue_id") or item.get("principle_id")
        if iid == target:
            item["status"] = "approved"
            found = True
            _update_graph_status(target, item.get("type", "principle"), "approved")
            # If it's an issue, also add it to issues.yaml
            if item.get("type") == "issue":
                _add_to_issues_yaml(item)
            print(f"Approved: {target}")
            break
    if not found:
        print(f"Not found in queue: {target}")
        sys.exit(1)
    save_queue(queue)


def cmd_merge(args) -> None:
    queue = load_queue()
    old_id = args.id
    new_id = args.target_id
    found = False
    for item in queue:
        iid = item.get("issue_id") or item.get("principle_id")
        if iid == old_id:
            item["status"] = f"merged_into:{new_id}"
            found = True
            _update_graph_status(old_id, item.get("type", "principle"), "merged", merged_into=new_id)
            print(f"Merged {old_id} → {new_id}")
            break
    if not found:
        print(f"Not found in queue: {old_id}")
        sys.exit(1)
    save_queue(queue)


def cmd_reject(args) -> None:
    queue = load_queue()
    target = args.id
    found = False
    for item in queue:
        iid = item.get("issue_id") or item.get("principle_id")
        if iid == target:
            item["status"] = "rejected"
            found = True
            _update_graph_status(target, item.get("type", "principle"), "rejected")
            print(f"Rejected: {target}")
            break
    if not found:
        print(f"Not found in queue: {target}")
        sys.exit(1)
    save_queue(queue)


def cmd_approve_all(args) -> None:
    queue = load_queue()
    count = 0
    for item in queue:
        if item.get("status") == "proposed":
            iid = item.get("issue_id") or item.get("principle_id")
            item["status"] = "approved"
            _update_graph_status(iid, item.get("type", "principle"), "approved")
            count += 1
    save_queue(queue)
    print(f"Approved {count} item(s).")


def _add_to_issues_yaml(item: dict) -> None:
    with _ISSUES_PATH.open() as f:
        data = yaml.safe_load(f) or {}
    issues = data.get("issues", [])
    new_issue = {
        "issue_id": item.get("issue_id"),
        "question": item.get("question", ""),
        "domain": "General",
        "status": "approved",
    }
    if not any(i.get("issue_id") == new_issue["issue_id"] for i in issues):
        issues.append(new_issue)
        data["issues"] = issues
        with _ISSUES_PATH.open("w") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="approve_proposals.py")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all items in the review queue")

    approve_p = sub.add_parser("approve", help="Approve a proposed item")
    approve_p.add_argument("id", help="issue_id or principle_id")

    merge_p = sub.add_parser("merge", help="Merge proposed item into an existing one")
    merge_p.add_argument("id", help="Proposed item ID")
    merge_p.add_argument("target_id", help="Existing item ID to merge into")

    reject_p = sub.add_parser("reject", help="Reject a proposed item")
    reject_p.add_argument("id", help="issue_id or principle_id")

    sub.add_parser("approve-all", help="Approve all pending proposed items")

    args = parser.parse_args()
    dispatch = {
        "list": cmd_list,
        "approve": cmd_approve,
        "merge": cmd_merge,
        "reject": cmd_reject,
        "approve-all": cmd_approve_all,
    }
    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
