"""CLI entry point: python -m ingest.caselaw run [options]

Usage:
    python -m ingest.caselaw run [--case KEY] [--dry-run] [--no-llm]
    python -m ingest.caselaw run --help
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure backend/ is on the path when run as a module from the project root
_backend = Path(__file__).resolve().parents[2]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_backend / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest.caselaw")


def cmd_run(args) -> None:
    from .pipeline import run_pipeline

    logger.info("Starting ingest pipeline — dry_run=%s no_llm=%s case=%s",
                args.dry_run, args.no_llm, args.case or "all")
    summary = run_pipeline(
        case_filter=args.case,
        dry_run=args.dry_run,
        no_llm=args.no_llm,
    )
    print("\n=== Ingest Summary ===")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ingest.caselaw",
        description="SCC case-law ingest pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run ingest pipeline")
    run_p.add_argument("--case", default=None,
                       help="Filter to a single case by partial case_key or filename")
    run_p.add_argument("--dry-run", action="store_true",
                       help="Parse and extract but do NOT write to Chroma or Neo4j")
    run_p.add_argument("--no-llm", action="store_true",
                       help="Skip LLM calls (regex/header extraction only)")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
