#!/usr/bin/env python3
"""Determinism harness for SCC analysis.

Runs the same case N times (force_rerun=True) and asserts that:
  AC1: confidence values are identical across runs (deterministic formula)
  AC2: cert_eligible is identical across runs
  AC3: verdict is identical, or explains differences via unstable_fields
  AC8: calls_used ≤ 2 on every run

Usage:
    python scripts/determinism_harness.py [case_file.json ...] [options]

Options:
    --cases PATH [...]   One or more case JSON files to test.
    --reruns N           Number of re-runs per case (default: 5).
    --end-to-end         Hit the Flask /sccvision/case/analyze HTTP endpoint
                         instead of calling analyze_case() directly.
    --url URL            Base URL for --end-to-end (default: http://localhost:5000)
    --token TOKEN        Bearer token for --end-to-end.
    --out PATH           Write CSV report to this file (default: stdout).
    --max-calls N        Maximum allowed calls_used (default: 2, per AC8).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_CONF_TOL = 0.0  # AC1: fully deterministic (must be identical, not just within ±0.05)


def _load_case(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Direct (in-process) runner
# ---------------------------------------------------------------------------

def _run_direct(case_file: dict, reruns: int, max_calls: int) -> list[dict]:
    from SCCVision.analyst.pipeline import analyze_case

    results = []
    for run_idx in range(reruns):
        t0 = time.monotonic()
        try:
            r = analyze_case(case_file, force_rerun=True, uid=None)
        except Exception as exc:
            results.append({"run": run_idx + 1, "error": str(exc)})
            continue
        elapsed = time.monotonic() - t0
        results.append({
            "run": run_idx + 1,
            "confidence": r.get("confidence"),
            "cert_eligible": (r.get("cert_eligibility") or {}).get("eligible"),
            "verdict": r.get("verdict"),
            "calls_used": r.get("calls_used"),
            "trigger_fired": r.get("trigger_fired"),
            "unstable_fields": sorted(r.get("unstable_fields") or []),
            "integrity_rating": r.get("integrity_rating"),
            "elapsed_s": round(elapsed, 2),
        })
    return results


# ---------------------------------------------------------------------------
# HTTP (end-to-end) runner
# ---------------------------------------------------------------------------

def _run_http(case_file: dict, reruns: int, max_calls: int, url: str, token: str) -> list[dict]:
    import urllib.request
    import urllib.error

    endpoint = f"{url.rstrip('/')}/sccvision/case/analyze"
    body_base = json.dumps({**case_file, "force_rerun": True}).encode("utf-8")

    results = []
    for run_idx in range(reruns):
        req = urllib.request.Request(
            endpoint,
            data=body_base,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                r = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            results.append({"run": run_idx + 1, "error": f"HTTP {exc.code}: {exc.reason}"})
            continue
        except Exception as exc:
            results.append({"run": run_idx + 1, "error": str(exc)})
            continue
        elapsed = time.monotonic() - t0
        results.append({
            "run": run_idx + 1,
            "confidence": r.get("confidence"),
            "cert_eligible": (r.get("cert_eligibility") or {}).get("eligible"),
            "verdict": r.get("verdict"),
            "calls_used": r.get("calls_used"),
            "trigger_fired": r.get("trigger_fired"),
            "unstable_fields": sorted(r.get("unstable_fields") or []),
            "integrity_rating": r.get("integrity_rating"),
            "elapsed_s": round(elapsed, 2),
        })
    return results


# ---------------------------------------------------------------------------
# Check assertions
# ---------------------------------------------------------------------------

def _check_ac(results: list[dict], max_calls: int) -> list[str]:
    """Return list of assertion failures (empty = all pass)."""
    failures = []
    good = [r for r in results if "error" not in r]
    if not good:
        failures.append("All runs errored — cannot check assertions")
        return failures

    confs = [r["confidence"] for r in good if r.get("confidence") is not None]
    eligibles = [r["cert_eligible"] for r in good]
    verdicts = [r["verdict"] for r in good]
    calls = [r["calls_used"] for r in good if r.get("calls_used") is not None]

    # AC1: confidence identical
    if len(set(confs)) > 1:
        failures.append(
            f"AC1 FAIL: confidence not identical across runs: {confs}"
        )

    # AC2: cert_eligible identical
    if len(set(str(e) for e in eligibles)) > 1:
        failures.append(
            f"AC2 FAIL: cert_eligible differs across runs: {eligibles}"
        )

    # AC3: verdict identical (or all unstable_fields non-empty when it differs)
    if len(set(verdicts)) > 1:
        # Allowed only if all runs have unstable_fields
        all_have_unstable = all(r.get("unstable_fields") for r in good)
        if not all_have_unstable:
            failures.append(
                f"AC3 FAIL: verdict differs across runs with no unstable_fields: {verdicts}"
            )
        else:
            failures.append(
                f"AC3 WARN: verdict differs across runs but all have unstable_fields: {verdicts}"
            )

    # AC8: calls_used ≤ max_calls
    over = [c for c in calls if c > max_calls]
    if over:
        failures.append(
            f"AC8 FAIL: calls_used exceeded {max_calls} on {len(over)} run(s): {over}"
        )

    return failures


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _report_case(case_name: str, results: list[dict], failures: list[str]) -> str:
    lines = [f"\n## Case: {case_name}"]
    for r in results:
        if "error" in r:
            lines.append(f"  Run {r['run']}: ERROR — {r['error']}")
        else:
            lines.append(
                f"  Run {r['run']}: confidence={r['confidence']}, "
                f"eligible={r['cert_eligible']}, verdict={r['verdict']!r}, "
                f"calls={r['calls_used']}, trigger={r['trigger_fired']}, "
                f"elapsed={r['elapsed_s']}s"
            )
    if failures:
        lines.append("  FAILURES:")
        for f in failures:
            lines.append(f"    ✗ {f}")
    else:
        lines.append("  PASS — all AC1/AC2/AC3/AC8 assertions met.")
    return "\n".join(lines)


def _write_csv(all_rows: list[dict], out_path: str | None) -> None:
    fieldnames = [
        "case", "run", "confidence", "cert_eligible", "verdict",
        "calls_used", "trigger_fired", "integrity_rating", "elapsed_s",
        "unstable_fields", "error", "ac_pass",
    ]
    fh = open(out_path, "w", newline="", encoding="utf-8") if out_path else sys.stdout
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in all_rows:
        writer.writerow(row)
    if out_path and fh is not sys.stdout:
        fh.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="SCC Analysis Determinism Harness")
    parser.add_argument("cases", nargs="*", help="Case JSON file paths")
    parser.add_argument("--cases", dest="cases_flag", nargs="+",
                        help="Case JSON file paths (alternative)")
    parser.add_argument("--reruns", type=int, default=5, help="Re-runs per case")
    parser.add_argument("--end-to-end", action="store_true",
                        help="Use HTTP endpoint instead of direct call")
    parser.add_argument("--url", default="http://localhost:5000",
                        help="Base URL for --end-to-end")
    parser.add_argument("--token", default=os.environ.get("SCC_TEST_TOKEN", ""),
                        help="Bearer token for --end-to-end")
    parser.add_argument("--out", help="Write CSV report to this file")
    parser.add_argument("--max-calls", type=int, default=2,
                        help="Max allowed Groq calls per run (AC8)")
    args = parser.parse_args()

    case_paths = list(args.cases or []) + list(args.cases_flag or [])
    if not case_paths:
        print("ERROR: provide at least one case JSON file.\n")
        parser.print_help()
        return 1

    all_csv_rows: list[dict] = []
    total_failures = 0

    for path in case_paths:
        case_name = Path(path).stem
        print(f"\nTesting: {path} ({args.reruns} reruns) …", flush=True)

        try:
            case_file = _load_case(path)
        except Exception as exc:
            print(f"  ERROR loading case file: {exc}")
            total_failures += 1
            continue

        if args.end_to_end:
            if not args.token:
                print("  ERROR: --token is required for --end-to-end mode")
                total_failures += 1
                continue
            results = _run_http(case_file, args.reruns, args.max_calls,
                                 args.url, args.token)
        else:
            results = _run_direct(case_file, args.reruns, args.max_calls)

        failures = _check_ac(results, args.max_calls)
        total_failures += sum(1 for f in failures if "FAIL" in f)

        print(_report_case(case_name, results, failures))

        # Accumulate CSV rows
        for r in results:
            all_csv_rows.append({
                "case": case_name,
                "ac_pass": "PASS" if not any("FAIL" in f for f in failures) else "FAIL",
                **r,
            })

    if args.out:
        _write_csv(all_csv_rows, args.out)
        print(f"\nCSV report written to: {args.out}")
    elif all_csv_rows:
        print("\n--- CSV ---")
        _write_csv(all_csv_rows, None)

    summary = (
        f"\nSummary: {len(case_paths)} case(s) tested, "
        f"{total_failures} assertion failure(s)."
    )
    print(summary)
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
