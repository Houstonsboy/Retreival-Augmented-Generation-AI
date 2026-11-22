"""Verification orchestration for the analyst result.

Runs the deterministic rules against the parsed :class:`AnalystResult`, the
masked ``analysis_input`` and (optionally) the case file, then assembles a
``verification`` block that is returned alongside the (possibly mutated) result.

This is the transparency layer: it shows whether every quote traced back to a
message, whether the payment table matches the parsed M-Pesa records, whether
statuses were corrected, and whether the verdict was softened.
"""

from __future__ import annotations

from typing import Any

from .models import AnalystResult, OverallVerdict
from .rules import (
    check_element_statuses,
    check_mpesa_cross_match,
    check_premature_breach,
    check_quotes_grounded,
    check_timeline,
    compute_verdict,
    merge_followups,
)


def verify(
    result: AnalystResult,
    analysis_input: dict[str, Any],
    case_file: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all rules, mutate ``result`` where required, and return a report."""
    integrity_flags = analysis_input.get("integrity_flags") or []
    date_checks = analysis_input.get("date_checks") or {}

    ungrounded = check_quotes_grounded(result, analysis_input)
    mpesa_cross_check = check_mpesa_cross_match(result, analysis_input)
    status_issues = check_element_statuses(result)
    timeline_unreferenced = check_timeline(result)
    premature_breach = check_premature_breach(result, date_checks)

    verdict_before = result.overall_verdict
    verdict_after, downgraded, reasons = compute_verdict(result, integrity_flags, date_checks)
    result.overall_verdict = verdict_after

    if case_file is not None:
        result.follow_up_requests = merge_followups(result, analysis_input, case_file)

    return {
        "quotes_grounded": not ungrounded,
        "ungrounded_quotes": ungrounded,
        "mpesa_cross_check": mpesa_cross_check,
        "status_issues": status_issues,
        "timeline_unreferenced": timeline_unreferenced,
        "premature_breach": premature_breach,
        "verdict_before": verdict_before.value,
        "verdict_after": verdict_after.value,
        "verdict_downgraded": downgraded,
        "downgrade_reasons": reasons,
    }


def effective_verdict(result: AnalystResult, verification: dict[str, Any]) -> OverallVerdict:
    """Return the verdict to present to the user after downgrades."""
    return OverallVerdict(verification["verdict_after"])
