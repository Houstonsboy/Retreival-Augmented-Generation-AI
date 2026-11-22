"""End-to-end contract-validity analysis pipeline.

Turns one or more OCR results (the output of :func:`SCCVision.service.extract_text`)
into a contract-validity analysis:

1. Merge multi-screenshot OCR results (``merge_ocr_results``).
2. Build the masked ``analysis_input`` (``build_analysis_input``).
3. Call the Analyst LLM and parse its result (``AnalystLLM``).
4. Run deterministic verification rules (``verify``).

Nothing here persists data or logs message text.
"""

from __future__ import annotations

from typing import Any

from . import PROMPT_VERSION
from .client import AnalystError, AnalystLLM
from .dates import compute_date_checks
from .enrich import build_analysis_input, build_analysis_input_from_case, merge_ocr_results
from .models import AnalystResult
from .verify import verify


def analyze(
    ocr_results: list[dict[str, Any]],
    case_context: dict[str, Any] | None = None,
    *,
    llm: AnalystLLM | None = None,
) -> dict[str, Any]:
    """Run the full analysis over one or more OCR results.

    ``ocr_results`` entries are the dicts returned by ``extract_text`` (each has
    ``contact``, ``messages`` and ``system_events``). ``case_context`` carries
    ``user_role`` and ``user_account``.
    """
    if not ocr_results:
        raise ValueError("At least one OCR result is required")

    cleaned = merge_ocr_results(ocr_results)
    image_count = len(ocr_results)
    analysis_input = build_analysis_input(cleaned, case_context, image_count)

    client = llm or AnalystLLM()
    result: AnalystResult = client.complete(analysis_input)

    verification = verify(result, analysis_input)

    return {
        "analysis_input": analysis_input,
        "result": result.model_dump(),
        "verification": verification,
        "prompt_version": PROMPT_VERSION,
        "model": client.model,
    }


def analyze_case(
    case_file: dict[str, Any],
    *,
    llm: AnalystLLM | None = None,
) -> dict[str, Any]:
    """Run the full analysis over a whole (frontend-held) case file.

    ``case_file`` is the versioned case-file object described in the input
    spec: ``case_context``, ``evidence``, ``user_provided_facts``,
    ``rejected_facts``, ``inferred_facts``, ``requests`` and ``analyses``.
    """
    if not case_file or not case_file.get("evidence"):
        raise ValueError("The case file must contain at least one evidence item")

    date_checks = compute_date_checks(case_file)
    analysis_input = build_analysis_input_from_case(case_file, date_checks)

    client = llm or AnalystLLM()
    result: AnalystResult = client.complete(analysis_input)

    verification = verify(result, analysis_input, case_file)

    return {
        "analysis_input": analysis_input,
        "result": result.model_dump(),
        "verification": verification,
        "prompt_version": PROMPT_VERSION,
        "model": client.model,
    }
