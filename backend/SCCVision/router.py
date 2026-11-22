"""Flask Blueprint: SCCVision OCR and contract-validity analysis endpoints."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from flask import Blueprint, g, jsonify, request
from requests import HTTPError

from auth import require_auth
from groq_client import GroqPayloadTooLargeError, GroqRateLimitError, call_groq
from .analysis_store import save_analysis_snapshot
from .analyst.client import AnalystError
from .analyst.dates import compute_date_checks, get_timezone, parse_iso, validate_user_fact_dates
from .analyst.mpssa import classify_evidence, parse_mpesa_screenshot
from .analyst.followup_catalog import compute_round_diff
from .analyst.pipeline import analyze, analyze_case
from .analyst.versions import PROMPT_VERSION, RULES_VERSION, VERIFY_PROMPT_VERSION
from .engine import get_engine
from .preprocess import InvalidImageError, PreprocessOptions
from .service import extract_text

sccvision_bp = Blueprint("sccvision", __name__)

ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB

# OCR is CPU-bound; cap concurrency so a burst of uploads cannot overwhelm
# the process. Each request still runs off the Flask worker thread.
_executor = ThreadPoolExecutor(max_workers=2)

# Startup self-check: log version constants on import so server logs confirm 2.2.0
import logging as _logging
_logging.getLogger(__name__).info(
    "SCCVision router loaded — PROMPT_VERSION=%s VERIFY_PROMPT_VERSION=%s RULES_VERSION=%s",
    PROMPT_VERSION, VERIFY_PROMPT_VERSION, RULES_VERSION,
)


@sccvision_bp.route("/sccvision/health", methods=["GET"])
def health():
    engine = get_engine()
    return jsonify(
        {
            "status": "ok",
            "module": "SCCVision",
            "engine": "rapidocr",
            "loaded": engine is not None,
            "prompt_version": PROMPT_VERSION,
            "verify_prompt_version": VERIFY_PROMPT_VERSION,
            "rules_version": RULES_VERSION,
        }
    )


@sccvision_bp.route("/sccvision/extract", methods=["POST"])
@require_auth
def extract():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    upload = request.files["file"]
    if not upload or not upload.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
    if ext not in ALLOWED_EXTS:
        return jsonify(
            {"error": f"Unsupported file type '.{ext}'. Allowed: png, jpg, jpeg, webp"}
        ), 400

    data = upload.read()
    if not data:
        return jsonify({"error": "Empty file"}), 400
    if len(data) > MAX_BYTES:
        return jsonify({"error": "File too large. Maximum size is 10 MB"}), 413

    server_sha256 = hashlib.sha256(data).hexdigest()

    options = PreprocessOptions(
        upscale=_bool_flag("upscale", default=True),
        invert=_bool_flag("invert", default=True),
    )

    try:
        future = _executor.submit(extract_text, data, options, upload.filename)
        result = future.result(timeout=120)
    except InvalidImageError:
        return jsonify(
            {"error": "Could not decode image. Provide a valid PNG, JPG, or WEBP"}
        ), 400
    except Exception:
        return jsonify({"error": "OCR failed"}), 500

    result["server_sha256"] = server_sha256
    return jsonify(result), 200


@sccvision_bp.route("/sccvision/analyze", methods=["POST"])
@require_auth
def analyze_upload():
    """OCR one or more screenshots, then run the contract-validity analysis.

    Accepts multipart files under ``files`` (or a single ``file``) and an
    optional ``case_context`` form field / query param containing JSON:
    ``{"user_role": "claimant|respondent|unknown", "user_account": "..."}``.
    """
    uploads = request.files.getlist("files") or request.files.getlist("file")
    if not uploads:
        return jsonify({"error": "No files provided"}), 400

    if any(not u or not u.filename for u in uploads):
        return jsonify({"error": "No file selected"}), 400

    for upload in uploads:
        ext = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
        if ext not in ALLOWED_EXTS:
            return jsonify(
                {"error": f"Unsupported file type '.{ext}'. Allowed: png, jpg, jpeg, webp"}
            ), 400
        upload.seek(0, 2)
        if upload.tell() > MAX_BYTES:
            return jsonify({"error": "File too large. Maximum size is 10 MB"}), 413
        upload.seek(0)

    case_context = _parse_case_context(request.form.get("case_context") or request.args.get("case_context"))

    options = PreprocessOptions(
        upscale=_bool_flag("upscale", default=True),
        invert=_bool_flag("invert", default=True),
    )

    def _extract(upload):
        data = upload.read()
        if not data:
            raise ValueError("Empty file")
        return extract_text(data, options, upload.filename)

    try:
        futures = [_executor.submit(_extract, u) for u in uploads]
        ocr_results = [f.result(timeout=120) for f in futures]
        analysis = analyze(ocr_results, case_context)
    except InvalidImageError:
        return jsonify(
            {"error": "Could not decode image. Provide a valid PNG, JPG, or WEBP"}
        ), 400
    except AnalystError as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 502
    except GroqPayloadTooLargeError as exc:
        return jsonify({"error": str(exc)}), 413
    except GroqRateLimitError as exc:
        return jsonify({"error": str(exc), "retry_after_seconds": 30}), 429
    except HTTPError as exc:
        return jsonify({"error": f"Upstream model error: {exc}"}), 502
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "OCR or analysis failed"}), 500

    resp = jsonify(analysis)
    resp.headers["Deprecation"] = "true"
    resp.headers["Warning"] = '299 - "deprecated; use /sccvision/evidence/process and /sccvision/case/analyze"'
    return resp, 200


@sccvision_bp.route("/sccvision/analyze-messages", methods=["POST"])
@require_auth
def analyze_messages():
    """Run analysis over already-cleaned OCR results (no image upload).

    Body: ``{"ocr_results": [<extract_text output>...], "case_context": {...}}``.
    Useful for re-running analysis over previously extracted conversations.
    """
    data = request.get_json(silent=True) or {}
    ocr_results = data.get("ocr_results")
    if not isinstance(ocr_results, list) or not ocr_results:
        return jsonify({"error": "ocr_results must be a non-empty list"}), 400

    case_context = data.get("case_context") or {}

    try:
        analysis = analyze(ocr_results, case_context)
    except AnalystError as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 502
    except GroqPayloadTooLargeError as exc:
        return jsonify({"error": str(exc)}), 413
    except GroqRateLimitError as exc:
        return jsonify({"error": str(exc), "retry_after_seconds": 30}), 429
    except HTTPError as exc:
        return jsonify({"error": f"Upstream model error: {exc}"}), 502
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Analysis failed"}), 500

    resp = jsonify(analysis)
    resp.headers["Deprecation"] = "true"
    resp.headers["Warning"] = '299 - "deprecated; use /sccvision/evidence/process and /sccvision/case/analyze"'
    return resp, 200


@sccvision_bp.route("/sccvision/evidence/process", methods=["POST"])
@require_auth
def evidence_process():
    """Classify and normalize an OCR result into a case-file evidence item.

    Accepts the output of ``/sccvision/extract`` (no image bytes) plus optional
    ``classification``, ``evidence_id``, ``file_name`` and ``sha256``. For
    ``mpesa_screenshot`` evidence the deterministic parser runs and fills
    ``parsed``; for chat evidence the contact/messages/system_events are stored.
    """
    data = request.get_json(silent=True) or {}
    ocr = data.get("ocr_result")
    if not isinstance(ocr, dict):
        return jsonify({"error": "ocr_result must be an object"}), 400

    requested = data.get("classification") or "auto"
    allowed = {"chat_screenshot", "mpesa_screenshot", "other"}
    etype = requested if requested in allowed else classify_evidence(ocr)

    evidence_id = data.get("evidence_id") or uuid.uuid4().hex
    file_name = (
        data.get("file_name")
        or (ocr.get("meta") or {}).get("file_name")
        or ""
    )

    client_sha256 = data.get("sha256")
    server_sha256 = data.get("server_sha256") or (ocr.get("server_sha256") if isinstance(ocr, dict) else None)
    hash_match = (
        bool(client_sha256 and server_sha256 and client_sha256 == server_sha256)
    )

    item = {
        "evidence_id":  evidence_id,
        "type":         etype,
        "file_name":    file_name,
        "sha256":       client_sha256,
        "server_sha256": server_sha256,
        "hash_match":   hash_match,
        "added_at":     datetime.now(get_timezone()).isoformat(),
        "ocr":          None,
        "parsed":       None,
        "user_edits":   {},
    }

    if etype == "mpesa_screenshot":
        parsed = parse_mpesa_screenshot(ocr, evidence_id)
        item["parsed"] = parsed.get("entries") or []
        item["parse_layout"] = parsed.get("layout")
        if parsed.get("layout") == "unsupported_layout":
            item["raw_text"] = parsed.get("raw_text")
    elif etype == "chat_screenshot":
        item["ocr"] = {
            "contact": ocr.get("contact") or {},
            "messages": ocr.get("messages") or [],
            "system_events": ocr.get("system_events") or [],
        }
    else:
        item["ocr"] = {"full_text": ocr.get("full_text") or ""}

    return jsonify({"evidence": item}), 200


def _slim_case_file(case_file: dict) -> dict:
    """Strip display-only and large fields before analysis.

    Evidence items: drop ``parse_layout`` and ``raw_text`` (UI-only OCR data).
    Past analyses: drop ``analysis_input`` and ``verification`` — only the
    three-field summary from ``_previous_analysis`` is used by the pipeline,
    so storing the full Groq input/output from prior runs is pure payload waste.
    """
    _DROP_EV = frozenset({"parse_layout", "raw_text"})
    _DROP_AN = frozenset({"analysis_input", "verification"})
    slimmed = {k: v for k, v in case_file.items() if k not in ("evidence", "analyses")}
    slimmed["evidence"] = [
        {k: v for k, v in ev.items() if k not in _DROP_EV}
        for ev in (case_file.get("evidence") or [])
    ]
    slimmed["analyses"] = [
        {k: v for k, v in an.items() if k not in _DROP_AN}
        for an in (case_file.get("analyses") or [])
    ]
    return slimmed


@sccvision_bp.route("/sccvision/case/analyze", methods=["POST"])
@require_auth
def case_analyze():
    """Run the contract-validity analysis over a whole case file (no images).

    Accepts an optional ``force_rerun: true`` field in the JSON body.
    Without it, a cached result is returned when the same canonical input was
    analyzed before (0 Groq calls, identical response).
    """
    body = request.get_json(silent=True) or {}

    # Support both {case_file: {...}} and direct case_file at root
    if "evidence" in body:
        case_file = body
        force_rerun = False
    else:
        case_file = body.get("case_file") or body
        force_rerun = bool(body.get("force_rerun", False))

    if not isinstance(case_file, dict) or not case_file.get("evidence"):
        return jsonify({"error": "case_file with an evidence[] list is required"}), 400

    case_file = _slim_case_file(case_file)

    try:
        date_checks = compute_date_checks(case_file)
        analysis_date = parse_iso(date_checks.get("analysis_date"))
        rejected = validate_user_fact_dates(case_file, analysis_date)
        if rejected:
            return jsonify({"error": "future_dated_facts", "rejected": rejected}), 422

        analysis = analyze_case(case_file, force_rerun=force_rerun, uid=g.uid)
    except AnalystError as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 502
    except GroqPayloadTooLargeError as exc:
        return jsonify({"error": str(exc)}), 413
    except GroqRateLimitError as exc:
        return jsonify({"error": str(exc), "retry_after_seconds": 30}), 429
    except HTTPError as exc:
        return jsonify({"error": f"Upstream model error: {exc}"}), 502
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Analysis failed"}), 500

    # Persist snapshot (skip if served from cache — it's already stored).
    if not analysis.get("from_cache"):
        analysis_id = save_analysis_snapshot(g.uid, analysis)
        analysis["analysis_id"] = analysis_id
        analysis.pop("analysis_input", None)  # don't send raw input to client

    return jsonify(analysis), 200


# ---------------------------------------------------------------------------
# Supplement endpoint rate limiter (in-memory, per uid, 3 req / 60 s)
# ---------------------------------------------------------------------------

_SUPPLEMENT_RATE: dict[str, deque] = defaultdict(deque)
_SUPPLEMENT_MAX = 3
_SUPPLEMENT_WINDOW = 60  # seconds


def _check_supplement_rate(uid: str) -> bool:
    """Return True if the request is within rate limit, False if exceeded."""
    now = time.monotonic()
    q: deque = _SUPPLEMENT_RATE[uid]
    while q and now - q[0] > _SUPPLEMENT_WINDOW:
        q.popleft()
    if len(q) >= _SUPPLEMENT_MAX:
        return False
    q.append(now)
    return True


@sccvision_bp.route("/sccvision/case/supplement", methods=["POST"])
@require_auth
def case_supplement():
    """Merge new evidence / statements into a case file and re-analyse.

    Request body:
    {
      "case_file": {...},            // existing case file
      "supplement": {
        "files": [<evidence item>],  // already processed by /evidence/process
        "statements": [              // typed user answers
          {"field_key": "...", "value": "...", "request_id": "...", "label": "..."}
        ]
      },
      "parent_analysis_id": "...",  // optional: analysis_id from previous round
      "force_rerun": false          // optional
    }

    Returns the new analysis response with an extra "round_diff" key.
    """
    if not _check_supplement_rate(g.uid):
        return jsonify({
            "error": "Rate limit reached for supplements. Try again in 60 seconds.",
            "retry_after_seconds": _SUPPLEMENT_WINDOW,
        }), 429

    body = request.get_json(silent=True) or {}
    case_file = body.get("case_file")
    supplement = body.get("supplement") or {}
    parent_analysis_id = body.get("parent_analysis_id")
    force_rerun = bool(body.get("force_rerun", True))  # supplements always re-run by default

    if not isinstance(case_file, dict) or not case_file.get("evidence"):
        return jsonify({"error": "case_file with an evidence[] list is required"}), 400

    new_files = supplement.get("files") or []
    statements = supplement.get("statements") or []

    if not new_files and not statements:
        return jsonify({"error": "supplement must contain at least one file or statement"}), 400

    # --- Merge new files into case file ---
    existing_ids = {ev.get("evidence_id") for ev in case_file.get("evidence") or []}
    merged_evidence = list(case_file.get("evidence") or [])
    for item in new_files:
        if isinstance(item, dict) and item.get("evidence_id") not in existing_ids:
            merged_evidence.append(item)
            existing_ids.add(item.get("evidence_id"))

    # --- Merge statements as user_provided_facts ---
    merged_facts = list(case_file.get("user_provided_facts") or [])
    answered_request_ids = set()
    for stmt in statements:
        if not isinstance(stmt, dict) or not stmt.get("field_key") or not stmt.get("value"):
            continue
        merged_facts.append({
            "id": f"stmt-{uuid.uuid4().hex[:8]}",
            "kind": "text",
            "fact": str(stmt["value"]),
            "applies_to": stmt.get("applies_to"),
            "answers_request_id": stmt.get("request_id"),
            "status": "claimant-statement",
            "label": stmt.get("label") or stmt.get("field_key"),
        })
        if stmt.get("request_id"):
            answered_request_ids.add(stmt["request_id"])

    # --- Mark answered requests as answered ---
    merged_requests = []
    for req in case_file.get("requests") or []:
        if req.get("id") in answered_request_ids and req.get("status") == "open":
            merged_requests.append({**req, "status": "answered"})
        else:
            merged_requests.append(req)

    # Build the enriched case file for this round
    round_case_file = {
        **case_file,
        "evidence": merged_evidence,
        "user_provided_facts": merged_facts,
        "requests": merged_requests,
    }
    round_case_file = _slim_case_file(round_case_file)
    next_round = len(round_case_file.get("analyses") or []) + 1

    # --- Load previous snapshot for diff computation ---
    prev_snapshot: dict = {}
    if parent_analysis_id:
        try:
            from .analysis_store import get_analysis_snapshot
            prev_snapshot = get_analysis_snapshot(g.uid, parent_analysis_id) or {}
        except Exception as exc:
            print(f"[WARN] supplement: could not load parent snapshot: {exc}")

    try:
        analysis = analyze_case(
            round_case_file,
            force_rerun=force_rerun,
            uid=g.uid,
            round_num=next_round,
            parent_analysis_id=parent_analysis_id,
        )
    except Exception as exc:
        from .analyst.client import AnalystError
        from groq_client import GroqPayloadTooLargeError, GroqRateLimitError
        if isinstance(exc, AnalystError):
            return jsonify({"error": f"Analysis failed: {exc}"}), 502
        if isinstance(exc, GroqPayloadTooLargeError):
            return jsonify({"error": str(exc)}), 413
        if isinstance(exc, GroqRateLimitError):
            return jsonify({"error": str(exc), "retry_after_seconds": 30}), 429
        if isinstance(exc, ValueError):
            return jsonify({"error": str(exc)}), 400
        return jsonify({"error": "Re-analysis failed"}), 500

    # Persist new snapshot
    if not analysis.get("from_cache"):
        from .analysis_store import save_analysis_snapshot
        analysis_id = save_analysis_snapshot(g.uid, analysis)
        analysis["analysis_id"] = analysis_id
        analysis.pop("analysis_input", None)

    # --- Compute round diff ---
    prev_follow_ups_raw = prev_snapshot.get("guided_follow_ups") or []
    new_follow_ups_raw = analysis.get("guided_follow_ups") or []
    # Reconstruct minimal FollowUpRequestV2-compatible objects for diff
    from .analyst.followups import FollowUpRequestV2
    try:
        prev_reqs = [FollowUpRequestV2(**r) for r in prev_follow_ups_raw]
        new_reqs = [FollowUpRequestV2(**r) for r in new_follow_ups_raw]
    except Exception:
        prev_reqs, new_reqs = [], []

    round_diff = compute_round_diff(
        prev_snapshot=prev_snapshot,
        new_snapshot=analysis,
        prev_requests=prev_reqs,
        new_requests=new_reqs,
        round_num=next_round,
    )
    analysis["round_diff"] = round_diff

    return jsonify(analysis), 200


@sccvision_bp.route("/sccvision/case/title", methods=["POST"])
@require_auth
def case_title():
    """Generate a short LLM case title from the user's account of the matter.

    Request body: { "user_account": "..." }
    Response:     { "title": "..." }
    """
    data = request.get_json(silent=True) or {}
    user_account = str(data.get("user_account", "")).strip()
    if not user_account:
        return jsonify({"title": "Untitled Case"}), 200

    try:
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are a legal case assistant. "
                    "Generate a short, descriptive case title (3–8 words) based on "
                    "the user's description. Output ONLY the title — no punctuation "
                    "at the end, no quotes, no explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Case description:\n{user_account[:600]}\n\nCase title:",
            },
        ]
        title = call_groq(msgs, max_tokens=40, temperature=0)
        return jsonify({"title": title.strip().strip('"').strip("'")}), 200
    except GroqRateLimitError as exc:
        return jsonify({"error": str(exc), "retry_after_seconds": 30}), 429
    except GroqPayloadTooLargeError as exc:
        return jsonify({"error": str(exc)}), 413
    except Exception:
        return jsonify({"title": "Untitled Case"}), 200


def _parse_case_context(raw) -> dict | None:
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _bool_flag(name: str, default: bool) -> bool:
    raw = request.args.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
