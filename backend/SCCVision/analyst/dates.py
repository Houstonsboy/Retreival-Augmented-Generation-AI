"""Deterministic date handling for the Contract Evidence Analyst.

Collects dates from the whole case file (chat date chips, M-Pesa
confirmations, user-provided facts, message text, file names), resolves years
conservatively, compares everything against ``analysis_date`` ("today" in the
configured timezone) and produces ``date_checks`` plus enforcement helpers.

Nothing here calls the LLM and nothing is persisted or logged.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time as dtime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from config import SCC_TIMEZONE

_MONTHS: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_ALT = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"

# Day-first month-name date: "8 November 2026", "2 Oct", "16 Nov".
_DAY_FIRST_RE = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<mon>{_MONTH_ALT})[a-z]*"
    rf"(?:[,\s]+(?P<year>\d{{2,4}}))?\b",
    re.IGNORECASE,
)
# Month-first ordinal date: "November 16th", "November 16".
_MONTH_FIRST_RE = re.compile(
    rf"\b(?P<mon>{_MONTH_ALT})[a-z]*\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:[,\s]+(?P<year>\d{{2,4}}))?\b",
    re.IGNORECASE,
)
# Numeric day-first: "16/11", "16/11/2026", "16-11-26".
_NUM_DATE_RE = re.compile(
    r"\b(?P<day>\d{1,2})[/\-](?P<mon>\d{1,2})(?:[/\-](?P<year>\d{2,4}))?\b"
)
_RELATIVE_RE = re.compile(r"\b(today|yesterday|tomorrow)\b", re.IGNORECASE)
_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
)

# occurrence date roles: things that have happened (cannot be in the future)
OCCURRENCE_ROLES = {"conversation", "payment", "delivery"}
DUE_ROLES = {"delivery", "service", "other"}

_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def get_timezone() -> ZoneInfo:
    try:
        return ZoneInfo(SCC_TIMEZONE or "Africa/Nairobi")
    except ZoneInfoNotFoundError:
        return ZoneInfo("Africa/Nairobi")


def parse_iso(value: str | None) -> date | None:
    if not value:
        return None
    m = _ISO_RE.match(str(value).strip())
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _normalize_year(year: int) -> int:
    return 2000 + year if year < 100 else year


def resolve_analysis_date(case_file: dict[str, Any], now: datetime | None = None) -> tuple[date, bool]:
    """Return ``(analysis_date, overridden)`` for a case file."""
    override = ((case_file.get("case_context") or {}).get("analysis_date_override") or "").strip()
    if override:
        d = parse_iso(override)
        if d is None:
            raise ValueError("analysis_date_override must be YYYY-MM-DD")
        return d, True
    now_dt = now or datetime.now(get_timezone())
    return now_dt.date(), False


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _month_from(name: str | None) -> int | None:
    if not name:
        return None
    key = str(name).lower()[:3]
    return _MONTHS.get(key)


def _build_date(day: int | None, month: int | None, year: int | None) -> date | None:
    if day is None or month is None:
        return None
    y = _normalize_year(year) if year is not None else None
    try:
        return date(y, month, day) if y else date(2000, month, day)
    except ValueError:
        return None


def _resolve_year(day: int, month: int, year: int | None, base: date | None) -> tuple[date | None, bool]:
    """Resolve a day/month (and optional year) into a concrete date.

    Returns ``(date, year_assumed)``; ``date`` is ``None`` when the year cannot
    be resolved. A two-digit year is expanded to 20xx.
    """
    if year is not None:
        y = _normalize_year(year)
        try:
            return date(y, month, day), False
        except ValueError:
            return None, False
    if base is None:
        return None, False
    candidate = date(base.year, month, day)
    if candidate < base:
        candidate = date(base.year + 1, month, day)
    return candidate, True


def extract_date_candidates(text: str | None) -> list[dict[str, Any]]:
    """Return structured date candidates found in free text (chips or messages)."""
    out: list[dict[str, Any]] = []
    t = text or ""

    for m in _DAY_FIRST_RE.finditer(t):
        out.append({
            "raw": m.group(0).strip(),
            "kind": "absolute",
            "day": int(m.group("day")),
            "month": _month_from(m.group("mon")),
            "year": int(m.group("year")) if m.group("year") else None,
        })
    for m in _MONTH_FIRST_RE.finditer(t):
        raw = m.group(0).strip()
        if any(c["raw"] == raw for c in out):
            continue
        out.append({
            "raw": raw,
            "kind": "absolute",
            "day": int(m.group("day")),
            "month": _month_from(m.group("mon")),
            "year": int(m.group("year")) if m.group("year") else None,
        })
    for m in _NUM_DATE_RE.finditer(t):
        raw = m.group(0).strip()
        if any(c["raw"] == raw for c in out):
            continue
        day = int(m.group("day"))
        month = int(m.group("mon"))
        if month < 1 or month > 12:
            continue
        out.append({
            "raw": raw,
            "kind": "numeric",
            "day": day,
            "month": month,
            "year": int(m.group("year")) if m.group("year") else None,
        })

    for m in _RELATIVE_RE.finditer(t):
        out.append({"raw": m.group(0), "kind": "relative", "day": None, "month": None, "year": None})
    for m in _WEEKDAY_RE.finditer(t):
        raw = m.group(0)
        if any(c["raw"].lower() == raw.lower() for c in out):
            continue
        out.append({"raw": raw, "kind": "weekday", "day": None, "month": None, "year": None})

    return out


def _filename_hint(file_name: str | None) -> date | None:
    if not file_name:
        return None
    # A WhatsApp export often names files like "WhatsApp Image 2026-11-08 at ...".
    m = re.search(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})", file_name or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_date_checks(case_file: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Compute ``date_checks`` for a case file. Deterministic and offline."""
    tz = get_timezone()
    now_dt = now or datetime.now(tz)
    analysis_date, overridden = resolve_analysis_date(case_file, now_dt)

    evidence = case_file.get("evidence") or []
    user_facts = case_file.get("user_provided_facts") or []

    conversation_dates: list[dict[str, Any]] = []
    undated_evidence_ids: list[str] = []
    mentioned_dates: list[dict[str, Any]] = []
    future_dated_events: list[dict[str, Any]] = []
    order_violations: list[dict[str, Any]] = []

    # Known conversation date (used for year resolution): earliest confirmed
    # date from a chat chip or a user-stated "conversation" date.
    known_conv: date | None = None

    # 1. User-provided conversation/payment/delivery dates.
    for f in user_facts:
        if f.get("kind") != "date":
            continue
        role = f.get("date_role") or "other"
        d = parse_iso(f.get("fact")) if isinstance(f.get("fact"), str) else None
        if d is None:
            continue
        if role == "conversation":
            status = "confirmed" if f.get("confidence") == "sure" else "proposed"
            conversation_dates.append({
                "evidence_id": f.get("applies_to"),
                "date": d.isoformat(),
                "source": "user_stated",
                "status": status,
            })
            if known_conv is None and status == "confirmed":
                known_conv = d
        elif role in ("payment", "delivery"):
            # A user-stated payment/delivery date is an occurrence date.
            future = _is_future(d, None, analysis_date, now_dt)
            if future:
                future_dated_events.append({"what": f"user-stated {role} date", "date": d.isoformat(), "source": "user_typed"})

    # 2. Evidence: chat chips, message text, M-Pesa confirmations, file names.
    for ev in evidence:
        ev_id = ev.get("evidence_id")
        etype = ev.get("type")
        ocr = ev.get("ocr") or {}
        parsed = ev.get("parsed") or []

        if etype == "chat_screenshot":
            has_resolved_chip = False
            for se in ocr.get("system_events") or []:
                candidates = extract_date_candidates(se.get("text"))
                for c in candidates:
                    if c["kind"] in ("relative", "weekday"):
                        continue
                    resolved, year_assumed = _resolve_year(c["day"], c["month"], c["year"], known_conv)
                    if resolved is not None:
                        has_resolved_chip = True
                        entry = {
                            "evidence_id": ev_id,
                            "date": resolved.isoformat(),
                            "source": "chat_chip",
                            "status": "proposed" if year_assumed else "confirmed",
                        }
                        if entry not in conversation_dates:
                            conversation_dates.append(entry)
                        if known_conv is None:
                            known_conv = resolved
                        if _is_future(resolved, None, analysis_date, now_dt):
                            future_dated_events.append({
                                "what": f"conversation date chip {c['raw']}",
                                "date": resolved.isoformat(),
                                "source": "chat_chip",
                            })
            # Mentioned dates in message text.
            for m in ocr.get("messages") or []:
                for c in extract_date_candidates(m.get("text")):
                    if c["kind"] in ("relative", "weekday"):
                        mentioned_dates.append({
                            "msg_index": m.get("index"),
                            "raw": c["raw"],
                            "resolved": None,
                            "year_assumed": False,
                            "relation": "unresolved",
                            "candidate_due_date": False,
                        })
                        continue
                    resolved, year_assumed = _resolve_year(c["day"], c["month"], c["year"], known_conv)
                    relation, candidate_due, future = _relation(resolved, analysis_date, known_conv, now_dt)
                    mentioned_dates.append({
                        "msg_index": m.get("index"),
                        "raw": c["raw"],
                        "resolved": resolved.isoformat() if resolved else None,
                        "year_assumed": year_assumed,
                        "relation": relation,
                        "candidate_due_date": candidate_due,
                    })
                    if future and not candidate_due:
                        future_dated_events.append({
                            "what": f"mentioned date {c['raw']}",
                            "date": resolved.isoformat(),
                            "source": "chat_message",
                        })
            if not has_resolved_chip:
                undated_evidence_ids.append(ev_id)

        elif etype == "mpesa_screenshot":
            for p in parsed:
                d = parse_iso(p.get("date")) if p.get("date") else None
                t = p.get("time")
                if d is None:
                    continue
                if _is_future(d, _parse_time(t), analysis_date, now_dt):
                    future_dated_events.append({
                        "what": f"M-Pesa confirmation {p.get('code') or ''}".strip(),
                        "date": d.isoformat(),
                        "source": "mpesa_confirmation",
                        "evidence_id": ev_id,
                    })

        hint = _filename_hint(ev.get("file_name"))
        if hint is not None:
            conversation_dates.append({
                "evidence_id": ev_id,
                "date": hint.isoformat(),
                "source": "filename_hint",
                "status": "proposed",
            })

    # De-duplicate conversation dates.
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for cd in conversation_dates:
        key = (cd.get("date"), cd.get("evidence_id") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cd)
    conversation_dates = deduped

    # 3. Order violations across resolved mentioned dates (in message order).
    ordered = sorted(
        [m for m in mentioned_dates if m.get("resolved") and m.get("msg_index") is not None],
        key=lambda m: m["msg_index"],
    )
    prev_date: date | None = None
    for m in ordered:
        d = parse_iso(m["resolved"])
        if d and prev_date and d < prev_date:
            order_violations.append({
                "type": "date_order_violation",
                "msg_index": m["msg_index"],
                "date": d.isoformat(),
            })
        if d:
            prev_date = d

    # 4. Performance due date.
    due_candidates = sorted(
        [m for m in mentioned_dates if m.get("candidate_due_date") and m.get("resolved")],
        key=lambda m: m["resolved"],
    )
    performance_due: dict[str, Any] = {"date": None, "status": "unknown"}
    if due_candidates:
        due_date = parse_iso(due_candidates[0]["resolved"])
        performance_due["date"] = due_date.isoformat() if due_date else None
        if due_date:
            if due_date > analysis_date:
                performance_due["status"] = "not_yet_due"
            elif due_date == analysis_date:
                performance_due["status"] = "due_today"
            else:
                performance_due["status"] = "overdue"

    no_dates = not conversation_dates

    return {
        "analysis_date": analysis_date.isoformat(),
        "overridden": overridden,
        "timezone": str(tz),
        "conversation_dates": conversation_dates,
        "undated_evidence_ids": undated_evidence_ids,
        "mentioned_dates": mentioned_dates,
        "future_dated_events": future_dated_events,
        "order_violations": order_violations,
        "performance_due": performance_due,
        "no_dates": no_dates,
    }


def _relation(
    resolved: date | None,
    analysis_date: date,
    known_conv: date | None,
    now_dt: datetime | None,
) -> tuple[str, bool, bool]:
    """Return (relation, candidate_due_date, is_future_occurrence)."""
    if resolved is None:
        return "unresolved", False, False
    if resolved > analysis_date:
        relation = "future"
    elif resolved == analysis_date:
        relation = "today"
    else:
        relation = "past"
    candidate_due = False
    if known_conv is not None and resolved > known_conv:
        candidate_due = True
    elif resolved >= analysis_date:
        candidate_due = True
    future = _is_future(resolved, None, analysis_date, now_dt)
    return relation, candidate_due, future


def _is_future(
    d: date,
    t: dtime | None,
    analysis_date: date,
    now_dt: datetime | None,
) -> bool:
    if d > analysis_date:
        return True
    if d == analysis_date and t is not None and now_dt is not None and now_dt.date() == analysis_date:
        return t > now_dt.time()
    return False


def _parse_time(value: str | None) -> dtime | None:
    if not value:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})\s?(AM|PM)?$", str(value).strip(), re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1)) % 12
    if m.group(3) and m.group(3).upper() == "PM":
        hour += 12
    return dtime(hour, int(m.group(2)))


def validate_user_fact_dates(case_file: dict[str, Any], analysis_date: date) -> list[dict[str, Any]]:
    """Return user-typed occurrence dates that are in the future (must be rejected)."""
    rejected: list[dict[str, Any]] = []
    for f in case_file.get("user_provided_facts") or []:
        if f.get("kind") != "date":
            continue
        role = f.get("date_role") or "other"
        if role not in OCCURRENCE_ROLES:
            continue
        d = parse_iso(f.get("fact")) if isinstance(f.get("fact"), str) else None
        if d and d > analysis_date:
            rejected.append({
                "id": f.get("id"),
                "fact": f.get("fact"),
                "reason": f"future {role} date: {d.isoformat()} is after analysis_date {analysis_date.isoformat()}",
            })
    return rejected
