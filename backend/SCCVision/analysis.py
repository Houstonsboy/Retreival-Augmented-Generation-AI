"""WhatsApp-aware interpretation of raw OCR lines.

Turns a flat list of detected text lines into a structured conversation:
contact header, grouped messages (with sender), and system events. The
``analyze`` entry point mutates each line in place by adding a ``role`` and
returns the structured result.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

STATUS_BAR_RATIO = 0.045
HEADER_BOTTOM = 0.10
AVATAR_LEFT_MAX = 0.13
SYSTEM_CENTER_TOL = 0.07
SYSTEM_MAX_WIDTH = 0.5

_TIMESTAMP_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(\s?[AaPp][Mm])?\s*$")
_EDITED_RE = re.compile(
    r"^\s*Edited\s+(\d{1,2}):(\d{2})(\s?[AaPp][Mm])?\s*$"
)


def parse_timestamp(text: str) -> dict | None:
    """Parse a timestamp OCR line into ``{time, edited, ticks}`` or ``None``.

    ``ticks`` is True when the raw text carries a trailing space (WhatsApp
    read receipts, which only appear on outgoing messages).
    """
    raw = text or ""
    ticks = bool(raw) and raw != raw.rstrip(" ")
    m = _EDITED_RE.match(raw) or _TIMESTAMP_RE.match(raw)
    if not m:
        return None
    edited = bool(_EDITED_RE.match(raw))
    hh = int(m.group(1))
    mm = int(m.group(2))
    ampm = (m.group(3) or "").strip().upper()
    hour = hh
    if ampm:
        if ampm.startswith("P") and hh != 12:
            hour = hh + 12
        elif ampm.startswith("A") and hh == 12:
            hour = 0
    return {"time": f"{hour:02d}:{mm:02d}", "edited": edited, "ticks": ticks}


def _box_metrics(box: list, width: int, height: int) -> dict | None:
    if not box or len(box) < 3:
        return None
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
        "x_center": (left + right) / 2.0,
        "y_center": (top + bottom) / 2.0,
        "x_center_norm": ((left + right) / 2.0) / width if width else 0.0,
        "y_center_norm": ((top + bottom) / 2.0) / height if height else 0.0,
        "width_norm": (right - left) / width if width else 0.0,
        "height_norm": (bottom - top) / height if height else 0.0,
    }


def _is_system(m: dict) -> bool:
    return (
        abs(m["x_center_norm"] - 0.5) <= SYSTEM_CENTER_TOL
        and m["width_norm"] < SYSTEM_MAX_WIDTH
    )


def _classify(line: dict, chat_bottom: float | None) -> str:
    m = line["_m"]
    if not m:
        return "ui_noise"
    y = m["y_center_norm"]
    x = m["x_center_norm"]
    text = (line.get("text") or "").strip()

    if y < STATUS_BAR_RATIO:
        return "status_bar"
    if y < HEADER_BOTTOM:
        if x < AVATAR_LEFT_MAX and len(text) == 1:
            return "avatar"
        return "header"

    if chat_bottom is not None and y > chat_bottom:
        return "ui_noise"
    if line.get("_ts"):
        return "timestamp"
    if _is_system(m):
        return "system"
    if len(text) <= 1:
        return "ui_noise"
    return "message_text"


def _detect_contact(lines: list[dict]) -> dict | None:
    header = [l for l in lines if l["role"] == "header"]
    if not header:
        return None
    name = max(header, key=lambda l: l["_m"]["height_norm"])
    rest = [l for l in header if l is not name]
    presence = max(rest, key=lambda l: l["_m"]["y_center_norm"]) if rest else None
    return {
        "name": name["text"].strip(),
        "presence": presence["text"].strip() if presence else None,
    }


def _union_box(buffer: list[dict]) -> list:
    xs = [float(p[0]) for l in buffer for p in l["box"]]
    ys = [float(p[1]) for l in buffer for p in l["box"]]
    if not xs:
        return []
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return [
        [int(left), int(top)],
        [int(right), int(top)],
        [int(right), int(bottom)],
        [int(left), int(bottom)],
    ]


def _is_green(r: float, g: float, b: float) -> bool:
    return g > r and g > b and (g - r) >= 8.0 and (g - b) >= 8.0


def _is_neutral(r: float, g: float, b: float) -> bool:
    return (
        abs(int(r) - int(g)) <= 12
        and abs(int(g) - int(b)) <= 12
        and abs(int(r) - int(b)) <= 12
    )


def _sample_bubble_color(rgb: np.ndarray | None, box: list) -> tuple | None:
    """Sample the bubble color from a band just outside the text block."""
    if rgb is None or not box or len(box) < 3:
        return None
    h, w = rgb.shape[:2]
    xs = [int(p[0]) for p in box]
    ys = [int(p[1]) for p in box]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)

    bands: list[np.ndarray] = []

    x0, x1 = max(0, left - 8), max(0, left - 3)
    if x1 > x0:
        y0, y1 = max(0, top + 2), min(h, bottom - 2)
        if y1 > y0:
            bands.append(rgb[y0:y1, x0:x1].reshape(-1, 3))

    y0, y1 = max(0, top - 8), max(0, top - 3)
    if y1 > y0:
        x0, x1 = max(0, left + 2), min(w, right - 2)
        if x1 > x0:
            bands.append(rgb[y0:y1, x0:x1].reshape(-1, 3))

    if not bands:
        return None
    px = np.concatenate(bands, axis=0)
    if px.size == 0:
        return None
    median = np.median(px, axis=0)
    return (float(median[0]), float(median[1]), float(median[2]))


def _left_edge_norm(buffer: list[dict], width: int) -> float | None:
    if not width:
        return None
    lefts = [l["_m"]["left"] for l in buffer if l.get("_m")]
    return min(lefts) / width if lefts else None


def _sender_label(decision: str | None, contact: dict | None) -> str:
    if decision == "outgoing":
        return "me"
    if contact and contact.get("name"):
        return contact["name"]
    return "them"


def _combine(
    color_decision: str | None,
    edge_decision: str | None,
    tick_decision: str | None,
    contact: dict | None,
) -> tuple[str, str]:
    if color_decision is not None:
        confidence = "high"
        if edge_decision is not None and edge_decision != color_decision:
            confidence = "low"
        return _sender_label(color_decision, contact), confidence
    if edge_decision is not None:
        return _sender_label(edge_decision, contact), "low"
    if tick_decision is not None:
        return _sender_label(tick_decision, contact), "low"
    return "unknown", "low"


def _detect_sender(
    buffer: list[dict],
    box: list,
    ts: dict | None,
    orig_rgb: np.ndarray | None,
    image_size: dict,
    contact: dict | None,
) -> tuple[str, str, dict]:
    signals: dict = {}

    color = _sample_bubble_color(orig_rgb, box)
    color_decision: str | None = None
    if color is not None:
        r, g, b = color
        signals["color"] = [int(r), int(g), int(b)]
        if _is_green(r, g, b):
            color_decision = "outgoing"
        elif _is_neutral(r, g, b):
            color_decision = "incoming"
    else:
        signals["color"] = None

    left_norm = _left_edge_norm(buffer, int(image_size.get("width", 0)))
    signals["left_edge_norm"] = round(left_norm, 3) if left_norm is not None else None
    edge_decision: str | None = None
    if left_norm is not None:
        if left_norm < 0.12:
            edge_decision = "incoming"
        else:
            edge_decision = "outgoing"

    ticks = bool(ts and ts.get("ticks"))
    signals["ticks"] = ticks
    tick_decision: str | None = "outgoing" if ticks else None

    sender, confidence = _combine(
        color_decision, edge_decision, tick_decision, contact
    )
    return sender, confidence, signals


def _flush(
    buffer: list[dict],
    messages: list[dict],
    orig_rgb: np.ndarray | None,
    image_size: dict,
    contact: dict | None,
    ts: dict | None,
) -> None:
    if not buffer:
        return
    text = " ".join((l["text"] or "").strip() for l in buffer).strip()
    box = _union_box(buffer)
    sender, confidence, signals = _detect_sender(
        buffer, box, ts, orig_rgb, image_size, contact
    )
    avg_conf = round(
        sum(l["confidence"] for l in buffer) / len(buffer), 3
    )
    messages.append(
        {
            "index": len(messages) + 1,
            "sender": sender,
            "sender_confidence": confidence,
            "sender_signals": signals,
            "text": text,
            "lines": [l["text"] for l in buffer],
            "timestamp": ts["time"] if ts else None,
            "edited": ts["edited"] if ts else False,
            "incomplete": ts is None,
            "avg_confidence": avg_conf,
            "box": box,
        }
    )


def _group(
    lines: list[dict],
    orig_rgb: np.ndarray | None,
    image_size: dict,
    contact: dict | None,
) -> tuple[list[dict], list[dict]]:
    messages: list[dict] = []
    system_events: list[dict] = []
    buffer: list[dict] = []

    relevant = [
        l for l in lines if l["role"] in ("message_text", "timestamp", "system")
    ]
    relevant.sort(key=lambda l: (l["_m"]["top"], l["_m"]["left"]) if l["_m"] else (0.0, 0.0))

    for line in relevant:
        role = line["role"]
        if role == "message_text":
            buffer.append(line)
        elif role == "timestamp":
            _flush(buffer, messages, orig_rgb, image_size, contact, line.get("_ts"))
            buffer = []
        elif role == "system":
            if buffer:
                _flush(buffer, messages, orig_rgb, image_size, contact, None)
                buffer = []
            system_events.append(
                {
                    "after_message_index": len(messages),
                    "text": (line["text"] or "").strip(),
                }
            )
    if buffer:
        _flush(buffer, messages, orig_rgb, image_size, contact, None)
    return messages, system_events


def _build_stats(lines: list[dict], messages: list[dict]) -> dict:
    removed: dict[str, int] = {}
    for l in lines:
        if l["role"] in ("status_bar", "ui_noise", "avatar"):
            removed[l["role"]] = removed.get(l["role"], 0) + 1
    kept = sum(
        1 for l in lines if l["role"] in ("message_text", "timestamp", "system", "header")
    )
    return {
        "raw_line_count": len(lines),
        "kept_line_count": kept,
        "removed_by_role": removed,
        "message_count": len(messages),
    }


def _build_warnings(
    messages: list[dict], contact: dict | None
) -> list[str]:
    warnings: list[str] = []
    if contact is None or not contact.get("name"):
        warnings.append("contact_name_not_detected")
    incomplete = [m for m in messages if m["incomplete"]]
    if incomplete:
        warnings.append(f"{len(incomplete)} message(s) missing a timestamp")
    low = [m for m in messages if m["sender_confidence"] != "high"]
    if low:
        warnings.append(f"{len(low)} message(s) with low sender confidence")
    return warnings


def analyze(
    lines: list[dict],
    orig_rgb: np.ndarray | None,
    image_size: dict,
) -> dict:
    """Annotate lines with roles and return the structured conversation."""
    width = int(image_size.get("width", 0))
    height = int(image_size.get("height", 0))

    for line in lines:
        line["_m"] = _box_metrics(line.get("box"), width, height)
        line["_ts"] = parse_timestamp(line.get("text", ""))

    chat_bottom: float | None = None
    for line in lines:
        if (
            line["_m"]
            and line["_m"]["y_center_norm"] >= STATUS_BAR_RATIO
            and line.get("_ts")
        ):
            if chat_bottom is None or line["_m"]["y_center_norm"] > chat_bottom:
                chat_bottom = line["_m"]["y_center_norm"]

    for line in lines:
        line["role"] = _classify(line, chat_bottom)

    contact = _detect_contact(lines)
    messages, system_events = _group(lines, orig_rgb, image_size, contact)
    stats = _build_stats(lines, messages)
    warnings = _build_warnings(messages, contact)

    for line in lines:
        line.pop("_m", None)
        line.pop("_ts", None)

    return {
        "contact": contact,
        "messages": messages,
        "system_events": system_events,
        "stats": stats,
        "warnings": warnings,
    }
