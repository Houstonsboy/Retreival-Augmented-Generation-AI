"""High-level text extraction for SCCVision."""

from __future__ import annotations

import time
from importlib import metadata
from typing import Any

import cv2

from .analysis import analyze
from .engine import get_engine
from .preprocess import PreprocessOptions, decode, preprocess_image


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def extract_text(
    image_bytes: bytes,
    options: PreprocessOptions | None = None,
    file_name: str | None = None,
) -> dict:
    """Extract and interpret text from a WhatsApp screenshot.

    Runs fully in memory; nothing is written to disk and nothing is logged.
    """
    options = options or PreprocessOptions()

    t_start = time.perf_counter()

    t0 = time.perf_counter()
    orig_rgb = decode(image_bytes)
    img, preprocess_info = preprocess_image(orig_rgb, options)
    preprocess_ms = (time.perf_counter() - t0) * 1000.0

    original_size = preprocess_info["original_size"]
    scale = float(preprocess_info.get("scale", 1.0))

    # RapidOCR treats a plain ndarray as BGR, so convert from RGB.
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()
    result = get_engine()(bgr)
    ocr_ms = (time.perf_counter() - t0) * 1000.0

    lines = _parse_lines(result, original_size, scale)

    t0 = time.perf_counter()
    analysis = analyze(lines, orig_rgb, original_size)
    analysis_ms = (time.perf_counter() - t0) * 1000.0

    confidences = [line["confidence"] for line in lines]
    avg_confidence = (
        round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    )

    return {
        "meta": {
            "engine": "rapidocr",
            "engine_version": _package_version("rapidocr"),
            "onnxruntime_version": _package_version("onnxruntime"),
            "file_name": file_name or "",
            "options_used": {
                "invert": bool(options.invert),
                "upscale": bool(options.upscale),
            },
        },
        "image_size": {
            "width": int(original_size["width"]),
            "height": int(original_size["height"]),
        },
        "timings_ms": {
            "preprocess": round(preprocess_ms, 2),
            "ocr": round(ocr_ms, 2),
            "analysis": round(analysis_ms, 2),
            "total": round((time.perf_counter() - t_start) * 1000.0, 2),
        },
        "full_text": "\n".join(line["text"] for line in lines),
        "line_count": len(lines),
        "avg_confidence": avg_confidence,
        "contact": analysis["contact"],
        "messages": analysis["messages"],
        "system_events": analysis["system_events"],
        "stats": analysis["stats"],
        "warnings": analysis["warnings"],
        "lines": lines,
    }


def _parse_lines(result: Any, original_size: dict, scale: float) -> list[dict]:
    txts = getattr(result, "txts", None)
    if not txts:
        return []

    boxes = getattr(result, "boxes", None)
    scores = getattr(result, "scores", None)

    orig_w = int(original_size["width"])
    orig_h = int(original_size["height"])

    raw: list[dict] = []
    for i, txt in enumerate(txts):
        if not txt:
            continue

        score = float(scores[i]) if scores is not None else 0.0
        pts = boxes[i] if boxes is not None else None

        box: list[list[int]] = []
        x_center_norm = 0.0
        y_center_norm = 0.0
        sort_key: tuple[float, float] = (0.0, 0.0)

        if pts is not None:
            xs: list[int] = []
            ys: list[int] = []
            for p in pts:
                x = int(round(float(p[0]) / scale))
                y = int(round(float(p[1]) / scale))
                box.append([x, y])
                xs.append(x)
                ys.append(y)

            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            x_center_norm = round(cx / orig_w, 3) if orig_w else 0.0
            y_center_norm = round(cy / orig_h, 3) if orig_h else 0.0
            sort_key = (float(min(ys)), float(min(xs)))

        raw.append(
            {
                "text": str(txt),
                "confidence": round(score, 3),
                "box": box,
                "x_center_norm": x_center_norm,
                "y_center_norm": y_center_norm,
                "_sort": sort_key,
            }
        )

    # Reading order: top-to-bottom, then left-to-right.
    raw.sort(key=lambda line: line["_sort"])

    for idx, line in enumerate(raw):
        line["index"] = idx
        del line["_sort"]

    return raw
