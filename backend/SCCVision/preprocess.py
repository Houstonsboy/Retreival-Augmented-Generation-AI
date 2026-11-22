"""Image decoding and minimal, configurable preprocessing for SCCVision OCR."""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when raw bytes do not decode to a usable image."""


@dataclass
class PreprocessOptions:
    """Toggleable preprocessing switches so results can be compared.

    ``invert`` enables dark-mode detection/inversion; ``upscale`` enlarges
    small images so the text detector has enough pixels to work with.
    """

    upscale: bool = True
    invert: bool = True
    min_side: int = 800
    dark_mode_threshold: float = 128.0

    def to_dict(self) -> dict:
        return asdict(self)


def decode(data: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB ndarray, applying EXIF orientation.

    Raises :class:`InvalidImageError` for anything that is not a decodable
    image. Works entirely in memory, never touching disk.
    """
    if not data:
        raise InvalidImageError("empty image bytes")

    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            if im is None:
                raise InvalidImageError("could not decode image")
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("not a valid image") from exc


def preprocess_image(
    img: np.ndarray, options: PreprocessOptions | None = None
) -> tuple[np.ndarray, dict]:
    """Preprocess an already-decoded RGB image, returning image plus metadata.

    Kept separate from :func:`decode` so callers can retain the original
    (non-inverted, non-upscaled) RGB array for pixel sampling.
    """
    options = options or PreprocessOptions()
    info: dict = {}

    height, width = img.shape[:2]
    info["original_size"] = {"width": int(width), "height": int(height)}

    short_side = min(height, width)
    if options.upscale and 0 < short_side < options.min_side:
        scale = options.min_side / float(short_side)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        info["upscaled"] = True
        info["scale"] = scale
    else:
        info["upscaled"] = False
        info["scale"] = 1.0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(gray.mean())
    dark_mode = mean_brightness < options.dark_mode_threshold
    info["mean_brightness"] = round(mean_brightness, 2)
    info["dark_mode"] = dark_mode

    if options.invert and dark_mode:
        img = cv2.bitwise_not(img)
        info["inverted"] = True
    else:
        info["inverted"] = False

    info["processed_size"] = {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
    }
    return img, info


def preprocess(
    data: bytes, options: PreprocessOptions | None = None
) -> tuple[np.ndarray, dict]:
    """Decode and preprocess raw bytes, returning an RGB image plus metadata."""
    img = decode(data)
    return preprocess_image(img, options)
