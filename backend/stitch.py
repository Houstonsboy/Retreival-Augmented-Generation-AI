"""
stitch.py — merge multiple WhatsApp/chat screenshots into one tall image.

Usage:
    python stitch.py <image1> <image2> [image3 ...] [-o output.jpg]
    python stitch.py SCCImages/                       # stitch all images in a folder

The images are stitched top-to-bottom in the order given (or alphabetically
when a folder is passed).  The result is saved as JPEG quality 85.

Examples:
    python stitch.py SCCImages/
    python stitch.py img1.jpg img2.jpg img3.jpg -o combined.jpg
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required:  pip install Pillow")

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(sources: list[str]) -> list[Path]:
    paths: list[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            paths.extend(
                sorted(f for f in p.iterdir()
                       if f.suffix.lower() in SUPPORTED
                       and not f.name.startswith("stitched_"))
            )
        elif p.is_file() and p.suffix.lower() in SUPPORTED:
            paths.append(p)
        else:
            print(f"[skip] {src} — not a recognised image or folder")
    return paths


def stitch(paths: list[Path], output: Path, quality: int = 85) -> None:
    if not paths:
        sys.exit("No images found.")

    print(f"Stitching {len(paths)} image(s) in order:")
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        print(f"  {p.name}  ({img.width}×{img.height})")
        images.append(img)

    width   = max(im.width  for im in images)
    total_h = sum(im.height for im in images)
    canvas  = Image.new("RGB", (width, total_h), (255, 255, 255))

    y = 0
    for im in images:
        x = (width - im.width) // 2   # centre if widths differ
        canvas.paste(im, (x, y))
        y += im.height

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "JPEG", quality=quality)
    size_kb = output.stat().st_size // 1024
    print(f"\nSaved → {output}  ({width}×{total_h}  {size_kb} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch chat screenshots vertically.")
    parser.add_argument("sources", nargs="+", help="Image files or a folder")
    parser.add_argument("-o", "--output", default="stitched_conversation.jpg",
                        help="Output file (default: stitched_conversation.jpg)")
    parser.add_argument("-q", "--quality", type=int, default=85,
                        help="JPEG quality 1-95 (default: 85)")
    args = parser.parse_args()

    paths  = collect_images(args.sources)
    output = Path(args.output)
    if output.suffix.lower() not in {".jpg", ".jpeg"}:
        output = output.with_suffix(".jpg")

    stitch(paths, output, quality=args.quality)


if __name__ == "__main__":
    main()
