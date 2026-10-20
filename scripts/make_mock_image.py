"""Render a synthetic shelf image from a detection-output JSON, so the mock
image always matches the mock detections.

    python scripts/make_mock_image.py [contracts/detection_output.example.json]
"""

import colorsys
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from shelfsense.contracts import load_detection_output

ROOT = Path(__file__).resolve().parents[1]


def sku_colour(sku: str | None) -> tuple[int, int, int]:
    if sku is None:
        return (150, 150, 150)
    hue = int(hashlib.md5(sku.encode()).hexdigest(), 16) % 360 / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.85)
    return int(r * 255), int(g * 255), int(b * 255)


def main(path: str) -> None:
    det = load_detection_output(path)
    img = Image.new("RGB", (det.image_width, det.image_height), (235, 230, 220))
    draw = ImageDraw.Draw(img)

    for shelf in det.shelves:
        x1, y1, x2, y2 = shelf.bbox
        draw.rectangle([x1, y1, x2, y2], fill=(205, 195, 180))
        draw.rectangle([x1, y2 - 36, x2, y2], fill=(120, 95, 70))  # shelf edge

    for gap in det.gaps:
        draw.rectangle(gap.bbox, fill=(60, 55, 50))

    for item in det.items:
        draw.rectangle(item.bbox, fill=sku_colour(item.sku), outline=(40, 40, 40), width=2)
        x1, y1, _, _ = item.bbox
        draw.text((x1 + 6, y1 + 8), item.sku or "????", fill=(20, 20, 20))

    for label in det.labels:
        draw.rectangle(label.bbox, fill=(250, 250, 245), outline=(30, 30, 30))
        x1, y1, _, _ = label.bbox
        draw.text((x1 + 6, y1 + 6), label.ocr_text, fill=(0, 0, 0))

    out = ROOT / det.image_path
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "contracts/detection_output.example.json"))
