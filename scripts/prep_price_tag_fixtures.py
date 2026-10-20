"""Regenerate tests/fixtures/price_tags/ from data/supermarket_shelves/
(the Humans in the Loop "Supermarket shelves" dataset, CC0 -- see
data/README.md). Crops a hand-picked, hand-transcribed set of `Price`
boxes so label_ocr.py has a small, committed, ground-truth fixture to
test against, without checking the ~140MB raw dataset into git.

    python scripts/prep_price_tag_fixtures.py

Re-run this if you want to change the selection; it's a thin, reviewable
list of (image, Price-object index, ground truth text) below -- ground
truth was transcribed by eye from each crop. `None` marks a tag left in
on purpose because it's illegible even to a human (digital shelf-edge
displays caught at a bad angle), to exercise the unreadable-price path.
"""

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "supermarket_shelves"
OUT = ROOT / "tests" / "fixtures" / "price_tags"

# (source image, index into that image's Price objects in object order,
# hand-transcribed ground truth text, or None if illegible)
SELECTION = [
    ("002.jpg", 0, "965"),
    ("002.jpg", 1, "2.140"),
    ("002.jpg", 2, "1.505"),
    ("002.jpg", 3, "2.235"),
    ("003.jpg", 0, "118円"),
    ("003.jpg", 1, "94円"),
    ("003.jpg", 2, "258円"),
    ("003.jpg", 3, "218円"),
    ("015.jpg", 1, "8.95"),
    ("015.jpg", 3, "8.65"),
    ("015.jpg", 0, None),  # digital display, unreadable angle
    ("015.jpg", 2, None),  # digital display, unreadable angle
    ("017.jpg", 0, "3/$10.99"),
    ("017.jpg", 1, None),  # tiny/blurry
    ("029.jpg", 0, "0.70"),
    ("029.jpg", 1, "RM 15.90"),
]


def price_boxes(image_name: str) -> list[dict]:
    ann = json.loads((RAW / "annotations" / f"{image_name}.jpg.json").read_text())
    return [o for o in ann["objects"] if o["classTitle"] == "Price"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str | None] = {}

    # Pick the same candidates prep used when transcribing (>150px wide,
    # same shuffle), so indices line up with SELECTION above. One global
    # seed, images visited in this fixed order -- matches how they were
    # originally sampled and transcribed by eye.
    import random

    random.seed(7)
    picks_by_image: dict[str, list[dict]] = {}
    for img_name in ("002.jpg", "003.jpg", "015.jpg", "017.jpg", "029.jpg", "038.jpg"):
        stem = img_name.removesuffix(".jpg")
        boxes = price_boxes(stem)
        boxes = [b for b in boxes if (b["points"]["exterior"][1][0] - b["points"]["exterior"][0][0]) > 150]
        random.shuffle(boxes)
        picks_by_image[img_name] = boxes

    for img_name, idx, truth in SELECTION:
        stem = img_name.removesuffix(".jpg")
        box = picks_by_image[img_name][idx]
        (x1, y1), (x2, y2) = box["points"]["exterior"]
        im = Image.open(RAW / "images" / img_name)
        pad = 5
        crop = im.crop((max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad))
        out_name = f"{stem}_{idx}.png"
        crop.save(OUT / out_name)
        manifest[out_name] = truth

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(manifest)} crops + manifest.json to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
