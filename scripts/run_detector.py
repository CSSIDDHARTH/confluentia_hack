"""Run the product-block detector on images and save detection_output JSON + an overlay.

    python scripts/run_detector.py --weights outputs/runs/grozi_v1/weights/best.pt \
        --images data/grozi/Testing/store1/images/15.jpg data/test_images [--out outputs/detections]

For every image: <out>/<id>.json (the DetectionOutput contract) and <out>/<id>.png with
predictions in green and, for Grozi images that have annotations, ground truth in blue.

With --identify each box is also matched to a product (name shown on the box; orange =
confidence too low, left for review). --bay restricts matching to the image's own
products plus 20 random distractors, which stands in for a known planogram (demo mode).
"""

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shelfsense.contracts import save_detection_output
from shelfsense.vision.detect import ProductDetector
from shelfsense.vision.pipeline import identify_items
from shelfsense.vision.sku_match import SkuMatcher, class_index, sku_id

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    return out


def grozi_ground_truth(img: Path, size: tuple[int, int], with_skus: bool = False):
    """Pixel boxes from a Grozi annotation CSV next to the image, if there is one.
    With with_skus, returns (sku or None, box) pairs instead."""
    ann = img.parent.parent / "annotation" / f"{img.stem}.csv"
    if not ann.exists():
        return []
    w, h = size
    out = []
    with open(ann, newline="") as f:
        for r in csv.DictReader(f):
            box = [float(r["left_x"]) * w, float(r["top_y"]) * h, float(r["right_x"]) * w, float(r["bottom_y"]) * h]
            sku = None if r["label"] == "-1" else sku_id(int(r["class"]), int(r["label"]))
            out.append((sku, box) if with_skus else box)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "product_detector.pt"))
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "detections"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--identify", action="store_true", help="also name the product in each box")
    ap.add_argument("--bay", action="store_true", help="with --identify: restrict to the image's products + 20 distractors")
    ap.add_argument("--adapter", default=str(ROOT / "models" / "sku_adapter.npy"))
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = ProductDetector(a.weights, conf=a.conf)
    grozi = ROOT / "data" / "grozi"
    matcher = SkuMatcher.load(grozi / "sku_index.npz", a.adapter) if a.identify else None
    names = class_index(grozi) if a.identify else {}
    rng = random.Random(0)

    for img in collect(a.images):
        det = detector.detect(img.resolve(), rel_to=ROOT) if img.resolve().is_relative_to(ROOT) else detector.detect(img)
        # make the id unique across stores (Grozi images are just "15.jpg")
        stem = f"{img.parent.parent.name}_{img.stem}" if img.parent.name == "images" else img.stem
        det.image_id = stem

        im = Image.open(img).convert("RGB")
        if matcher is not None:
            cands = None
            if a.bay:
                known = set(matcher.index.unique_skus)
                gt = grozi_ground_truth(img, im.size, with_skus=True)
                cands = {s for s, _ in gt if s in known} | set(rng.sample(sorted(known), 20))
            identify_items(det, im, matcher, cands)
        save_detection_output(det, out_dir / f"{stem}.json")

        draw = ImageDraw.Draw(im)
        lw = max(4, im.width // 400)
        font = ImageFont.load_default(size=max(18, im.width // 55))
        for box in grozi_ground_truth(img, im.size):
            draw.rectangle(box, outline=(30, 90, 255), width=lw)
        for it in det.items:
            named = it.sku is not None
            colour = (0, 220, 60) if (named or matcher is None) else (255, 150, 0)
            draw.rectangle(it.bbox, outline=colour, width=lw)
            label = f"{it.det_confidence:.2f}"
            if matcher is not None:
                cls = names.get(int(it.sku[1:3]), "?").split("/")[-1] if named else "review"
                label = f"{cls} {it.sku[4:] if named else ''} ({it.sku_confidence:.2f})"
            tx, ty = it.bbox[0] + 8, it.bbox[1] + 8
            draw.rectangle(draw.textbbox((tx, ty), label, font=font), fill=(0, 0, 0))
            draw.text((tx, ty), label, fill=(255, 255, 0), font=font)
        im.thumbnail((1600, 1600))
        im.save(out_dir / f"{stem}.png")
        print(f"{stem}: {len(det.items)} items in {len(det.shelves)} rows")


if __name__ == "__main__":
    main()
