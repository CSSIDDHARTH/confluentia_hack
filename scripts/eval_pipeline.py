"""End-to-end check on the held-out Grozi stores: detect, then identify, then score.

    python scripts/eval_pipeline.py [--weights outputs/runs/grozi_v1/weights/best.pt]

A detected box counts as matched to a labelled product when IoU >= 0.5 (greedy, highest
confidence first). Reports how many labelled products were found and named correctly,
for open-set matching and for a planogram-restricted bay (the image's products plus 20
random distractors).
"""

import argparse
import csv
import random
from pathlib import Path

from PIL import Image

from shelfsense.vision.detect import ProductDetector
from shelfsense.vision.grozi import VAL_STORES
from shelfsense.vision.pipeline import DEFAULT_MIN_CONFIDENCE, identify_items
from shelfsense.vision.sku_match import SkuMatcher, sku_id

ROOT = Path(__file__).resolve().parents[1]
GROZI = ROOT / "data" / "grozi"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def ground_truth(img: Path, size):
    w, h = size
    ann = img.parent.parent / "annotation" / f"{img.stem}.csv"
    if not ann.exists():
        return []
    out = []
    with open(ann, newline="") as f:
        for r in csv.DictReader(f):
            sku = None if r["label"] == "-1" else sku_id(int(r["class"]), int(r["label"]))
            out.append((sku, [float(r["left_x"]) * w, float(r["top_y"]) * h, float(r["right_x"]) * w, float(r["bottom_y"]) * h]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models/product_detector.pt"))
    ap.add_argument("--adapter", default=str(ROOT / "models/sku_adapter.npy"))
    a = ap.parse_args()

    detector = ProductDetector(a.weights)
    matcher = SkuMatcher.load(GROZI / "sku_index.npz", a.adapter)
    known = set(matcher.index.unique_skus)
    rng = random.Random(0)
    all_skus = list(known)

    stats = {m: dict(labelled=0, found=0, named=0, correct=0, assigned_wrong=0) for m in ("open", "bay")}
    for store in VAL_STORES:
        for img in sorted((GROZI / "Testing" / store / "images").glob("*.jpg")):
            im = Image.open(img).convert("RGB")
            gt = [(s, b) for s, b in ground_truth(img, im.size) if s in known]
            if not gt:
                continue
            bay = {s for s, _ in gt} | set(rng.sample(all_skus, 20))
            for mode, cands in (("open", None), ("bay", bay)):
                det = identify_items(detector.detect(img), im, matcher, cands)
                used = set()
                st = stats[mode]
                st["labelled"] += len(gt)
                for item in sorted(det.items, key=lambda i: -i.det_confidence):
                    best, bj = 0.0, None
                    for j, (_, gb) in enumerate(gt):
                        if j in used:
                            continue
                        v = iou(item.bbox, gb)
                        if v > best:
                            best, bj = v, j
                    if best < 0.5:
                        continue
                    used.add(bj)
                    st["found"] += 1
                    if item.sku is not None:
                        st["named"] += 1
                        if item.sku == gt[bj][0]:
                            st["correct"] += 1
                        else:
                            st["assigned_wrong"] += 1

    print(f"min confidence for a name: {DEFAULT_MIN_CONFIDENCE}")
    for mode, st in stats.items():
        n = st["labelled"]
        print(
            f"{mode:5s} labelled products {n} | found by detector {st['found']} ({st['found']/n:.0%}) | "
            f"named {st['named']} | named correctly {st['correct']} ({st['correct']/n:.0%} of all labelled; "
            f"{st['correct']/max(st['named'], 1):.0%} of those named)"
        )


if __name__ == "__main__":
    main()
