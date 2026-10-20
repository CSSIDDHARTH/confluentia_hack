"""Measure SKU matching on human-labelled Grozi boxes from the held-out stores.

    python scripts/eval_sku_match.py

Crops each labelled box, embeds it and looks up the nearest studio reference.
Reports exact-product top-1/top-5 and category-level top-1, for a few crop styles.
"""

import csv
from pathlib import Path

from PIL import Image

from shelfsense.vision.sku_match import Embedder, ReferenceIndex, class_index, sku_id

ROOT = Path(__file__).resolve().parents[1]
GROZI = ROOT / "data" / "grozi"
VAL_STORES = ["store1", "store4"]


def load_boxes():
    for store in VAL_STORES:
        for ann in sorted((GROZI / "Testing" / store / "annotation").glob("*.csv")):
            img = GROZI / "Testing" / store / "images" / f"{ann.stem}.jpg"
            with open(ann, newline="") as f:
                for r in csv.DictReader(f):
                    if r["label"] == "-1":
                        continue
                    yield img, int(r["class"]), int(r["label"]), [float(r[k]) for k in ("left_x", "top_y", "right_x", "bottom_y")]


def crop(im: Image.Image, rel: list[float], style: str) -> Image.Image:
    w, h = im.size
    x1, y1, x2, y2 = rel[0] * w, rel[1] * h, rel[2] * w, rel[3] * h
    if style == "first_pack":  # a block is a row of identical packs; keep roughly one
        bw, bh = x2 - x1, y2 - y1
        pack_w = min(bw, bh * 0.75)
        x2 = x1 + pack_w
    elif style == "centre_pack":
        bw, bh = x2 - x1, y2 - y1
        pack_w = min(bw, bh * 0.75)
        x1 = x1 + (bw - pack_w) / 2
        x2 = x1 + pack_w
    return im.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))


def main() -> None:
    index = ReferenceIndex.load(GROZI / "sku_index.npz")
    known = set(index.skus)
    emb = Embedder()
    boxes = [b for b in load_boxes() if sku_id(b[1], b[2]) in known]
    print(f"{len(boxes)} labelled val boxes whose product has a reference image")

    by_img: dict[Path, list] = {}
    for b in boxes:
        by_img.setdefault(b[0], []).append(b)

    for style in ("full", "first_pack", "centre_pack"):
        crops, truth = [], []
        for img, items in by_img.items():
            im = Image.open(img).convert("RGB")
            for _, c, lab, rel in items:
                crops.append(crop(im, rel, style))
                truth.append(sku_id(c, lab))
        res = index.query(emb.embed(crops), k=5)
        top1 = sum(r[0][0] == t for r, t in zip(res, truth)) / len(truth)
        top5 = sum(t in [s for s, _ in r] for r, t in zip(res, truth)) / len(truth)
        cat1 = sum(r[0][0].split("-")[0] == t.split("-")[0] for r, t in zip(res, truth)) / len(truth)
        print(f"{style:12s} exact top-1 {top1:.2f} | exact top-5 {top5:.2f} | category top-1 {cat1:.2f}")


if __name__ == "__main__":
    main()
