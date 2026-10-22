"""Measure pack counting (one box per pack) against hand-boxed packs, with and without the post-processing.

    PYTHONPATH=src python3 scripts/eval_packs.py            # the 200 held-out SKU-110K photos (data/sku110k_yolo/.../val)

Compares the pack detector's raw boxes, the boxes after drop_fragments (the old counting), and after
merge_split_packs + drop_fragments (bottle caps, necks and bodies merged into one pack). A predicted box is
correct when it matches an unmatched labelled pack with IoU >= 0.5. Count error is |predicted - labelled| /
labelled per photo.
"""

import argparse
from pathlib import Path
from statistics import mean, median

import numpy as np

from shelfsense.vision.bay import drop_fragments, merge_split_packs
from shelfsense.vision.detect import ProductDetector

ROOT = Path(__file__).resolve().parents[1]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda m: (m[:, 2] - m[:, 0]) * (m[:, 3] - m[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def matches(pred: list[list[float]], truth: list[list[float]], thr: float = 0.5) -> int:
    if not pred or not truth:
        return 0
    m = iou_matrix(np.array(pred), np.array(truth))
    hit = 0
    for i in np.argsort(-m.max(axis=1)):
        j = int(m[i].argmax())
        if m[i, j] >= thr:
            hit += 1
            m[:, j] = -1
    return hit


def labels(path: Path, w: int, h: int) -> list[list[float]]:
    out = []
    for line in path.read_text().split("\n"):
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=ROOT / "data" / "sku110k_yolo" / "images" / "val")
    a = ap.parse_args()
    from PIL import Image

    det = ProductDetector(ROOT / "models" / "pack_detector.pt", imgsz=1024, max_det=600)
    ways = {
        "raw boxes": lambda b, c: (b, c),
        "drop_fragments (old)": lambda b, c: drop_fragments(b, c),
        "merge + drop_fragments (new)": lambda b, c: drop_fragments(*merge_split_packs(b, c)),
    }
    tally = {k: {"tp": 0, "pred": 0, "truth": 0, "err": []} for k in ways}
    for img in sorted(a.images.glob("*.jpg")):
        w, h = Image.open(img).size
        truth = labels(img.parents[2] / "labels" / img.parent.name / f"{img.stem}.txt", w, h)
        boxes, confs, _ = det.boxes(img)
        for name, f in ways.items():
            pred, _ = f(boxes, confs)
            t = tally[name]
            t["tp"] += matches(pred, truth)
            t["pred"] += len(pred)
            t["truth"] += len(truth)
            t["err"].append(abs(len(pred) - len(truth)) / max(len(truth), 1))
    print(f"{len(tally['raw boxes']['err'])} photos, {tally['raw boxes']['truth']} labelled packs")
    for name, t in tally.items():
        p, r = t["tp"] / max(t["pred"], 1), t["tp"] / max(t["truth"], 1)
        print(f"  {name:30s} precision {p:.3f} recall {r:.3f} F1 {2 * p * r / max(p + r, 1e-9):.3f}  "
              f"count error mean {100 * mean(t['err']):.1f}% median {100 * median(t['err']):.1f}%")


if __name__ == "__main__":
    main()
