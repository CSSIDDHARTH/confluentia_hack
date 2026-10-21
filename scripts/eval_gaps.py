"""Measure how well empty-space detection works, and tune its thresholds.

    python scripts/eval_gaps.py [--detect]      # --detect (re)runs the detector and caches its boxes

Reference: gaps found from the human-labelled pack boxes of the 200 held-out SKU-110K images
(dense and near-complete annotation, so an uncovered stretch wide enough for a pack is a real
empty space). Two checks:

  planted   remove one interior pack from a row of the annotation; does the gap finder see a gap there?
  detector  gaps from the detector's boxes against gaps from the annotation (precision / recall / F1,
            a predicted gap matches a reference gap when they overlap by >= 50% of the smaller one).

Tried and rejected: dropping gaps whose image region is as busy (edge density) as the row's packs, on the
idea that a missed product is textured and a real empty space is smooth. It only traded recall for
precision, with no F1 gain at any threshold.

Also rejected: a detector retrained on 1,600 images (sku110k_pack_v2): better mAP50 (0.897 vs 0.888)
but worse gap F1 (0.55 vs 0.67), so box-level mAP is not a safe proxy for gap quality.

The 200 images are split in two: thresholds are chosen on the first half and reported on the
second, so the final numbers are not tuned on the images they are measured on.
"""

import argparse
import itertools
import json
import random
from pathlib import Path

from PIL import Image

from shelfsense.vision.detect import ProductDetector
from shelfsense.vision.gaps import MIN_GAP_WIDTHS, ROW_VALLEY, cluster_rows, find_gaps

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data" / "sku110k_yolo"
CACHE = ROOT / "outputs" / "gap_eval_cache.json"
LOW_CONF = 0.05  # detect once at a low threshold, then filter by confidence offline


def load_gt(stem: str, size: tuple[int, int]) -> list[list[float]]:
    w, h = size
    boxes = []
    rows = [ln.split() for ln in (VAL / "labels" / "val" / f"{stem}.txt").read_text().splitlines() if ln.strip()]
    for _, cx, cy, bw, bh in rows:
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return boxes


def iou(a, b) -> float:
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def overlap(a, b) -> float:
    """Intersection area over the smaller box's area: tolerant of rows whose heights differ a little."""
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller else 0.0


def match(pred: list[list[float]], ref: list[list[float]], thr: float = 0.5) -> tuple[int, int, int]:
    used, tp = set(), 0
    for p in pred:
        best, bj = 0.0, None
        for j, r in enumerate(ref):
            if j not in used and (v := overlap(p, r)) > best:
                best, bj = v, j
        if best >= thr:
            tp += 1
            used.add(bj)
    return tp, len(pred) - tp, len(ref) - tp


def detect_all(stems: list[str], weights: str, imgsz: int) -> dict:
    det = ProductDetector(weights, conf=LOW_CONF, imgsz=imgsz, max_det=600)
    out = {}
    for k, stem in enumerate(stems):
        boxes, confs, size = det.boxes(VAL / "images" / "val" / f"{stem}.jpg")
        out[stem] = {"boxes": boxes, "confs": confs, "size": list(size)}
        if k % 40 == 0:
            print(f"  detected {k}/{len(stems)}", flush=True)
    return out


def gaps_of(boxes, **kw) -> list[list[float]]:
    return [g.bbox for g in find_gaps(boxes, **kw)[1]]


def planted_recall(stems, gt, rng, **kw) -> tuple[int, int, int]:
    """Sell out one whole product column: drop every pack in a well-filled row whose centre lies in the
    chosen pack's x-range (so stacked packs go too), and expect a gap there."""
    found = total = extra = 0
    for stem in stems:
        boxes = gt[stem]
        rows = [r for r in cluster_rows(boxes, valley=kw.get("valley", ROW_VALLEY)) if len(r.boxes) >= 8]
        if not rows:
            continue
        row = rng.choice(rows)
        victim = row.boxes[rng.randrange(2, len(row.boxes) - 2)]
        in_row = {id(b) for b in row.boxes}
        cx = lambda b: (b[0] + b[2]) / 2  # noqa: E731
        gone = {id(b) for b in row.boxes if victim[0] <= cx(b) <= victim[2]}
        without = [b for b in boxes if id(b) not in gone]
        base = len(gaps_of(boxes, **kw))
        new = gaps_of(without, **kw)
        total += 1
        found += any(g[0] <= cx(victim) <= g[2] and g[1] - 5 <= (victim[1] + victim[3]) / 2 <= g[3] + 5 for g in new)
        extra += max(0, len(new) - base - 1)
        del in_row
    return found, total, extra


def detector_scores(stems, gt, cache, conf, **kw) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for stem in stems:
        c = cache[stem]
        boxes = [b for b, s in zip(c["boxes"], c["confs"]) if s >= conf]
        a, b, d = match(gaps_of(boxes, **kw), gaps_of(gt[stem], **kw))
        tp, fp, fn = tp + a, fp + b, fn + d
    return tp, fp, fn


def prf(tp, fp, fn):
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", action="store_true", help="run the detector (needed once; cached afterwards)")
    ap.add_argument("--weights", default=str(ROOT / "models" / "pack_detector.pt"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--cache", default=str(CACHE), help="where detector boxes are cached (use one per setting)")
    a = ap.parse_args()

    stems = sorted(p.stem for p in (VAL / "images" / "val").glob("*.jpg"))
    sizes = {s: Image.open(VAL / "images" / "val" / f"{s}.jpg").size for s in stems}
    gt = {s: load_gt(s, sizes[s]) for s in stems}
    cache_path = Path(a.cache)
    cache = detect_all(stems, a.weights, a.imgsz) if a.detect or not cache_path.exists() else json.loads(cache_path.read_text())
    if a.detect:
        cache_path.write_text(json.dumps(cache))
    tune, test = stems[::2], stems[1::2]
    print(f"{len(stems)} images: {len(tune)} to tune on, {len(test)} to report on")

    rng = random.Random(0)
    print("\n-- planted gaps (annotation boxes, one interior pack removed) --")
    f, t, extra = planted_recall(test, gt, rng)
    print(f"current defaults (gap>={MIN_GAP_WIDTHS} widths, valley {ROW_VALLEY}): found {f}/{t} = {f / max(t, 1):.0%} | extra gaps created: {extra}")

    print("\n-- detector gaps vs annotation gaps (reported on the held-out half) --")
    base = detector_scores(test, gt, cache, 0.25)
    p, r, f1 = prf(*base)
    print(f"current defaults (conf 0.25): precision {p:.2f} recall {r:.2f} F1 {f1:.2f}  [tp {base[0]} fp {base[1]} fn {base[2]}]")

    grid = list(itertools.product([0.2, 0.25, 0.3], [0.6, 0.8, 1.0], [0.3, 0.4]))
    scored = []
    for conf, widths, valley in grid:
        s = prf(*detector_scores(tune, gt, cache, conf, min_gap_widths=widths, valley=valley))
        scored.append((s[2], conf, widths, valley, s))
    scored.sort(reverse=True)
    print("\ntop settings on the tuning half (conf, min gap widths, valley -> F1):")
    for f1_, conf, widths, valley, s in scored[:5]:
        print(f"  conf {conf:.2f} widths {widths:.1f} valley {valley:.1f} -> P {s[0]:.2f} R {s[1]:.2f} F1 {f1_:.2f}")
    _, conf, widths, valley, _ = scored[0]
    best = detector_scores(test, gt, cache, conf, min_gap_widths=widths, valley=valley)
    p, r, f1 = prf(*best)
    print(f"\nbest tuned setting (conf {conf}, widths {widths}, valley {valley}) on the HELD-OUT half: precision {p:.2f} recall {r:.2f} F1 {f1:.2f}  [tp {best[0]} fp {best[1]} fn {best[2]}]")
    f, t, extra = planted_recall(test, gt, random.Random(0), min_gap_widths=widths, valley=valley)
    print(f"planted gaps with the tuned setting: found {f}/{t} = {f / max(t, 1):.0%} | extra gaps created: {extra}")


if __name__ == "__main__":
    main()
