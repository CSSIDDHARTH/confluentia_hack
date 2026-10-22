"""Measure empty-space detection against REAL hand-labelled empty shelf spots (Roboflow "Empty shelf detection").

    PYTHONPATH=src python3 scripts/eval_voids.py "<dataset dir>" [--void-weights models/void_detector.pt]

Dataset: Policenta, Roboflow Universe, CC BY 4.0, 1,050 images 416x416, one class `Void`. Unlike
eval_gaps.py (which compares against gaps derived from pack boxes) these labels are drawn by people on
the empty spots themselves. We evaluate on its valid+test splits, minus 50 images that are near-duplicates
of SKU-110K images our pack detector may have trained on (found by 16x16 thumbnail similarity > 0.9).

A predicted box matches a labelled void when IoU >= 0.3 (strict) or when they overlap by >= 50% of the
smaller box (lenient: a long thin void split in two still counts).
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image

from shelfsense.vision.gaps import find_gaps

ROOT = Path(__file__).resolve().parents[1]
SKU = ROOT / "data" / "sku110k_yolo" / "images"


def sig(path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("L").resize((16, 16)), float).ravel()
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-9)


def clean_images(root: Path, splits=("valid", "test")) -> list[Path]:
    ref = np.stack([sig(f) for f in sorted(glob.glob(str(SKU / "*" / "*.jpg")))])
    keep = []
    for split in splits:
        for f in sorted((root / split / "images").glob("*.jpg")):
            if (ref @ sig(f)).max() <= 0.9:
                keep.append(f)
    return keep


def load_labels(img: Path) -> list[list[float]]:
    w, h = Image.open(img).size
    lab = img.parent.parent / "labels" / f"{img.stem}.txt"
    out = []
    for ln in lab.read_text().splitlines() if lab.exists() else []:
        v = list(map(float, ln.split()))[1:]
        if len(v) == 4:  # box: cx cy w h
            x, y, bw, bh = v
            out.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
        else:  # polygon: x1 y1 x2 y2 ... -> its bounding box
            xs, ys = v[0::2], v[1::2]
            out.append([min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h])
    return out


def inter(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def iou(a, b) -> float:
    i = inter(a, b)
    return i / (area(a) + area(b) - i + 1e-9)


def lenient(a, b) -> float:
    return inter(a, b) / (min(area(a), area(b)) + 1e-9)


def match(pred, ref, score, thr) -> tuple[int, int, int]:
    used, tp = set(), 0
    for p in pred:
        best, bj = 0.0, None
        for j, r in enumerate(ref):
            if j not in used and (v := score(p, r)) > best:
                best, bj = v, j
        if best >= thr:
            tp += 1
            used.add(bj)
    return tp, len(pred) - tp, len(ref) - tp


def prf(tp, fp, fn):
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def report(name: str, preds: dict[str, list], refs: dict[str, list]) -> None:
    for label, fn, thr in (("strict IoU>=0.3", iou, 0.3), ("lenient overlap>=0.5", lenient, 0.5)):
        t = [0, 0, 0]
        for s in refs:
            t = [x + y for x, y in zip(t, match(preds.get(s, []), refs[s], fn, thr))]
        p, r, f = prf(*t)
        print(f"{name:34s} {label:22s} P {p:.2f} R {r:.2f} F1 {f:.2f}  (tp {t[0]} fp {t[1]} fn {t[2]})")


def geometry_preds(images, weights, imgsz, conf) -> dict[str, list]:
    from shelfsense.vision.detect import ProductDetector
    from shelfsense.vision.gaps import find_gaps

    det = ProductDetector(weights, conf=conf, imgsz=imgsz, max_det=600)
    out = {}
    for f in images:
        boxes, _, _ = det.boxes(f)
        out[f.stem] = [g.bbox for g in find_gaps(boxes)[1]]
    return out


# Tried and rejected (measured below): adding row gaps between merged packs that the void detector missed,
# kept only if a close look (the pack detector on the gap's crop) finds no pack there. It caught the plain gap
# between bottles on demo bay store4_9, but on these labelled photos it added 16 true and 253 false empty
# spaces (F1 0.64 -> 0.55; requiring gaps of 2+ pack widths: 0.60), so the chain uses the void detector alone.
# A close look at a suspected gap: the pack detector is re-run on the gap's crop (with this share of its width as
# margin on each side), and a pack at least CLOSE_LOOK_MIN_HEIGHT of the gap's height means the shelf is not empty
# there: the full-photo pass had missed it.
CLOSE_LOOK_MARGIN = 0.15
CLOSE_LOOK_MIN_HEIGHT = 0.25


def looks_empty(image: Image.Image, box: list[float], pack_detector) -> bool:
    """True when the pack detector finds no pack in a close-up of `box`."""
    m = CLOSE_LOOK_MARGIN * (box[2] - box[0])
    crop = image.crop((max(0, box[0] - m), box[1], min(image.width, box[2] + m), box[3]))
    res = pack_detector.model.predict(
        crop, conf=pack_detector.conf, imgsz=640, device=pack_detector.device, verbose=False
    )[0]
    return not any(b[3] - b[1] >= CLOSE_LOOK_MIN_HEIGHT * crop.height for b in res.boxes.xyxy.tolist())


def _overlaps(a: list[float], b: list[float], share: float = 0.5) -> bool:
    i = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return i >= share * min(area(a), area(b))


def confirmed_geometry_gaps(
    image: Image.Image,
    packs: list[list[float]],
    pack_detector,
    voids: list[tuple[list[float], float]] = (),
) -> list[tuple[list[float], float]]:
    """Stretches of a row no pack covers (find_gaps), not already found by the void detector, that still look
    empty on a close look. The void detector, trained on 416 px photos of other shops, misses some plain
    gaps between bottles; this adds them without taking a missed pack for a gap."""
    _, cands = find_gaps(packs)
    return [
        (g.bbox, g.confidence) for g in cands
        if not any(_overlaps(g.bbox, v) for v, _ in voids) and looks_empty(image, g.bbox, pack_detector)
    ]



def combined_preds(images, pack_weights, void_model) -> dict[str, list]:
    """The void detector at 0.3 plus confirmed geometry gaps (the rejected experiment above)."""
    from shelfsense.vision.bay import drop_fragments, merge_split_packs
    from shelfsense.vision.detect import ProductDetector

    det = ProductDetector(pack_weights, imgsz=1024, max_det=600)
    out = {}
    for f in images:
        voids = [(b, 0.3) for b in void_model.predict(str(f), imgsz=640, conf=0.3, verbose=False)[0].boxes.xyxy.tolist()]
        boxes, confs, _ = det.boxes(f)
        packs, _ = drop_fragments(*merge_split_packs(boxes, confs))
        extra = confirmed_geometry_gaps(Image.open(f).convert("RGB"), packs, det, voids)
        out[f.stem] = [b for b, _ in voids + extra]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--pack-weights", default=str(ROOT / "models" / "pack_detector.pt"))
    ap.add_argument("--void-weights", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--splits", nargs="+", default=["test"], help="valid is used to pick the trained model's best epoch, so test is the default")
    a = ap.parse_args()

    images = clean_images(Path(a.dataset), a.splits)
    refs = {f.stem: load_labels(f) for f in images}
    print(f"{len(images)} images from {'+'.join(a.splits)}, {sum(len(v) for v in refs.values())} labelled voids (near-duplicates of SKU-110K removed)\n")

    report("pack detector + gap geometry (ours)", geometry_preds(images, a.pack_weights, 1024, a.conf), refs)
    if a.void_weights:
        from ultralytics import YOLO

        m = YOLO(a.void_weights)
        for conf in (0.1, 0.2, 0.3, 0.4):
            preds = {f.stem: m.predict(str(f), imgsz=640, conf=conf, verbose=False)[0].boxes.xyxy.tolist() for f in images}
            report(f"trained void detector (conf {conf})", preds, refs)
        report("void 0.3 + confirmed geometry (rejected)", combined_preds(images, a.pack_weights, m), refs)


if __name__ == "__main__":
    main()
