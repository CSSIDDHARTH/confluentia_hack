"""Measure the trained label detector end to end on the hand-boxed Grozi labels.

For every full shelf photo that scripts/prep_grozi_label_fixtures.py boxed,
run detect_labels() and report, per hand-boxed label:
  found   - a detected box overlaps it (IoU >= 0.3)
  correct - and the OCR'd price equals the transcribed price

    python scripts/eval_label_detector.py [path/to/best.pt]
"""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from prep_grozi_label_fixtures import RAW, SELECTION  # noqa: E402

from shelfsense.pricing.label_detector import DEFAULT_WEIGHTS, detect_labels  # noqa: E402


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main() -> None:
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    shelves: dict[tuple[int, str], list] = {}
    for store, image, bbox, truth in SELECTION:
        shelves.setdefault((store, image), []).append((bbox, truth))

    found = correct = wrong = total = n_detected = 0
    for (store, image), truths in shelves.items():
        im = Image.open(RAW / f"store{store}" / "images" / f"{image}.jpg").convert("RGB")
        labels = detect_labels(im, weights)
        n_detected += len(labels)
        for bbox, truth in truths:
            total += 1
            best = max(labels, key=lambda l: iou(l.bbox, bbox), default=None)
            if best is None or iou(best.bbox, bbox) < 0.3:
                continue
            found += 1
            if best.price is not None and abs(best.price - float(truth)) < 0.005:
                correct += 1
            elif best.price is not None:
                wrong += 1
        print(f"store{store}/{image}: {len(labels)} detected, {len(truths)} boxed")
    print(f"found {found}/{total}, price correct {correct}/{total}, wrong {wrong}; {n_detected} boxes detected in all")


if __name__ == "__main__":
    main()
