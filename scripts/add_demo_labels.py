"""Add price labels to the demo bays (phases 5-6 on the vision stage's output).

    python scripts/add_demo_labels.py [data/demo_bays/store1_36 ...]

Per bay: detect and read the labels on photo.jpg, tie them to slots using the
compliant detection, and write them into detection_compliant.json and
detection_scenario.json. Real Grozi shelves carry no catalogue prices we can
check against (assumption 7), so catalogue.csv takes each slot's price from the
label read confidently (OCR confidence >= 0.5), which makes the compliant shelf
price-clean, and catalogue_scenario.csv raises one such SKU's price by 10%, so
that label becomes a planted price mismatch.

Labels are read on the full-resolution Grozi original named in the bay's README.txt
(the bay's photo.jpg is downsized and its prices are mostly unreadable), then their
boxes are scaled down to photo.jpg's pixels, which is what the detection files use.
"""

import csv
import sys
from pathlib import Path

from PIL import Image

from shelfsense.contracts import load_detection_output, load_planogram, save_detection_output
from shelfsense.pricing.label_assign import assign_labels_to_slots
from shelfsense.pricing.label_detector import detect_labels

ROOT = Path(__file__).resolve().parents[1]
CONFIDENT = 0.5


def _write_catalogue(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _read_labels(bay: Path):
    photo = Image.open(bay / "photo.jpg")
    first = (bay / "README.txt").read_text().splitlines()[0]
    source = ROOT / first.split()[3] if first.startswith("Held-out Grozi shelf") else None
    if source is None or not source.exists():
        return detect_labels(photo)
    original = Image.open(source)
    fx, fy = photo.width / original.width, photo.height / original.height
    scaled = []
    for lb in detect_labels(original):
        x1, y1, x2, y2 = lb.bbox
        scaled.append(lb.model_copy(update={"bbox": [round(x1 * fx), round(y1 * fy), round(x2 * fx), round(y2 * fy)]}))
    return scaled


def process(bay: Path) -> None:
    compliant = load_detection_output(bay / "detection_compliant.json")
    planogram = load_planogram(bay / "planogram_compliant.json")
    labels = assign_labels_to_slots(_read_labels(bay), compliant, planogram)
    for name in ("compliant", "scenario"):
        det = load_detection_output(bay / f"detection_{name}.json")
        det.labels = labels
        save_detection_output(det, bay / f"detection_{name}.json")

    with (bay / "catalogue.csv").open() as f:
        reader = csv.DictReader(f)
        fieldnames, rows = list(reader.fieldnames), list(reader)
    by_sku = {r["sku"]: r for r in rows}
    confident = [
        lb for lb in labels
        if lb.sku in by_sku and lb.price is not None and (lb.ocr_confidence or 0) >= CONFIDENT
    ]
    for lb in confident:
        by_sku[lb.sku]["price"] = f"{lb.price:.2f}"
    _write_catalogue(bay / "catalogue.csv", rows, fieldnames)
    if confident:
        by_sku[confident[0].sku]["price"] = f"{confident[0].price * 1.1:.2f}"
    _write_catalogue(bay / "catalogue_scenario.csv", rows, fieldnames)
    print(f"{bay.name}: {len(labels)} labels, {sum(lb.slot_id is not None for lb in labels)} on a slot, "
          f"{len(confident)} read confidently")


if __name__ == "__main__":
    bays = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "data" / "demo_bays").iterdir())
    for b in bays:
        process(b)
