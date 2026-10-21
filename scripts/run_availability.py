"""Find empty shelf space and low stock on shelf photos with the pack-level detector.

    python scripts/run_availability.py --images data/test_images [--weights models/pack_detector.pt]

Per image, in <out>/ (default outputs/availability): <id>.json (DetectionOutput: packs as items,
rows as shelves, gaps), <id>.report.json (AvailabilityReport) and <id>.jpg (packs in green, empty
spaces filled red, per-row fill %, and the replenishment actions under the photo).
"""

import argparse
from pathlib import Path

from PIL import Image

from shelfsense.contracts import save_detection_output
from shelfsense.vision.availability import analyze_packs
from shelfsense.vision.detect import ProductDetector
from shelfsense.vision.render import draw_availability

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "pack_detector.pt"))
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "availability"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None, help="e.g. cpu (default: GPU if available)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = ProductDetector(a.weights, conf=a.conf, imgsz=a.imgsz, max_det=600, device=a.device)
    paths = []
    for p in map(Path, a.images):
        paths += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]

    for img in paths:
        boxes, confs, size = detector.boxes(img)
        resolved = img.resolve()
        shown = str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)
        det, report = analyze_packs(img.stem, shown, size, boxes, confs, produced_by=f"yolo:{detector.name}")
        save_detection_output(det, out / f"{img.stem}.json")
        (out / f"{img.stem}.report.json").write_text(report.model_dump_json(indent=2))
        canvas = draw_availability(Image.open(img), det, report)
        canvas.thumbnail((1600, 2400))
        canvas.save(out / f"{img.stem}.jpg", quality=88)
        print(f"{img.stem}: {len(det.items)} packs, {len(det.shelves)} rows, {report.empty_spaces} empty spaces, fill {report.overall_fill:.0%}")


if __name__ == "__main__":
    main()
