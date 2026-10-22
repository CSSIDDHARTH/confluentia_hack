"""Find empty shelf space and low stock on shelf photos: a pack-level detector (rows, fill) and an
empty-space detector trained on hand-labelled voids.

    python scripts/run_availability.py --images data/test_images [--weights models/pack_detector.pt]

Per image, in <out>/ (default outputs/availability): <id>.json (DetectionOutput: packs as items,
rows as shelves, gaps), <id>.report.json (AvailabilityReport) and <id>.jpg (packs in green, empty
spaces filled red, per-row fill %, and the replenishment actions under the photo).
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from shelfsense.contracts import save_detection_output
from shelfsense.vision.availability import analyze_packs
from shelfsense.vision.detect import VOID_CONF_FLOOR, ProductDetector, detect_voids
from shelfsense.vision.render import draw_availability

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "pack_detector.pt"))
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "availability"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--void-weights", default=str(ROOT / "models" / "void_detector.pt"), help="empty-space detector; if the file is missing, gaps are inferred from missing packs instead")
    ap.add_argument("--void-conf", type=float, default=0.3, help="chosen on held-out labelled data (scripts/eval_voids.py)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None, help="e.g. cpu (default: GPU if available)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = ProductDetector(a.weights, conf=a.conf, imgsz=a.imgsz, max_det=600, device=a.device)
    voider = ProductDetector(a.void_weights, conf=VOID_CONF_FLOOR, imgsz=640, max_det=150, device=a.device) if Path(a.void_weights).exists() else None
    paths = []
    for p in map(Path, a.images):
        paths += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]

    for img in paths:
        boxes, confs, size = detector.boxes(img)
        resolved = img.resolve()
        shown = str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)
        voids = detect_voids(voider, img) if voider else None
        kept = [v for v in voids if v[1] >= a.void_conf] if voids is not None else None
        det, report = analyze_packs(img.stem, shown, size, boxes, confs, produced_by=f"yolo:{detector.name}", voids=kept)
        if voids is not None:  # every candidate, so the dashboard's confidence slider can recompute without a GPU
            (out / f"{img.stem}.voids.json").write_text(json.dumps([{"bbox": [round(x, 1) for x in b], "conf": round(c, 3)} for b, c in voids]))
        save_detection_output(det, out / f"{img.stem}.json")
        (out / f"{img.stem}.report.json").write_text(report.model_dump_json(indent=2))
        canvas = draw_availability(Image.open(img), det, report)
        canvas.thumbnail((1600, 2400))
        canvas.save(out / f"{img.stem}.jpg", quality=88)
        print(f"{img.stem}: {len(det.items)} packs, {len(det.shelves)} rows, {report.empty_spaces} empty spaces, fill {report.overall_fill:.0%}")


if __name__ == "__main__":
    main()
