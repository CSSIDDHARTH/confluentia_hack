"""Run the product-block detector on images and save detection_output JSON + an overlay.

    python scripts/run_detector.py --weights outputs/runs/grozi_v1/weights/best.pt \
        --images data/grozi/Testing/store1/images/15.jpg data/test_images [--out outputs/detections]

For every image: <out>/<id>.json (the DetectionOutput contract) and <out>/<id>.png with
predictions in green and, for Grozi images that have annotations, ground truth in blue.
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from shelfsense.contracts import save_detection_output
from shelfsense.vision.detect import ProductDetector

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    return out


def grozi_ground_truth(img: Path, size: tuple[int, int]) -> list[list[float]]:
    """Pixel boxes from a Grozi annotation CSV next to the image, if there is one."""
    ann = img.parent.parent / "annotation" / f"{img.stem}.csv"
    if not ann.exists():
        return []
    w, h = size
    with open(ann, newline="") as f:
        return [
            [float(r["left_x"]) * w, float(r["top_y"]) * h, float(r["right_x"]) * w, float(r["bottom_y"]) * h]
            for r in csv.DictReader(f)
        ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "detections"))
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = ProductDetector(a.weights, conf=a.conf)

    for img in collect(a.images):
        det = detector.detect(img.resolve(), rel_to=ROOT) if img.resolve().is_relative_to(ROOT) else detector.detect(img)
        # make the id unique across stores (Grozi images are just "15.jpg")
        stem = f"{img.parent.parent.name}_{img.stem}" if img.parent.name == "images" else img.stem
        det.image_id = stem
        save_detection_output(det, out_dir / f"{stem}.json")

        im = Image.open(img).convert("RGB")
        draw = ImageDraw.Draw(im)
        lw = max(4, im.width // 400)
        for box in grozi_ground_truth(img, im.size):
            draw.rectangle(box, outline=(30, 90, 255), width=lw)
        for it in det.items:
            draw.rectangle(it.bbox, outline=(0, 220, 60), width=lw)
            draw.text((it.bbox[0] + 6, it.bbox[1] + 6), f"{it.det_confidence:.2f}", fill=(255, 255, 0))
        im.thumbnail((1600, 1600))
        im.save(out_dir / f"{stem}.png")
        print(f"{stem}: {len(det.items)} items in {len(det.shelves)} rows")


if __name__ == "__main__":
    main()
