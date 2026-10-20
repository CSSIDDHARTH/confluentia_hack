"""Find empty shelf space and low stock on shelf photos with the pack-level detector.

    python scripts/run_availability.py --images data/test_images [--weights models/pack_detector.pt]

Per image, in <out>/ (default outputs/availability): <id>.json (DetectionOutput: packs as items,
rows as shelves, gaps), <id>.report.json (AvailabilityReport) and <id>.jpg (packs in green, empty
spaces filled red, per-row fill %, and the replenishment actions under the photo).
"""

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shelfsense.contracts import save_detection_output
from shelfsense.vision.availability import AvailabilityReport, analyze_packs
from shelfsense.vision.detect import ProductDetector

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def render(im: Image.Image, det, report: AvailabilityReport) -> Image.Image:
    im = im.convert("RGB")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for g in det.gaps:
        od.rectangle(g.bbox, fill=(214, 40, 40, 110), outline=(214, 40, 40, 255), width=max(4, im.width // 300))
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(16, im.width // 60))
    lw = max(2, im.width // 700)
    for it in det.items:
        d.rectangle(it.bbox, outline=(0, 220, 60), width=lw)
    for g in det.gaps:
        label = f"EMPTY ~{g.estimated_missing_facings}"
        tx, ty = g.bbox[0] + 6, g.bbox[1] + 6
        d.rectangle(d.textbbox((tx, ty), label, font=font), fill=(0, 0, 0))
        d.text((tx, ty), label, fill=(255, 255, 255), font=font)
    shelf = {s.shelf_id: s for s in det.shelves}
    for r in report.rows:
        x1, y1, _, _ = shelf[r.shelf_id].bbox
        label = f"{r.shelf_id}: {r.fill_ratio:.0%} full" + ("  LOW" if r.low_stock else "")
        d.rectangle(d.textbbox((8, y1 + 4), label, font=font), fill=(200, 0, 0) if r.low_stock else (0, 90, 40))
        d.text((8, y1 + 4), label, fill=(255, 255, 255), font=font)

    lines = [f"Overall fill {report.overall_fill:.0%} | empty spaces {report.empty_spaces} | about {report.missing_facings} packs to refill"]
    for a in report.actions:
        lines += textwrap.wrap("- " + a, width=max(40, im.width // (font.size // 2 + 1)))
    if not report.actions:
        lines.append("No replenishment needed.")
    line_h = font.size + 8
    canvas = Image.new("RGB", (im.width, im.height + line_h * len(lines) + 20), (20, 20, 20))
    canvas.paste(im, (0, 0))
    cd = ImageDraw.Draw(canvas)
    for k, line in enumerate(lines):
        cd.text((10, im.height + 10 + k * line_h), line, fill=(255, 255, 255), font=font)
    return canvas


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
        canvas = render(Image.open(img), det, report)
        canvas.thumbnail((1600, 2400))
        canvas.save(out / f"{img.stem}.jpg", quality=88)
        print(f"{img.stem}: {len(det.items)} packs, {len(det.shelves)} rows, {report.empty_spaces} empty spaces, fill {report.overall_fill:.0%}")


if __name__ == "__main__":
    main()
