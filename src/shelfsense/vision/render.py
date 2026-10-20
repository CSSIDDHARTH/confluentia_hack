"""Drawing for the availability view, shared by scripts/run_availability.py and the dashboard page."""

import textwrap

from PIL import Image, ImageDraw, ImageFont

from shelfsense.contracts import DetectionOutput
from shelfsense.vision.availability import AvailabilityReport


def draw_availability(
    im: Image.Image, det: DetectionOutput, report: AvailabilityReport, footer: bool = True
) -> Image.Image:
    """Packs in green, empty spaces filled red with 'EMPTY ~n', per-row fill % at each row's left
    edge (red when low stock) and, if footer, the replenishment actions under the photo."""
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
    if not footer:
        return im

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
