"""Drawing shared by the scripts and the dashboard: the availability view and the planogram check."""

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shelfsense.contracts import DetectionOutput
from shelfsense.report.compliance import ComplianceReport, ViolationKind
from shelfsense.vision.availability import AvailabilityReport


def draw_availability(
    im: Image.Image, det: DetectionOutput, report: AvailabilityReport, footer: bool = True
) -> Image.Image:
    """Packs outlined green, empty spaces filled red with 'EMPTY ~n', per-row fill % at each row's left
    edge (red when low stock) and, if footer, the replenishment actions under the photo."""
    im = im.convert("RGB")
    lw = max(2, im.width // 700)
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for g in det.gaps:
        od.rectangle(g.bbox, fill=(217, 48, 37, 80), outline=(234, 67, 53, 255), width=lw + 2)
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(im)
    font = _font(max(16, im.width // 60))
    pad = max(4, font.size // 3)
    for it in det.items:
        d.rectangle(it.bbox, outline=_PACK_OUTLINE["ok"], width=lw)
    for g in det.gaps:
        _tag(d, (g.bbox[0] + lw, g.bbox[1] + lw), f"EMPTY ~{g.estimated_missing_facings}", "bad", font, pad)
    shelf = {s.shelf_id: s for s in det.shelves}
    for r in report.rows:
        _, y1, _, _ = shelf[r.shelf_id].bbox
        _tag(d, (8, y1 + 4), f"{r.shelf_id}: {r.fill_ratio:.0%} full" + ("  LOW" if r.low_stock else ""),
             "bad" if r.low_stock else "ok", font, pad)
    if not footer:
        return im

    lines = [f"Overall fill {report.overall_fill:.0%} | empty spaces {report.empty_spaces} | about {report.missing_facings} packs to refill"]
    width = max(40, im.width // (font.size // 2 + 1))
    for w in report.warnings:
        lines += textwrap.wrap("WARNING: " + w, width=width)
    for a in report.actions:
        lines += textwrap.wrap("- " + a, width=width)
    if not report.actions and not report.warnings:
        lines.append("No replenishment needed.")
    line_h = font.size + 8
    canvas = Image.new("RGB", (im.width, im.height + line_h * len(lines) + 20), (20, 20, 20))
    canvas.paste(im, (0, 0))
    cd = ImageDraw.Draw(canvas)
    for k, line in enumerate(lines):
        cd.text((10, im.height + 10 + k * line_h), line, fill=(255, 255, 255), font=font)
    return canvas


def draw_bay_report(im: Image.Image, det: DetectionOutput, report: ComplianceReport, title: str, footer: bool = True) -> Image.Image:
    """A planogram check drawn on the photo: packs coloured by problem (red misplaced, orange unrecognised), empty spaces in red, banner on top and
    the recommended actions under the photo."""
    # every pack of a flagged product: (slot, sku) for misplaced, (slot, None) for unrecognised
    bad = {(v.slot_id, v.sku): v.kind for v in report.violations if v.kind in (ViolationKind.MISPLACED, ViolationKind.UNRECOGNISED_ITEM)}
    colours = {ViolationKind.MISPLACED: (214, 40, 40), ViolationKind.UNRECOGNISED_ITEM: (240, 160, 32)}
    im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(16, im.width // 60))
    lw = max(3, im.width // 450)
    for it in det.items:
        d.rectangle(it.bbox, outline=colours.get(bad.get((it.slot_id, it.sku)), (46, 158, 68)), width=lw)
    for g in det.gaps:
        d.rectangle(g.bbox, outline=(214, 40, 40), width=lw + 2)
        d.text((g.bbox[0] + 6, g.bbox[1] + 4), f"EMPTY ~{g.estimated_missing_facings}", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    seen = set()
    for it in det.items:  # one slot label per slot, on its leftmost pack
        if it.slot_id and it.slot_id not in seen:
            seen.add(it.slot_id)
            n = sum(1 for o in det.items if o.slot_id == it.slot_id)
            label = f"{it.slot_id} x{n}"
            tx, ty = it.bbox[0] + 6, it.bbox[1] + 6
            d.rectangle(d.textbbox((tx, ty), label, font=font), fill=(0, 0, 0))
            d.text((tx, ty), label, fill=(255, 255, 0), font=font)

    if not footer:
        return im
    head = f"{title}: score {report.score:.0f}  " + "  ".join(f"{k.replace('_', ' ')} {n}" for k, n in report.counts.items())
    lines = [f"- {v.recommended_action}" for v in report.violations[:8]]
    wrapped = [w for ln in lines for w in textwrap.wrap(ln, width=max(40, im.width // (font.size // 2 + 1)))]
    line_h = font.size + 8
    top = font.size + 24
    canvas = Image.new("RGB", (im.width, top + im.height + line_h * len(wrapped) + 16), (20, 20, 20))
    canvas.paste(im, (0, top))
    cd = ImageDraw.Draw(canvas)
    cd.text((10, 10), head, fill=(255, 255, 255), font=font)
    for k, ln in enumerate(wrapped):
        cd.text((10, top + im.height + 8 + k * line_h), ln, fill=(255, 255, 255), font=font)
    return canvas


# One palette for the photo and the drawn plan, so a slot looks the same in both: Google's Material colours.
# STATUS: solid colour (tags and outlines on the photo); TONAL: (background, text) for chips and plan tiles.
STATUS = {
    "ok": "#188038",  # as planned
    "low": "#e8710a",  # low stock
    "bad": "#d93025",  # missing, or the wrong product
    "check": "#f9ab00",  # product not recognised, label to check
    "unread": "#80868b",  # not read yet
    "label": "#1a73e8",  # price label read and fine
}
TONAL = {
    "ok": ("#e6f4ea", "#137333"),
    "low": ("#feefe3", "#b3480b"),
    "bad": ("#fce8e6", "#c5221f"),
    "check": ("#fef7e0", "#8a5a00"),
    "unread": ("#f1f3f4", "#5f6368"),
    "label": ("#e8f0fe", "#1967d2"),
}
_TAG_TEXT = {"check": (32, 33, 36)}  # dark text on yellow, white on the rest
_PACK_OUTLINE = {"ok": (52, 168, 83), "bad": (234, 67, 53), "check": (251, 188, 4)}
_FONTS = (
    Path(__file__).resolve().parents[3] / "app" / "static" / "fonts" / "Roboto-Medium.ttf",
    Path("/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Medium.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)


def slot_status(report: ComplianceReport) -> dict[str, tuple[str, str]]:
    """slot_id -> (status key, words) from the report's violations, worst first."""
    order = [  # a slot holding another product is "wrong product"; "missing" means nothing is there at all
        (ViolationKind.MISPLACED, "bad", "wrong product"),
        (ViolationKind.STOCK_OUT, "bad", "missing"),
        (ViolationKind.LOW_STOCK, "low", "low stock"),
        (ViolationKind.UNRECOGNISED_ITEM, "check", "check product"),
    ]
    kinds: dict[str, set] = {}
    for v in report.violations:
        kinds.setdefault(v.slot_id, set()).add(v.kind)
    out = {}
    for slot_id, here in kinds.items():
        for kind, key, words in order:
            if kind in here:
                out[slot_id] = (key, words)
                break
    return out


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONTS:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _tag(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, key: str, font, pad: int) -> list[float]:
    """A rounded label with the status colour; returns its box."""
    x, y = xy
    _, _, r, b = d.textbbox((x + pad, y + pad), text, font=font)
    box = [x, y, r + pad, b + pad]
    d.rounded_rectangle(box, radius=pad, fill=STATUS[key])
    d.text((x + pad, y + pad), text, fill=_TAG_TEXT.get(key, (255, 255, 255)), font=font)
    return box


def draw_audit(
    im: Image.Image,
    det: DetectionOutput,
    report: ComplianceReport,
    expected: dict[str, str],
    label_notes: dict[str, tuple[str, str]],
    highlight: set[str] | None = None,
) -> Image.Image:
    """The audit on the photo, in the plan's colours: each pack outlined green (the slot's product), red (another
    product) or yellow (not recognised); each slot tagged with its id and status; empty spaces filled red; each
    price label tagged with what was read (label_notes: label_id -> (text, status key)).
    expected: slot_id -> the sku the plan puts there. highlight: slot ids to keep bright, the rest is faded."""
    im = im.convert("RGB")
    lw = max(3, im.width // 400)
    font = _font(max(18, im.width // 55))
    pad = max(5, font.size // 3)

    shade = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for g in det.gaps:
        sd.rectangle(g.bbox, fill=(217, 48, 37, 80), outline=(234, 67, 53, 255), width=lw + 1)
    im = Image.alpha_composite(im.convert("RGBA"), shade).convert("RGB")
    d = ImageDraw.Draw(im)

    for it in det.items:
        key = "check" if it.sku is None else ("ok" if expected.get(it.slot_id) == it.sku else "bad")
        d.rectangle(it.bbox, outline=_PACK_OUTLINE[key], width=lw)
    for g in det.gaps:
        _tag(d, (g.bbox[0] + lw, g.bbox[1] + lw), f"EMPTY ~{g.estimated_missing_facings}", "bad", font, pad)

    status = slot_status(report)
    regions: dict[str, list[float]] = {}  # each slot's area on the photo, for highlighting
    first: dict[str, list[float]] = {}
    for it in det.items:  # each slot's tag sits on its leftmost, highest pack
        if not it.slot_id:
            continue
        r = regions.setdefault(it.slot_id, list(it.bbox))
        r[:] = [min(r[0], it.bbox[0]), min(r[1], it.bbox[1]), max(r[2], it.bbox[2]), max(r[3], it.bbox[3])]
        if it.slot_id not in first or (it.bbox[0], it.bbox[1]) < (first[it.slot_id][0], first[it.slot_id][1]):
            first[it.slot_id] = it.bbox
    for slot_id, box in first.items():
        key, words = status.get(slot_id, ("ok", ""))
        _tag(d, (box[0] + lw, box[1] + lw), f"{slot_id} {words}".strip(), key, font, pad)

    for lb in det.labels:
        text, key = label_notes.get(lb.label_id, (f"{lb.price:.2f}" if lb.price is not None else "?", "label"))
        d.rectangle(lb.bbox, outline=STATUS[key], width=lw)
        th = font.size + 2 * pad
        tag = _tag(d, (lb.bbox[0], max(0, lb.bbox[1] - th - lw)), text, key, font, pad)
        if lb.slot_id:
            r = regions.setdefault(lb.slot_id, list(lb.bbox))
            r[:] = [min(r[0], tag[0]), min(r[1], tag[1]), max(r[2], lb.bbox[2]), max(r[3], lb.bbox[3])]

    if highlight:
        faded = Image.blend(im, Image.new("RGB", im.size, (255, 255, 255)), 0.65)
        m = 2 * lw
        for slot_id in highlight:
            if slot_id in regions:
                x1, y1, x2, y2 = regions[slot_id]
                box = (max(0, int(x1 - m)), max(0, int(y1 - m)), min(im.width, int(x2 + m)), min(im.height, int(y2 + m)))
                faded.paste(im.crop(box), box[:2])
        im = faded
    return im
