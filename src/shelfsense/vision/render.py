"""Drawing shared by the scripts and the dashboard: the availability view and the planogram check."""

import textwrap

from PIL import Image, ImageDraw, ImageFont

from shelfsense.contracts import DetectionOutput
from shelfsense.report.compliance import ComplianceReport, ViolationKind
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


# One palette for the photo and the drawn plan, so a slot looks the same in both.
STATUS_COLOURS = {
    "ok": "#2e7d32",  # as planned
    "low": "#b86e00",  # low stock
    "bad": "#c62828",  # missing, or the wrong product
    "check": "#8d7b00",  # product not recognised
    "unread": "#5f6368",  # not read yet
    "label": "#1565c0",  # price label read and fine
}
_PACK_OUTLINE = {"ok": (60, 220, 90), "bad": (240, 50, 50), "check": (250, 205, 0)}


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
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _tag(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, colour: str, font, pad: int) -> None:
    x, y = xy
    l, t, r, b = d.textbbox((x + pad, y + pad), text, font=font)
    d.rectangle((x, y, r + pad, b + pad), fill=colour)
    d.text((x + pad, y + pad), text, fill=(255, 255, 255), font=font)


def draw_audit(
    im: Image.Image,
    det: DetectionOutput,
    report: ComplianceReport,
    expected: dict[str, str],
    label_notes: dict[str, tuple[str, str]],
) -> Image.Image:
    """The audit on the photo, in the plan's colours: each pack outlined green (the slot's product), red (another
    product) or yellow (not recognised); each slot tagged with its id and status; empty spaces filled red; each
    price label tagged with what was read (label_notes: label_id -> (text, status key)).
    expected: slot_id -> the sku the plan puts there."""
    im = im.convert("RGB")
    lw = max(3, im.width // 400)
    font = _font(max(18, im.width // 55))
    pad = max(4, font.size // 4)

    shade = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for g in det.gaps:
        sd.rectangle(g.bbox, fill=(220, 40, 40, 90), outline=(240, 50, 50, 255), width=lw + 1)
    im = Image.alpha_composite(im.convert("RGBA"), shade).convert("RGB")
    d = ImageDraw.Draw(im)

    for it in det.items:
        key = "check" if it.sku is None else ("ok" if expected.get(it.slot_id) == it.sku else "bad")
        d.rectangle(it.bbox, outline=_PACK_OUTLINE[key], width=lw)
    for g in det.gaps:
        _tag(d, (g.bbox[0] + lw, g.bbox[1] + lw), f"EMPTY ~{g.estimated_missing_facings}", STATUS_COLOURS["bad"], font, pad)

    status = slot_status(report)
    first: dict[str, list[float]] = {}
    for it in det.items:  # each slot's tag sits on its leftmost, highest pack
        if it.slot_id and (it.slot_id not in first or (it.bbox[0], it.bbox[1]) < (first[it.slot_id][0], first[it.slot_id][1])):
            first[it.slot_id] = it.bbox
    for slot_id, box in first.items():
        key, words = status.get(slot_id, ("ok", ""))
        _tag(d, (box[0] + lw, box[1] + lw), f"{slot_id} {words}".strip(), STATUS_COLOURS[key], font, pad)

    for lb in det.labels:
        text, key = label_notes.get(lb.label_id, (f"{lb.price:.2f}" if lb.price is not None else "?", "label"))
        d.rectangle(lb.bbox, outline=STATUS_COLOURS[key], width=lw)
        th = font.size + 2 * pad
        _tag(d, (lb.bbox[0], max(0, lb.bbox[1] - th - lw)), text, STATUS_COLOURS[key], font, pad)
    return im
