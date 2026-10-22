"""Drawing shared by the scripts and the dashboard: the availability view and the planogram check."""

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shelfsense.contracts import DetectionOutput
from shelfsense.report.compliance import ComplianceReport, ViolationKind
from shelfsense.vision.availability import AvailabilityReport


# The dashboard shows photos shrunk to fit this box; tags and lines are sized for that, not for the original.
SHOWN_BOX = (1100, 760)


def _shown_scale(im: Image.Image) -> float:
    """Original pixels per on-screen pixel when the photo is shown in SHOWN_BOX (at least 1)."""
    return max(1.0, im.width / SHOWN_BOX[0], im.height / SHOWN_BOX[1])


def draw_availability(
    im: Image.Image, det: DetectionOutput, report: AvailabilityReport, footer: bool = True
) -> Image.Image:
    """Packs outlined green, empty spaces filled red with 'EMPTY ~n', per-row fill % at each row's left
    edge (red when low stock) and, if footer, the replenishment actions under the photo."""
    im = im.convert("RGB")
    k = _shown_scale(im)
    lw = max(2, round(2 * k))
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for g in det.gaps:
        od.rectangle(g.bbox, fill=(217, 48, 37, 80), outline=(234, 67, 53, 255), width=lw + 2)
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(im)
    font = _font(round(16 * k))  # about 16 px on screen
    pad = max(4, font.size // 3)
    for it in det.items:
        d.rectangle(it.bbox, outline=_PACK_OUTLINE["ok"], width=lw)
    # each shelf row: a bar down the photo's left edge over the row's height, and a badge with its name and fill
    shelf = {s.shelf_id: s for s in det.shelves}
    bar = max(6, round(7 * k))
    taken: list[list[float]] = []
    for r in report.rows:
        _, y1, _, y2 = shelf[r.shelf_id].bbox
        key = "bad" if r.low_stock else "ok"
        d.rectangle((0, y1, bar, y2), fill=STATUS[key])
        _tag(d, (bar + lw, y1 + lw), f"{r.shelf_id} · {r.fill_ratio:.0%} full" + (" · LOW" if r.low_stock else ""),
             key, font, pad, taken, im.size)
    for g in det.gaps:
        _tag_centred(d, g.bbox, f"EMPTY ~{g.estimated_missing_facings}", "bad", font, pad, im.size, taken)
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
    "ok": "#146c2e",  # as planned
    "low": "#a85300",  # low stock
    "bad": "#b3261e",  # missing, or the wrong product
    "check": "#f9ab00",  # product not recognised, label to check
    "unread": "#747775",  # not read yet
    "label": "#0b57d0",  # price label read and fine
}
TONAL = {
    "ok": ("#c4eed0", "#0a3818"),
    "low": ("#ffdcbe", "#5a2800"),
    "bad": ("#f9dedc", "#8c1d18"),
    "check": ("#ffefc4", "#4d3800"),
    "unread": ("#eef0f4", "#444746"),
    "label": ("#d3e3fd", "#041e49"),
}
_TAG_TEXT = {"check": (31, 31, 31)}  # dark text on yellow, white on the rest
_PACK_OUTLINE = {"ok": (52, 168, 83), "bad": (234, 67, 53), "check": (251, 188, 4)}
_FONTS = (  # Google Sans Flex, served by the app (app/static/fonts), then fallbacks
    Path(__file__).resolve().parents[3] / "app" / "static" / "fonts" / "GoogleSansFlex-Medium.ttf",
    Path(__file__).resolve().parents[3] / "app" / "static" / "fonts" / "Roboto-Medium.ttf",
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


def _tag(
    d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, key: str, font, pad: int,
    taken: list[list[float]] | None = None, size: tuple[int, int] | None = None,
) -> list[float]:
    """A rounded label with the status colour; returns its box. With `taken` (boxes of tags already drawn) it
    slides right of any tag in its way, or below it when there is no room, and is added to `taken`."""
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    w, h = r - l + 2 * pad, b - t + 2 * pad
    x0, (x, y) = xy[0], xy
    for _ in range(40 if taken else 0):
        hit = next((o for o in taken if x < o[2] and o[0] < x + w and y < o[3] and o[1] < y + h), None)
        if hit is None:
            break
        if size is None or hit[2] + 2 + w <= size[0]:
            x = hit[2] + 2
        else:
            x, y = x0, hit[3] + 2
    if size is not None:
        x, y = min(max(0, x), size[0] - w), min(max(0, y), size[1] - h)
    box = [x, y, x + w, y + h]
    d.rounded_rectangle(box, radius=pad, fill=STATUS[key])
    d.text((x + pad - l, y + pad - t), text, fill=_TAG_TEXT.get(key, (255, 255, 255)), font=font)
    if taken is not None:
        taken.append(box)
    return box


def _tag_centred(
    d: ImageDraw.ImageDraw, box: list[float], text: str, key: str, font, pad: int, size: tuple[int, int],
    taken: list[list[float]] | None = None,
) -> None:
    """A tag in the middle of box (kept inside the photo, and clear of the tags in `taken`)."""
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    w, h = r - l + 2 * pad, b - t + 2 * pad
    _tag(d, ((box[0] + box[2]) / 2 - w / 2, (box[1] + box[3]) / 2 - h / 2), text, key, font, pad, taken, size)


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
    k = _shown_scale(im)
    lw = max(2, round(2 * k))
    font = _font(round(15 * k))  # about 15 px on screen
    pad = max(4, font.size // 3)

    shade = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for g in det.gaps:
        sd.rectangle(g.bbox, fill=(217, 48, 37, 80), outline=(234, 67, 53, 255), width=lw + 1)
    im = Image.alpha_composite(im.convert("RGBA"), shade).convert("RGB")
    d = ImageDraw.Draw(im)

    for it in det.items:
        key = "check" if it.sku is None else ("ok" if expected.get(it.slot_id) == it.sku else "bad")
        d.rectangle(it.bbox, outline=_PACK_OUTLINE[key], width=lw)
    status = slot_status(report)
    taken: list[list[float]] = []
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
        _tag(d, (box[0] + lw, box[1] + lw), f"{slot_id} {words}".strip(), key, font, pad, taken, im.size)

    for lb in det.labels:
        text, key = label_notes.get(lb.label_id, (f"{lb.price:.2f}" if lb.price is not None else "?", "label"))
        d.rectangle(lb.bbox, outline=STATUS[key], width=lw)
        th = font.size + 2 * pad
        tag = _tag(d, (lb.bbox[0], max(0, lb.bbox[1] - th - lw)), text, key, font, pad, taken, im.size)
        if lb.slot_id:
            r = regions.setdefault(lb.slot_id, list(lb.bbox))
            r[:] = [min(r[0], tag[0]), min(r[1], tag[1]), max(r[2], lb.bbox[2]), max(r[3], lb.bbox[3])]

    for g in det.gaps:
        _tag_centred(d, g.bbox, f"EMPTY ~{g.estimated_missing_facings}", "bad", font, pad, im.size, taken)

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
