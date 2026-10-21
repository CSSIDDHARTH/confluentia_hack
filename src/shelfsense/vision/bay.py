"""Phases 1-4 in one pass over a close-up shelf bay with a known planogram.

  product-block detector -> name each block (SKU) -> align blocks to planogram slots
  pack detector          -> count the packs inside each block (its facings)

Every pack becomes its own item carrying its block's SKU and slot, so the report code's
"items in a slot = facings" rule gives stock-outs and low stock, and misplaced products,
from the same photo. Empty spaces come from a void detector when given one, else from gaps between packs in a row.
"""

from pathlib import Path
from statistics import median

from PIL import Image

from shelfsense.contracts import DetectionOutput, Gap, Item, Planogram
from shelfsense.vision.detect import VOID_CONF, VOID_CONF_FLOOR, ProductDetector, detect_voids
from shelfsense.vision.gaps import find_gaps, voids_as_gaps
from shelfsense.vision.pipeline import DEFAULT_MIN_CONFIDENCE, identify_items
from shelfsense.vision.planogram_align import assign_slots
from shelfsense.vision.sku_match import SkuMatcher

# A pack box smaller than this share of the median pack area is a fragment (a bottle
# neck, a cap, a sliver of a neighbour) and is not counted as a facing.
MIN_PACK_AREA_RATIO = 0.25


def _area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def drop_fragments(
    boxes: list[list[float]], confs: list[float], min_ratio: float = MIN_PACK_AREA_RATIO
) -> tuple[list[list[float]], list[float]]:
    if not boxes:
        return [], []
    cutoff = min_ratio * median(_area(b) for b in boxes)
    kept = [(b, c) for b, c in zip(boxes, confs) if _area(b) >= cutoff]
    return [b for b, _ in kept], [c for _, c in kept]


def assign_packs_to_blocks(blocks: list[list[float]], packs: list[list[float]]) -> list[int | None]:
    """For each pack, the index of the smallest block containing its centre (None if outside all)."""
    out: list[int | None] = []
    for p in packs:
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        best, best_area = None, float("inf")
        for i, b in enumerate(blocks):
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3] and _area(b) < best_area:
                best, best_area = i, _area(b)
        out.append(best)
    return out


def facings_output(
    aligned: DetectionOutput, packs: list[list[float]], confs: list[float], produced_by: str = "packs"
) -> DetectionOutput:
    """Replace block items with one item per pack, each inheriting its block's sku and slot.

    A block with no pack inside it (the pack detector missed it) stays as a single item, so a
    detector miss does not turn into a false stock-out.
    """
    blocks = aligned.items
    owner = assign_packs_to_blocks([b.bbox for b in blocks], packs)
    items: list[Item] = []
    for bi, block in enumerate(blocks):
        mine = [(p, c) for p, c, o in zip(packs, confs, owner) if o == bi]
        if not mine:
            items.append(block.model_copy())
            continue
        for p, c in sorted(mine, key=lambda pc: pc[0][0]):
            items.append(
                Item(
                    item_id="", bbox=[round(v, 1) for v in p], det_confidence=round(float(c), 4),
                    sku=block.sku, sku_confidence=block.sku_confidence,
                    shelf_id=block.shelf_id, slot_id=block.slot_id,
                )
            )
    for n, it in enumerate(items, start=1):
        it.item_id = f"i{n:03d}"
    out = aligned.model_copy(update={"items": items})
    out.produced_by = {**aligned.produced_by, "facings": produced_by}
    return out


def _shelf_for(y_centre: float, det: DetectionOutput) -> str | None:
    best, best_dist = None, float("inf")
    for s in det.shelves:
        dist = 0.0 if s.bbox[1] <= y_centre <= s.bbox[3] else min(abs(y_centre - s.bbox[1]), abs(y_centre - s.bbox[3]))
        if dist < best_dist:
            best, best_dist = s.shelf_id, dist
    return best


def add_gaps(
    det: DetectionOutput, packs: list[list[float]], voids: list[tuple[list[float], float]] | None = None
) -> DetectionOutput:
    """Empty spaces for the bay. Gaps are tied to a shelf, not a slot.

    With `voids` (from a detector trained on labelled empty spots) those are the empty spaces; without, they are
    inferred from stretches of a row no pack covers (a far weaker method, kept as the fallback)."""
    rows, cands = find_gaps(packs)
    if voids is not None:
        cands = voids_as_gaps(rows, voids)
    det.gaps = [
        Gap(
            gap_id=f"g{k + 1:03d}", bbox=[round(v, 1) for v in g.bbox],
            shelf_id=_shelf_for((g.bbox[1] + g.bbox[3]) / 2, det), slot_id=None,
            confidence=g.confidence, estimated_missing_facings=g.estimated_missing,
        )
        for k, g in enumerate(cands)
    ]
    return det


def analyze_bay(
    image_path: str | Path,
    block_detector: ProductDetector,
    pack_detector: ProductDetector,
    matcher: SkuMatcher,
    planogram: Planogram,
    candidates: set[str] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    shown_path: str | None = None,
    void_detector: ProductDetector | None = None,
    void_conf: float = VOID_CONF,
) -> DetectionOutput:
    """One photo, one planogram: slot-aligned, pack-level DetectionOutput with gaps."""
    image_path = Path(image_path)
    det = block_detector.detect(image_path)
    if shown_path is not None:
        det.image_path = shown_path
    image = Image.open(image_path).convert("RGB")
    cands = candidates if candidates is not None else {s.sku for sh in planogram.shelves for s in sh.slots}
    identify_items(det, image, matcher, cands, min_confidence)
    assign_slots(det, planogram)
    packs, confs, _ = pack_detector.boxes(image_path)
    packs, confs = drop_fragments(packs, confs)
    det = facings_output(det, packs, confs, produced_by=f"yolo:{pack_detector.name}")
    voids = [v for v in detect_voids(void_detector, image_path) if v[1] >= void_conf] if void_detector else None
    return add_gaps(det, packs, voids)
