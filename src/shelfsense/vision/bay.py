"""Phases 1-4 in one pass over a close-up shelf bay with a known planogram.

  product-block detector -> name each block (SKU) -> align blocks to planogram slots
  pack detector          -> count the packs inside each block (its facings)

Every pack becomes its own item carrying its block's SKU and slot, so the report code's
"items in a slot = facings" rule gives stock-outs and low stock, and misplaced products,
from the same photo. Empty spaces come from a void detector when given one, else from gaps between packs in a row.
"""

import time
from contextlib import contextmanager
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


# Two pack boxes are pieces of one pack (a bottle's cap, neck and body, which the SKU-110K-trained detector
# boxes separately) when they share a column (horizontal overlap of at least SAME_COLUMN of the narrower box)
# and overlap vertically by at least SPLIT_OVERLAP of the shorter box. Stacked packs only touch, so they stay apart.
SAME_COLUMN = 0.7
SPLIT_OVERLAP = 0.25
# A bottle's neck can also be boxed apart from its body with a small vertical gap: a box at most NECK_WIDTH as
# wide as the box below it, lying (NECK_INSIDE of its width) within that box's span, with a gap of at most
# NECK_GAP of the lower box's height. Stacked packs are as wide as the pack under them, so they stay apart.
NECK_WIDTH = 0.6
NECK_INSIDE = 0.9
NECK_GAP = 0.2


def _neck_on_body(top: list[float], body: list[float]) -> bool:
    top_w, body_w = top[2] - top[0], body[2] - body[0]
    inside = min(top[2], body[2]) - max(top[0], body[0])
    return (
        (top[1] + top[3]) / 2 < (body[1] + body[3]) / 2
        and top_w <= NECK_WIDTH * body_w
        and inside >= NECK_INSIDE * top_w
        and body[1] - top[3] <= NECK_GAP * (body[3] - body[1])
        and top[3] <= body[3]
    )


def merge_split_packs(
    boxes: list[list[float]], confs: list[float]
) -> tuple[list[list[float]], list[float]]:
    """Merge boxes that are pieces of the same pack into one box (their union, best confidence)."""
    parent = list(range(len(boxes)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            overlapping = (
                ox >= SAME_COLUMN * min(a[2] - a[0], b[2] - b[0])
                and oy >= SPLIT_OVERLAP * min(a[3] - a[1], b[3] - b[1])
            )
            if overlapping or _neck_on_body(a, b) or _neck_on_body(b, a):
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(root(i), []).append(i)
    merged = [
        (
            [min(boxes[k][0] for k in g), min(boxes[k][1] for k in g), max(boxes[k][2] for k in g), max(boxes[k][3] for k in g)],
            max(confs[k] for k in g),
        )
        for g in groups.values()
    ]
    merged.sort(key=lambda bc: (bc[0][1], bc[0][0]))
    return [b for b, _ in merged], [c for _, c in merged]


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
    timings: dict[str, float] | None = None,
) -> DetectionOutput:
    """One photo, one planogram: slot-aligned, pack-level DetectionOutput with gaps.

    If `timings` is a dict, it is filled with the seconds each stage took (blocks, naming, alignment, packs, voids)."""

    @contextmanager
    def timed(stage: str):
        t = time.perf_counter()
        yield
        if timings is not None:
            timings[stage] = timings.get(stage, 0.0) + time.perf_counter() - t

    image_path = Path(image_path)
    with timed("find product blocks"):
        det = block_detector.detect(image_path)
    if shown_path is not None:
        det.image_path = shown_path
    image = Image.open(image_path).convert("RGB")
    cands = candidates if candidates is not None else {s.sku for sh in planogram.shelves for s in sh.slots}
    with timed("name each block"):
        identify_items(det, image, matcher, cands, min_confidence)
    with timed("align to the plan"):
        assign_slots(det, planogram)
    with timed("count packs"):
        packs, confs, _ = pack_detector.boxes(image_path)
        packs, confs = drop_fragments(*merge_split_packs(packs, confs))
        det = facings_output(det, packs, confs, produced_by=f"yolo:{pack_detector.name}")
    with timed("find empty spaces"):
        voids = [v for v in detect_voids(void_detector, image_path) if v[1] >= void_conf] if void_detector else None
        det = add_gaps(det, packs, voids)
    return det
