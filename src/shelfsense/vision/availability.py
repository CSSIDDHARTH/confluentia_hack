"""Phase 4: shelf availability report from pack-level detections.

Turns one box per pack into the standard DetectionOutput (packs as items, shelf rows,
gaps for empty spaces) plus an AvailabilityReport: per-row fill, empty spaces, low-stock
rows and plain-language replenishment actions. No planogram is needed, so it works on
any shelf photo; with a planogram, the slot alignment of phase 3 names the products.
"""

from datetime import datetime, timezone

from pydantic import BaseModel

from shelfsense.contracts import DetectionOutput, Gap, Item, ShelfRegion
from shelfsense.vision.gaps import LOW_FILL_RATIO, MIN_GAP_WIDTHS, find_gaps


class RowStatus(BaseModel):
    shelf_id: str
    packs: int
    fill_ratio: float
    low_stock: bool


class AvailabilityReport(BaseModel):
    image_id: str
    rows: list[RowStatus]
    empty_spaces: int
    missing_facings: int  # estimated packs needed to fill every empty space
    overall_fill: float  # mean fill ratio over rows, 0-1
    actions: list[str]  # most urgent first


def _where(x_center: float, left: float, right: float) -> str:
    third = (x_center - left) / max(right - left, 1e-6)
    return "left" if third < 1 / 3 else "right" if third > 2 / 3 else "middle"


def analyze_packs(
    image_id: str,
    image_path: str,
    image_size: tuple[int, int],
    boxes: list[list[float]],
    confidences: list[float],
    produced_by: str = "vision-0",
    low_fill: float = LOW_FILL_RATIO,
    min_gap_widths: float = MIN_GAP_WIDTHS,
) -> tuple[DetectionOutput, AvailabilityReport]:
    width, height = image_size
    clipped = [[max(0.0, b[0]), max(0.0, b[1]), min(float(width), b[2]), min(float(height), b[3])] for b in boxes]
    conf_of = {tuple(c): conf for c, conf in zip(clipped, confidences)}
    rows, gap_cands = find_gaps(clipped, min_gap_widths)

    shelves, items = [], []
    for row in rows:
        shelf_id = f"S{row.index + 1}"
        x1 = min(b[0] for b in row.boxes)
        x2 = max(b[2] for b in row.boxes)
        shelves.append(ShelfRegion(shelf_id=shelf_id, bbox=[x1, min(b[1] for b in row.boxes), x2, max(b[3] for b in row.boxes)]))
        for b in row.boxes:
            items.append(
                Item(
                    item_id=f"i{len(items) + 1:03d}", bbox=[round(v, 1) for v in b],
                    det_confidence=round(float(conf_of[tuple(b)]), 4), sku=None, sku_confidence=0.0,
                    shelf_id=shelf_id, slot_id=None,
                )
            )

    gaps = [
        Gap(
            gap_id=f"g{k + 1:03d}", bbox=[round(v, 1) for v in g.bbox], shelf_id=f"S{g.row + 1}", slot_id=None,
            confidence=g.confidence, estimated_missing_facings=g.estimated_missing,
        )
        for k, g in enumerate(gap_cands)
    ]

    det = DetectionOutput(
        image_id=image_id, image_path=image_path, image_width=width, image_height=height,
        planogram_id="unassigned", captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        produced_by={"vision": produced_by, "availability": "gaps-v1"}, shelves=shelves, items=items, gaps=gaps,
    )

    statuses = [
        RowStatus(shelf_id=f"S{r.index + 1}", packs=len(r.boxes), fill_ratio=round(r.fill_ratio, 3), low_stock=r.fill_ratio < low_fill)
        for r in rows
    ]
    bay_left = min((r.boxes[0][0] for r in rows), default=0.0)
    bay_right = max((max(b[2] for b in r.boxes) for r in rows), default=1.0)

    actions = []
    for g in sorted(gap_cands, key=lambda g: -g.estimated_missing):
        centre = (g.bbox[0] + g.bbox[2]) / 2
        n = g.estimated_missing
        actions.append(
            f"Shelf S{g.row + 1}: empty space on the {_where(centre, bay_left, bay_right)}, about {n} pack{'s' if n != 1 else ''} wide. Refill it."
        )
    for s in sorted((s for s in statuses if s.low_stock), key=lambda s: s.fill_ratio):
        actions.append(f"Shelf {s.shelf_id} is only {s.fill_ratio:.0%} stocked ({s.packs} packs). Replenish the whole row.")

    report = AvailabilityReport(
        image_id=image_id, rows=statuses, empty_spaces=len(gaps),
        missing_facings=sum(g.estimated_missing_facings for g in gaps),
        overall_fill=round(sum(s.fill_ratio for s in statuses) / len(statuses), 3) if statuses else 0.0,
        actions=actions,
    )
    return det, report
