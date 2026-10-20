"""Phase 1 vision stage: find product blocks on a shelf photo and emit a
DetectionOutput.

At this stage items have no SKU (phase 2) and no slot (phase 3); shelves are a
heuristic grouping of items into rows, which phase 3 will refine.
"""

from datetime import datetime, timezone
from pathlib import Path

from shelfsense.contracts import DetectionOutput, Item, ShelfRegion

UNASSIGNED_PLANOGRAM = "unassigned"


def group_into_rows(boxes: list[list[float]]) -> list[list[int]]:
    """Group box indices into shelf rows, top row first.

    Boxes are sorted by vertical centre and a box joins the current row while
    its centre lies inside the row's vertical extent; otherwise it starts a new
    row. Rows are returned top-to-bottom, boxes left-to-right.
    """
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    rows: list[list[int]] = []
    row_top = row_bottom = 0.0
    for i in order:
        cy = (boxes[i][1] + boxes[i][3]) / 2
        if rows and row_top <= cy <= row_bottom:
            rows[-1].append(i)
            row_top = min(row_top, boxes[i][1])
            row_bottom = max(row_bottom, boxes[i][3])
        else:
            rows.append([i])
            row_top, row_bottom = boxes[i][1], boxes[i][3]
    return [sorted(r, key=lambda i: boxes[i][0]) for r in rows]


def build_detection_output(
    image_id: str,
    image_path: str,
    image_size: tuple[int, int],
    boxes: list[list[float]],
    confidences: list[float],
    planogram_id: str = UNASSIGNED_PLANOGRAM,
    produced_by: str = "vision-0",
) -> DetectionOutput:
    """Turn raw detections ([x1, y1, x2, y2] pixels + confidence) into the contract."""
    width, height = image_size
    clipped = [
        [max(0.0, b[0]), max(0.0, b[1]), min(float(width), b[2]), min(float(height), b[3])]
        for b in boxes
    ]
    rows = group_into_rows(clipped)

    shelves, items = [], []
    for row_no, row in enumerate(rows, start=1):
        shelf_id = f"S{row_no}"
        xs1, ys1, xs2, ys2 = zip(*(clipped[i] for i in row))
        shelves.append(ShelfRegion(shelf_id=shelf_id, bbox=[min(xs1), min(ys1), max(xs2), max(ys2)]))
        for i in row:
            items.append(
                Item(
                    item_id=f"i{len(items) + 1:03d}",
                    bbox=[round(v, 1) for v in clipped[i]],
                    det_confidence=round(float(confidences[i]), 4),
                    sku=None,
                    sku_confidence=0.0,
                    shelf_id=shelf_id,
                    slot_id=None,
                )
            )

    return DetectionOutput(
        image_id=image_id,
        image_path=image_path,
        image_width=width,
        image_height=height,
        planogram_id=planogram_id,
        captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        produced_by={"vision": produced_by},
        shelves=shelves,
        items=items,
    )


class ProductDetector:
    """Thin wrapper over an Ultralytics YOLO model fine-tuned for product blocks."""

    def __init__(self, weights: str | Path, conf: float = 0.25, imgsz: int = 960, device: str | int | None = None):
        from ultralytics import YOLO  # heavy import, kept out of module load

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.name = Path(weights).parent.parent.name or Path(weights).stem

    def detect(self, image_path: str | Path, rel_to: Path | None = None, **kw) -> DetectionOutput:
        image_path = Path(image_path)
        res = self.model.predict(
            str(image_path), conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        h, w = res.orig_shape
        shown = image_path.relative_to(rel_to) if rel_to else image_path
        return build_detection_output(
            image_id=image_path.stem,
            image_path=str(shown),
            image_size=(w, h),
            boxes=res.boxes.xyxy.cpu().tolist(),
            confidences=res.boxes.conf.cpu().tolist(),
            produced_by=f"yolo:{self.name}",
            **kw,
        )
