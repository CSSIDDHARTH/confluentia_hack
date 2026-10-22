"""Find shelf-edge price labels in a full shelf image with a trained YOLO model.

Weights come from `yolo detect train` on data/price_tags/yolo (see
scripts/prep_price_tag_yolo.py) and live at models/label_det/weights/best.pt.
Needs the `vision` extra (ultralytics).

    labels = detect_labels(Image.open("shelf.jpg"))
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from shelfsense.contracts import Label
from shelfsense.pricing.label_ocr import label_from_crop

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "models" / "label_det" / "weights" / "best.pt"

_models: dict[Path, object] = {}


def _load(weights: Path):
    if weights not in _models:
        from ultralytics import YOLO  # heavy import, only when detection is used

        _models[weights] = YOLO(str(weights))
    return _models[weights]


def detect_label_boxes(
    image: Image.Image, weights: Path = DEFAULT_WEIGHTS, conf: float = 0.25, imgsz: int = 416
) -> list[tuple[list[float], float]]:
    """Label boxes as ([x1, y1, x2, y2] absolute pixels, score), best first."""
    result = _load(weights).predict(image.convert("RGB"), conf=conf, imgsz=imgsz, device="cpu", verbose=False)[0]
    boxes = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in result.boxes]
    return sorted(boxes, key=lambda t: -t[1])


# A whole shelf label is 1.8 to 3.6 times wider than tall (31 hand-boxed Grozi labels). A box that touches
# the photo's edge with a shape outside LABEL_SHAPE is a label the photo cuts off (its price is out of the
# frame) or not a label at all, so no one could read it: it is left out rather than reported as unreadable.
LABEL_SHAPE = (1.5, 4.5)
EDGE_FRACTION = 0.005  # of the photo's shorter side: 12 px on a 3264x2448 Grozi photo


def cut_by_frame(box: list[float], width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    at_edge = min(x1, y1, width - x2, height - y2) <= EDGE_FRACTION * min(width, height)
    shape = (x2 - x1) / max(y2 - y1, 1)
    return at_edge and not LABEL_SHAPE[0] <= shape <= LABEL_SHAPE[1]


def detect_labels(image: Image.Image, weights: Path = DEFAULT_WEIGHTS, conf: float = 0.25) -> list[Label]:
    """Detect every whole label and OCR it (labels the photo cuts off are skipped, see
    cut_by_frame). shelf_id/slot_id/sku are left None; the caller assigns them from the vision output."""
    image = image.convert("RGB")
    labels = []
    boxes = [box for box, _score in detect_label_boxes(image, weights, conf) if not cut_by_frame(box, *image.size)]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (round(v) for v in box)
        labels.append(
            label_from_crop(
                f"label_{i:03d}", [x1, y1, x2, y2],
                shelf_id=None, slot_id=None, sku=None,
                image=image.crop((x1, y1, x2, y2)),
            )
        )
    return labels
