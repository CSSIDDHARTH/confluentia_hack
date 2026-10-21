"""Phases 1+2 together: detect product blocks, then name the product in each one."""

from pathlib import Path

from PIL import Image

from shelfsense.contracts import DetectionOutput
from shelfsense.vision.detect import ProductDetector
from shelfsense.vision.sku_match import SkuMatcher, pack_crop

# Below this softmax confidence an item is left unrecognised (sku=None) for human review.
# On held-out Grozi stores, >= 0.5 keeps ~81% of boxes at ~85% accuracy (>= 0.7: 57% at 92%).
DEFAULT_MIN_CONFIDENCE = 0.5


def identify_items(
    det: DetectionOutput,
    image: Image.Image,
    matcher: SkuMatcher,
    candidates: set[str] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> DetectionOutput:
    """Fill sku / sku_confidence on every item. `candidates` restricts matching to the
    planogram's SKUs; None searches the whole reference set."""
    if not det.items:
        return det
    crops = [pack_crop(image, item.bbox) for item in det.items]
    for item, m in zip(det.items, matcher.match(crops, candidates=candidates)):
        item.sku_confidence = round(m.confidence, 4)
        item.sku = m.sku if m.confidence >= min_confidence else None
    det.produced_by["matcher"] = "dinov2-vits14+adapter" if matcher.adapter is not None else "dinov2-vits14"
    return det


def analyze_image(
    image_path: str | Path,
    detector: ProductDetector,
    matcher: SkuMatcher,
    candidates: set[str] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    rel_to: Path | None = None,
    **detect_kw,
) -> DetectionOutput:
    image_path = Path(image_path)
    det = detector.detect(image_path, rel_to=rel_to, **detect_kw)
    image = Image.open(image_path).convert("RGB")
    return identify_items(det, image, matcher, candidates, min_confidence)
