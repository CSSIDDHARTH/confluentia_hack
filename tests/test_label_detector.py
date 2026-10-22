"""The trained label detector, on a real Grozi shelf photo. Skipped when the
weights (models/label_det, git-ignored) or the Grozi data are not present."""

import shutil
from pathlib import Path

import pytest
from PIL import Image

from shelfsense.pricing.label_detector import DEFAULT_WEIGHTS, detect_labels

SHELF = Path(__file__).resolve().parents[1] / "data" / "grozi" / "Testing" / "store4" / "images" / "1.jpg"

pytestmark = pytest.mark.skipif(
    not (DEFAULT_WEIGHTS.exists() and SHELF.exists() and shutil.which("tesseract")),
    reason="needs models/label_det weights, Grozi data and tesseract",
)


def test_finds_the_hand_boxed_olive_oil_label():
    labels = detect_labels(Image.open(SHELF))

    # hand-boxed label on this photo: [1420, 1061, 1812, 1224]
    near = [l for l in labels if abs(l.bbox[0] - 1420) < 300 and abs(l.bbox[1] - 1061) < 100]
    assert near, f"no detection near the known label: {[l.bbox for l in labels]}"
