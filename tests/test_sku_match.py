import numpy as np
import pytest
from PIL import Image

from shelfsense.contracts import DetectionOutput
from shelfsense.vision.detect import build_detection_output
from shelfsense.vision.pipeline import identify_items
from shelfsense.vision.sku_match import ReferenceIndex, SkuMatcher, pack_crop

COLOURS = {"A": (250, 10, 10), "B": (10, 250, 10), "C": (10, 10, 250), "D": (240, 240, 20)}


class ColourEmbedder:
    """Stand-in for DINOv2: embeds an image as its normalised mean colour."""

    def embed(self, images):
        v = np.array([np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) for im in images])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def swatch(colour, size=(40, 60)):
    return Image.new("RGB", size, colour)


@pytest.fixture
def index():
    emb = ColourEmbedder()
    skus = list(COLOURS) + ["A"]  # product A has two reference photos
    imgs = [swatch(c) for c in COLOURS.values()] + [swatch((230, 40, 40))]
    return ReferenceIndex(emb.embed(imgs), skus, [f"{s}.jpg" for s in skus])


@pytest.fixture
def matcher(index):
    return SkuMatcher(index, embedder=ColourEmbedder())


def test_index_groups_references_by_product(index):
    assert list(index.unique_skus) == ["A", "B", "C", "D"]
    top = index.query(ColourEmbedder().embed([swatch((255, 0, 0))]), k=2)[0]
    assert top[0][0] == "A" and len(top) == 2 and top[0][1] >= top[1][1]


def test_match_finds_nearest_product_with_confidence(matcher):
    m = matcher.match([swatch((245, 15, 15)), swatch((15, 15, 245))])
    assert [x.sku for x in m] == ["A", "C"]
    assert all(0 < x.confidence <= 1 for x in m)
    assert m[0].alternatives  # runner-ups are returned


def test_candidates_restrict_the_answer(matcher):
    [m] = matcher.match([swatch((245, 15, 15))], candidates={"B", "C"})
    assert m.sku in {"B", "C"}


def test_match_empty_input(matcher):
    assert matcher.match([]) == []


def test_identity_adapter_changes_nothing(index):
    plain = SkuMatcher(index, embedder=ColourEmbedder())
    adapted = SkuMatcher(index, embedder=ColourEmbedder(), adapter=np.eye(3, dtype=np.float32))
    img = [swatch((12, 200, 30))]
    assert plain.match(img)[0].sku == adapted.match(img)[0].sku
    assert plain.match(img)[0].confidence == pytest.approx(adapted.match(img)[0].confidence, rel=1e-4)


def test_pack_crop_keeps_about_one_pack_from_a_wide_block():
    im = Image.new("RGB", (1000, 800))
    crop = pack_crop(im, [100, 100, 900, 500])  # 800 wide x 400 tall block
    assert crop.size == (300, 400)  # 0.75 * height wide, centred
    narrow = pack_crop(im, [100, 100, 250, 500])  # already narrower than one pack
    assert narrow.size == (150, 400)


def test_identify_items_fills_skus_and_respects_threshold(matcher):
    image = Image.new("RGB", (200, 100), (0, 0, 0))
    image.paste(swatch(COLOURS["A"], (80, 90)), (5, 5))
    image.paste(swatch((10, 128, 128), (80, 90)), (110, 5))  # exactly between B and C
    det = build_detection_output(
        "x", "x.jpg", (200, 100), boxes=[[5, 5, 85, 95], [110, 5, 190, 95]], confidences=[0.9, 0.9]
    )
    out = identify_items(det, image, matcher, candidates={"A", "B", "C"}, min_confidence=0.6)
    assert isinstance(out, DetectionOutput)
    first, second = sorted(out.items, key=lambda i: i.bbox[0])
    assert first.sku == "A" and first.sku_confidence >= 0.6
    assert second.sku is None and 0 < second.sku_confidence < 0.6  # left for human review
    assert out.produced_by["matcher"] == "dinov2-vits14"
    assert DetectionOutput.model_validate_json(out.model_dump_json()) == out
