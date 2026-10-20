from pathlib import Path

import pytest
from pydantic import ValidationError

from shelfsense.contracts import (
    Item,
    load_catalogue,
    load_detection_output,
    load_planogram,
)

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"


@pytest.fixture(scope="module")
def planogram():
    return load_planogram(CONTRACTS / "planogram.example.json")


@pytest.fixture(scope="module")
def catalogue():
    return load_catalogue(CONTRACTS / "catalogue.example.csv")


@pytest.fixture(scope="module")
def det():
    return load_detection_output(CONTRACTS / "detection_output.example.json")


def test_planogram_skus_exist_in_catalogue(planogram, catalogue):
    missing = {s.sku for sh in planogram.shelves for s in sh.slots} - catalogue.keys()
    assert not missing


def test_detection_refers_to_planogram(det, planogram):
    assert det.planogram_id == planogram.planogram_id
    slot_ids = planogram.slots_by_id().keys()
    shelf_ids = {sh.shelf_id for sh in planogram.shelves}
    for obj in [*det.items, *det.gaps, *det.labels]:
        assert obj.slot_id is None or obj.slot_id in slot_ids
        assert obj.shelf_id is None or obj.shelf_id in shelf_ids


def test_detected_skus_exist_in_catalogue(det, catalogue):
    skus = {i.sku for i in det.items if i.sku} | {lb.sku for lb in det.labels if lb.sku}
    assert skus <= catalogue.keys()


def test_bboxes_inside_image(det):
    for obj in [*det.shelves, *det.items, *det.gaps, *det.labels]:
        x1, y1, x2, y2 = obj.bbox
        assert 0 <= x1 < x2 <= det.image_width
        assert 0 <= y1 < y2 <= det.image_height


def test_ids_unique(det):
    for ids in ([i.item_id for i in det.items], [g.gap_id for g in det.gaps], [lb.label_id for lb in det.labels]):
        assert len(ids) == len(set(ids))


def test_bad_bbox_rejected():
    with pytest.raises(ValidationError):
        Item(item_id="x", bbox=[10, 10, 5, 20], det_confidence=0.9, sku=None,
             sku_confidence=0.1, shelf_id=None, slot_id=None)
