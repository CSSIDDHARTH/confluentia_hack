from pathlib import Path

import pytest

from shelfsense.contracts import load_detection_output, load_planogram
from shelfsense.pricing.label_assign import assign_labels_to_slots

ROOT = Path(__file__).resolve().parents[1]
PLANOGRAM = load_planogram(ROOT / "contracts" / "planogram.example.json")


@pytest.mark.parametrize(
    "path",
    ["contracts/detection_output.example.json", "data/mock/mock_shelf_002_lowstock.json"],
)
def test_recovers_the_slots_of_the_mock_labels(path):
    det = load_detection_output(ROOT / path)
    unassigned = [lb.model_copy(update={"shelf_id": None, "slot_id": None, "sku": None}) for lb in det.labels]

    assigned = assign_labels_to_slots(unassigned, det, PLANOGRAM)

    assert [(lb.shelf_id, lb.slot_id) for lb in assigned] == [(lb.shelf_id, lb.slot_id) for lb in det.labels]


def test_a_slot_takes_only_one_label():
    det = load_detection_output(ROOT / "contracts/detection_output.example.json")
    first = det.labels[0].model_copy(update={"label_id": "a", "slot_id": None, "sku": None})
    twin = first.model_copy(update={"label_id": "b"})

    a, b = assign_labels_to_slots([first, twin], det, PLANOGRAM)

    assert a.slot_id == "S1-01"
    assert b.slot_id is None and b.shelf_id == "S1"


def test_a_label_under_a_slots_products_but_far_from_its_left_edge_goes_to_that_slot():
    # S1-01's packs span x 40-360; a label at x 200 is 160 px from that edge (more than half the
    # tightest slot spacing on S1), as when a slot starts outside the photo.
    det = load_detection_output(ROOT / "contracts/detection_output.example.json")
    mid = det.labels[0].model_copy(update={"label_id": "mid", "bbox": [200, 278, 290, 302], "slot_id": None, "sku": None})

    (out,) = assign_labels_to_slots([mid], det, PLANOGRAM)

    assert out.slot_id == "S1-01" and out.sku == det.labels[0].sku


def test_a_label_under_a_taken_slot_stays_unassigned():
    det = load_detection_output(ROOT / "contracts/detection_output.example.json")
    edge = det.labels[0].model_copy(update={"label_id": "edge", "slot_id": None, "sku": None})
    mid = edge.model_copy(update={"label_id": "mid", "bbox": [200, 278, 290, 302]})

    a, b = assign_labels_to_slots([edge, mid], det, PLANOGRAM)

    assert a.slot_id == "S1-01" and b.slot_id is None
