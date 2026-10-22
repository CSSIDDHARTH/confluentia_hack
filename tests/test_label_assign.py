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
