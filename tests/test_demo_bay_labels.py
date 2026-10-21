"""Demo bays carry price labels tied to slots, and the planted price change shows up."""

from pathlib import Path

import pytest

from shelfsense.contracts import load_catalogue, load_detection_output
from shelfsense.pricing.price_check import PriceIssueType, check_prices

BAYS = Path(__file__).resolve().parents[1] / "data" / "demo_bays"


@pytest.mark.parametrize("bay", sorted(p.name for p in BAYS.iterdir()))
def test_labels_on_slots(bay):
    det = load_detection_output(BAYS / bay / "detection_compliant.json")
    assert det.labels
    assert any(lb.slot_id for lb in det.labels)


def test_planted_price_mismatch():
    bay = BAYS / "store1_36"
    det = load_detection_output(bay / "detection_scenario.json")

    def mismatches(catalogue_file):
        issues = check_prices(det, load_catalogue(bay / catalogue_file))
        return {i.label_id for i in issues if i.issue_type == PriceIssueType.PRICE_MISMATCH}

    assert not mismatches("catalogue.csv")
    assert mismatches("catalogue_scenario.csv")
