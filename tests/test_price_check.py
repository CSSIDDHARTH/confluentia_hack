from pathlib import Path

from shelfsense.contracts import DetectionOutput, Label, load_catalogue, load_detection_output
from shelfsense.pricing.price_check import PriceIssueType, check_prices

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"


def _label(**overrides) -> Label:
    defaults = dict(
        label_id="lX",
        bbox=[0, 0, 10, 10],
        shelf_id="S1",
        slot_id="S1-01",
        sku="SKU-1001",
        ocr_text="$3.49",
        price=3.49,
        ocr_confidence=0.9,
    )
    defaults.update(overrides)
    return Label(**defaults)


def _det_with(*labels: Label) -> DetectionOutput:
    return DetectionOutput(
        image_id="synthetic",
        image_path="data/mock/mock_shelf_001.png",
        image_width=900,
        image_height=900,
        planogram_id="PG-001",
        captured_at="2026-10-20T09:30:00Z",
        labels=list(labels),
    )


def test_example_detection_has_exactly_the_known_mismatch():
    # contracts/detection_output.example.json has one label whose price
    # (S2-03, $5.99) disagrees with the catalogue ($5.49); every other
    # label matches.
    det = load_detection_output(CONTRACTS / "detection_output.example.json")
    catalogue = load_catalogue(CONTRACTS / "catalogue.example.csv")

    issues = check_prices(det, catalogue)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.label_id == "l06"
    assert issue.issue_type == PriceIssueType.PRICE_MISMATCH
    assert issue.label_price == 5.99
    assert issue.catalogue_price == 5.49
    assert round(issue.difference, 2) == 0.50


def test_matching_price_raises_no_issue():
    catalogue = load_catalogue(CONTRACTS / "catalogue.example.csv")
    det = _det_with(_label(sku="SKU-1001", price=3.49))

    assert check_prices(det, catalogue) == []


def test_mismatched_price_flagged_with_signed_difference():
    catalogue = load_catalogue(CONTRACTS / "catalogue.example.csv")
    det = _det_with(_label(sku="SKU-1001", price=3.99))

    issues = check_prices(det, catalogue)

    assert len(issues) == 1
    assert issues[0].issue_type == PriceIssueType.PRICE_MISMATCH
    assert issues[0].label_price == 3.99
    assert issues[0].catalogue_price == 3.49
    assert round(issues[0].difference, 2) == 0.50


def test_mismatch_on_low_confidence_read_is_flagged_to_verify():
    catalogue = load_catalogue(CONTRACTS / "catalogue.example.csv")
    shaky = check_prices(_det_with(_label(price=3.99, ocr_confidence=0.4)), catalogue)
    firm = check_prices(_det_with(_label(price=3.99, ocr_confidence=0.6)), catalogue)

    assert shaky[0].issue_type == PriceIssueType.PRICE_MISMATCH
    assert shaky[0].low_confidence
    assert not firm[0].low_confidence


def test_unreadable_price_flagged():
    catalogue = load_catalogue(CONTRACTS / "catalogue.example.csv")
    det = _det_with(_label(sku="SKU-1001", price=None, ocr_text="$?.??"))

    issues = check_prices(det, catalogue)

    assert len(issues) == 1
    assert issues[0].issue_type == PriceIssueType.UNREADABLE_PRICE
    assert issues[0].label_price is None


def test_label_with_no_sku_flagged_as_unknown_product():
    catalogue = load_catalogue(CONTRACTS / "catalogue.example.csv")
    det = _det_with(_label(sku=None, price=3.49))

    issues = check_prices(det, catalogue)

    assert len(issues) == 1
    assert issues[0].issue_type == PriceIssueType.UNKNOWN_PRODUCT
    assert issues[0].sku is None


def test_label_with_sku_absent_from_catalogue_flagged_as_unknown_product():
    catalogue = load_catalogue(CONTRACTS / "catalogue.example.csv")
    det = _det_with(_label(sku="SKU-9999", price=1.00))

    issues = check_prices(det, catalogue)

    assert len(issues) == 1
    assert issues[0].issue_type == PriceIssueType.UNKNOWN_PRODUCT
    assert issues[0].sku == "SKU-9999"


def test_tiny_floating_point_difference_is_not_a_mismatch():
    catalogue = load_catalogue(CONTRACTS / "catalogue.example.csv")
    det = _det_with(_label(sku="SKU-1001", price=3.49 + 1e-9))

    assert check_prices(det, catalogue) == []


def test_multiple_labels_each_checked_independently():
    catalogue = load_catalogue(CONTRACTS / "catalogue.example.csv")
    det = _det_with(
        _label(label_id="ok", sku="SKU-1001", price=3.49),
        _label(label_id="bad", sku="SKU-1002", price=5.00),
        _label(label_id="unreadable", sku="SKU-1003", price=None),
        _label(label_id="unknown", sku=None, price=1.00),
    )

    issues = check_prices(det, catalogue)

    by_id = {i.label_id: i for i in issues}
    assert set(by_id) == {"bad", "unreadable", "unknown"}
    assert by_id["bad"].issue_type == PriceIssueType.PRICE_MISMATCH
    assert by_id["unreadable"].issue_type == PriceIssueType.UNREADABLE_PRICE
    assert by_id["unknown"].issue_type == PriceIssueType.UNKNOWN_PRODUCT
