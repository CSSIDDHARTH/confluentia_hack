from pathlib import Path

from shelfsense.contracts import load_catalogue, load_detection_output, load_planogram
from shelfsense.report.compliance import ViolationKind, compute_report
from shelfsense.report.impact import at_stake, currency

ROOT = Path(__file__).resolve().parents[1]


def setup():
    return (
        load_detection_output(ROOT / "data" / "mock" / "mock_shelf_002_lowstock.json"),
        load_planogram(ROOT / "contracts" / "planogram.example.json"),
        load_catalogue(ROOT / "contracts" / "catalogue.example.csv"),
    )


def test_missing_packs_and_price_gaps_are_priced():
    det, pg, cat = setup()
    stake = at_stake(compute_report(det, pg, cat), det, pg, cat)
    assert stake == {
        "v-S1-01-low_stock": 6.98,        # 3 planned, 1 seen, at 3.49
        "v-S2-02-stock_out": 9.87,        # 3 planned, none seen, at 3.29
        "v-l06-price_mismatch": 1.00,     # label 5.99 vs 5.49, 2 facings
    }


def test_misplaced_and_unrecognised_are_not_priced():
    det, pg, cat = setup()
    rep = compute_report(det, pg, cat)
    stake = at_stake(rep, det, pg, cat)
    unpriced = {v.violation_id for v in rep.violations if v.kind in (ViolationKind.MISPLACED, ViolationKind.UNRECOGNISED_ITEM)}
    assert unpriced and not unpriced & stake.keys()


def test_low_confidence_mismatch_is_a_label_to_check_not_a_priced_mismatch():
    det, pg, cat = setup()
    det.labels = [lb.model_copy(update={"ocr_confidence": 0.3}) if lb.label_id == "l06" else lb for lb in det.labels]
    rep = compute_report(det, pg, cat)
    v = next(v for v in rep.violations if v.slot_id == "S2-03")
    assert v.kind == ViolationKind.UNREADABLE_PRICE and "by eye" in v.recommended_action
    assert v.violation_id not in at_stake(rep, det, pg, cat)


def test_currency():
    _, _, cat = setup()
    assert currency(cat) == next(iter(cat.values())).currency
