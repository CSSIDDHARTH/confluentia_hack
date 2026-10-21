from shelfsense.contracts import CatalogueItem, DetectionOutput, Item, Planogram, Shelf, Slot
from shelfsense.report.compliance import ViolationKind, compute_report


def plan(facings=3, min_facings=2):
    return Planogram(
        planogram_id="P", fixture_name="f", effective_from="2026-01-01",
        shelves=[Shelf(shelf_id="S1", index=0, slots=[
            Slot(slot_id="S1-01", position=0, sku="A", facings=facings, min_facings=min_facings),
            Slot(slot_id="S1-02", position=1, sku="B", facings=facings, min_facings=min_facings),
        ])],
    )


def catalogue():
    return {s: CatalogueItem(sku=s, name=f"Product {s}", brand="b", category="c", size="", price=1.0, currency="CHF") for s in "ABC"}


def packs(slot_id, sku, n, start=0):
    return [
        Item(item_id=f"{slot_id}-{start + k}", bbox=[10 * (start + k), 0, 10 * (start + k) + 8, 50], det_confidence=0.9,
             sku=sku, sku_confidence=0.9 if sku else 0.1, shelf_id="S1", slot_id=slot_id)
        for k in range(n)
    ]


def det(items):
    return DetectionOutput(image_id="i", image_path="x.jpg", image_width=500, image_height=100, planogram_id="P",
                           captured_at="2026-01-01T00:00:00Z", items=items)


def kinds(report):
    return sorted(v.kind.value for v in report.violations)


def test_default_still_reports_one_violation_per_item():
    d = det(packs("S1-01", "C", 3) + packs("S1-02", "B", 3))  # three packs of C where A belongs
    assert kinds(compute_report(d, plan(), catalogue())).count("misplaced") == 3


def test_grouped_reports_one_misplaced_violation_per_slot_and_product():
    d = det(packs("S1-01", "C", 3) + packs("S1-02", "B", 3))
    r = compute_report(d, plan(), catalogue(), group_items=True)
    assert kinds(r) == ["misplaced", "stock_out"]  # A is absent from its slot, C is the intruder
    mis = next(v for v in r.violations if v.kind == ViolationKind.MISPLACED)
    assert "3 items of C" in mis.detail and "(3 packs)" in mis.recommended_action
    assert mis.slot_id == "S1-01" and mis.sku == "C"


def test_grouped_two_different_intruders_give_two_violations():
    d = det(packs("S1-01", "A", 3) + packs("S1-02", "B", 2) + packs("S1-02", "C", 2, start=5))
    r = compute_report(d, plan(), catalogue(), group_items=True)
    assert [v.kind for v in r.violations] == [ViolationKind.MISPLACED]
    assert r.violations[0].sku == "C"


def test_grouped_unrecognised_is_one_violation_per_slot():
    d = det(packs("S1-01", "A", 3) + packs("S1-02", None, 4))
    r = compute_report(d, plan(), catalogue(), group_items=True)
    assert sorted(kinds(r)) == ["stock_out", "unrecognised_item"]
    assert "4 items" in next(v for v in r.violations if v.kind == ViolationKind.UNRECOGNISED_ITEM).detail


def test_pack_counts_drive_low_stock_and_stock_out():
    d = det(packs("S1-01", "A", 1) + packs("S1-02", "B", 3))  # plan wants 3 (min 2); slot 1 has only 1 pack
    assert kinds(compute_report(d, plan(), catalogue(), group_items=True)) == ["low_stock"]
    d = det(packs("S1-02", "B", 3))  # nothing in slot 1
    assert kinds(compute_report(d, plan(), catalogue(), group_items=True)) == ["stock_out"]


def test_a_compliant_bay_scores_100():
    d = det(packs("S1-01", "A", 3) + packs("S1-02", "B", 3))
    r = compute_report(d, plan(), catalogue(), group_items=True)
    assert r.score == 100 and r.violations == []
