from shelfsense.contracts import DetectionOutput
from shelfsense.vision.availability import analyze_packs


def packs(xs, y1, y2, w=50):
    return [[x, y1, x + w, y2] for x in xs]


def test_full_bay_has_no_actions():
    boxes = packs([0, 50, 100, 150], 0, 100) + packs([0, 50, 100, 150], 200, 300)
    det, rep = analyze_packs("img", "x.jpg", (400, 400), boxes, [0.9] * 8)
    assert det.gaps == [] and rep.actions == []
    assert rep.empty_spaces == 0 and rep.overall_fill == 1.0
    assert [r.packs for r in rep.rows] == [4, 4]


def test_gap_and_short_row_produce_ordered_actions():
    boxes = (
        packs([0, 50, 150, 200], 0, 100)  # top row: one pack missing in the middle
        + packs([0, 50], 200, 300)  # bottom row: only half stocked
    )
    det, rep = analyze_packs("img", "x.jpg", (400, 400), boxes, [0.9] * len(boxes), min_packs=0)
    assert {g.shelf_id for g in det.gaps} == {"S1", "S2"}
    assert rep.empty_spaces == 2 and rep.missing_facings == 1 + 3
    assert [r.low_stock for r in rep.rows] == [False, True]
    # the bigger gap is listed before the smaller one, and the low-stock row is reported
    assert "S2" in rep.actions[0] and "3 packs" in rep.actions[0]
    assert any("only 40% stocked" in a for a in rep.actions)


def test_output_is_a_valid_contract_with_pack_items():
    boxes = packs([0, 50, 150], 0, 100)
    det, rep = analyze_packs("img", "x.jpg", (300, 200), boxes, [0.9, 0.8, 0.7])
    assert len(det.items) == 3 and all(i.sku is None and i.shelf_id == "S1" for i in det.items)
    assert DetectionOutput.model_validate_json(det.model_dump_json()) == det
    assert det.produced_by["availability"] == "gaps-v1"
    assert [i.det_confidence for i in det.items] == [0.9, 0.8, 0.7]


def test_no_detections():
    det, rep = analyze_packs("img", "x.jpg", (300, 200), [], [])
    assert det.items == [] and rep.rows == [] and rep.overall_fill == 0.0


def test_too_few_packs_gives_a_warning_and_no_refill_list():
    boxes = [[100.0 * k, 100.0, 100.0 * k + 80, 200.0] for k in range(5)]
    det, report = analyze_packs("close", "close.jpg", (800, 400), boxes, [0.9] * 5)
    assert report.warnings and report.actions == []


def test_enough_packs_gives_no_warning():
    boxes = [[100.0 * k, 100.0, 100.0 * k + 80, 200.0] for k in range(10)]
    det, report = analyze_packs("shelf", "shelf.jpg", (1200, 400), boxes, [0.9] * 10)
    assert report.warnings == []
