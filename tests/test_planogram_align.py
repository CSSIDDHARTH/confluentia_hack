import pytest

from shelfsense.contracts import Item, Planogram, Shelf, Slot
from shelfsense.vision.detect import build_detection_output
from shelfsense.vision.planogram_align import align_row, assign_slots


def slots(*skus):
    return [Slot(slot_id=f"S1-{i + 1:02d}", position=i, sku=s, facings=1, min_facings=1) for i, s in enumerate(skus)]


def items(*skus):
    return [
        Item(item_id=f"i{i + 1}", bbox=[10 + 100 * i, 10, 90 + 100 * i, 200], det_confidence=0.9,
             sku=s, sku_confidence=0.9 if s else 0.1, shelf_id="S1", slot_id=None)
        for i, s in enumerate(skus)
    ]


def run(detected, expected):
    its, sls = items(*detected), slots(*expected)
    mapping = align_row(its, sls)
    return {i.item_id: mapping[i.item_id] for i in its}


def test_exact_row_maps_in_order():
    assert run("ABC", "ABC") == {"i1": "S1-01", "i2": "S1-02", "i3": "S1-03"}


def test_missing_product_leaves_its_slot_empty():
    m = run("AC", "ABC")  # B is gone
    assert m == {"i1": "S1-01", "i2": "S1-03"}
    assert "S1-02" not in m.values()


def test_swapped_pair_is_two_substitutions():
    assert run("BA", "AB") == {"i1": "S1-01", "i2": "S1-02"}  # each item sits in the other's slot


def test_wrong_product_in_slot_stays_in_that_slot():
    assert run("AXC", "ABC") == {"i1": "S1-01", "i2": "S1-02", "i3": "S1-03"}


def test_extra_item_attaches_to_nearest_slot():
    m = run("AXB", "AB")
    assert m["i1"] == "S1-01" and m["i3"] == "S1-02"
    assert m["i2"] == "S1-01"  # X squeezed in after A, so it is reported against A's slot


def test_unrecognised_item_prefers_the_open_slot():
    # the None item should take B's slot rather than push A and C out of alignment
    assert run("A?C", "ABC") == {"i1": "S1-01", "i2": "S1-02", "i3": "S1-03"}


def test_empty_cases():
    assert align_row([], slots("A", "B")) == {}
    assert align_row(items("A"), []) == {}


def planogram():
    return Planogram(
        planogram_id="PG-T", fixture_name="t", effective_from="2026-01-01",
        shelves=[
            Shelf(shelf_id="S1", index=0, slots=slots("A", "B")),
            Shelf(shelf_id="S2", index=1, slots=[
                Slot(slot_id="S2-01", position=0, sku="C", facings=1, min_facings=1),
                Slot(slot_id="S2-02", position=1, sku="D", facings=1, min_facings=1),
            ]),
        ],
    )


def test_assign_slots_end_to_end_two_rows_and_extra_row():
    det = build_detection_output(
        "img", "x.jpg", (1000, 1000),
        boxes=[[0, 0, 100, 100], [200, 0, 300, 100],        # row 1: A B
               [0, 400, 100, 500], [200, 400, 300, 500],     # row 2: C D (D unrecognised)
               [0, 800, 100, 900]],                          # row 3: not in the plan
        confidences=[0.9] * 5,
    )
    for it, sku in zip(det.items, ["A", "B", "C", None, "Z"]):
        it.sku = sku
    out = assign_slots(det, planogram())
    by_id = {i.item_id: i for i in out.items}
    assert out.planogram_id == "PG-T"
    assert [by_id[k].slot_id for k in ("i001", "i002", "i003", "i004")] == ["S1-01", "S1-02", "S2-01", "S2-02"]
    assert [by_id[k].shelf_id for k in ("i001", "i003")] == ["S1", "S2"]
    assert by_id["i005"].slot_id is None  # row beyond the planogram
    assert out.produced_by["alignment"] == "nw-v1"
    assert type(out).model_validate_json(out.model_dump_json()) == out


@pytest.mark.parametrize("detected,expected", [("ABCD", "ABCD"), ("BADC", "ABCD"), ("AD", "ABCD")])
def test_every_item_gets_exactly_one_slot(detected, expected):
    m = run(detected, expected)
    assert len(m) == len(detected)
    assert all(v.startswith("S1-") for v in m.values())
