from shelfsense.contracts import DetectionOutput
from shelfsense.vision.bay import add_gaps, assign_packs_to_blocks, drop_fragments, facings_output
from shelfsense.vision.detect import build_detection_output


def test_drop_fragments_removes_slivers_but_keeps_normal_packs():
    packs = [[0, 0, 100, 200], [110, 0, 210, 200], [220, 0, 320, 200], [330, 0, 350, 30]]  # last: a bottle neck
    kept, confs = drop_fragments(packs, [0.9, 0.8, 0.7, 0.6])
    assert kept == packs[:3] and confs == [0.9, 0.8, 0.7]
    assert drop_fragments([], []) == ([], [])


def test_pack_goes_to_the_smallest_block_containing_its_centre():
    blocks = [[0, 0, 500, 500], [100, 100, 300, 300]]  # one nested in the other
    packs = [[150, 150, 250, 250], [350, 350, 450, 450], [600, 600, 700, 700]]
    assert assign_packs_to_blocks(blocks, packs) == [1, 0, None]


def aligned_two_blocks():
    det = build_detection_output(
        "img", "x.jpg", (1000, 400), boxes=[[0, 0, 400, 300], [500, 0, 900, 300]], confidences=[0.9, 0.9]
    )
    for item, sku, slot in zip(det.items, ["A", "B"], ["S1-01", "S1-02"]):
        item.sku, item.slot_id, item.sku_confidence = sku, slot, 0.8
    return det


def test_each_pack_becomes_an_item_inheriting_its_blocks_sku_and_slot():
    packs = [[10, 10, 110, 290], [120, 10, 220, 290], [230, 10, 330, 290], [510, 10, 610, 290]]
    out = facings_output(aligned_two_blocks(), packs, [0.9] * 4)
    assert len(out.items) == 4
    assert [(i.sku, i.slot_id) for i in out.items] == [("A", "S1-01")] * 3 + [("B", "S1-02")]
    assert [i.item_id for i in out.items] == ["i001", "i002", "i003", "i004"]
    assert out.produced_by["facings"] == "packs"
    assert DetectionOutput.model_validate_json(out.model_dump_json()) == out


def test_block_with_no_detected_pack_stays_as_one_item_not_a_stock_out():
    out = facings_output(aligned_two_blocks(), [[10, 10, 110, 290], [120, 10, 220, 290]], [0.9, 0.9])
    by_slot = {}
    for i in out.items:
        by_slot.setdefault(i.slot_id, []).append(i)
    assert len(by_slot["S1-01"]) == 2 and len(by_slot["S1-02"]) == 1
    assert by_slot["S1-02"][0].bbox == [500.0, 0.0, 900.0, 300.0]  # the block itself


def test_gaps_between_packs_are_added_and_tied_to_a_shelf():
    det = facings_output(aligned_two_blocks(), [[0, 0, 100, 300], [100, 0, 200, 300], [400, 0, 500, 300], [500, 0, 600, 300]], [0.9] * 4)
    det = add_gaps(det, [[0, 0, 100, 300], [100, 0, 200, 300], [400, 0, 500, 300], [500, 0, 600, 300]])
    assert len(det.gaps) == 1
    g = det.gaps[0]
    assert g.bbox[0] == 200 and g.bbox[2] == 400 and g.estimated_missing_facings == 2
    assert g.shelf_id == "S1" and g.slot_id is None
