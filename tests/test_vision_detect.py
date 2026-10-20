from shelfsense.vision.detect import build_detection_output, group_into_rows


def test_group_into_rows_orders_top_to_bottom_left_to_right():
    boxes = [
        [300, 500, 400, 700],  # row 2, right
        [10, 20, 100, 200],  # row 1, left
        [110, 30, 200, 210],  # row 1, right
        [10, 510, 100, 690],  # row 2, left
    ]
    assert group_into_rows(boxes) == [[1, 2], [3, 0]]


def test_group_into_rows_empty():
    assert group_into_rows([]) == []


def test_build_detection_output_is_valid_and_consistent():
    out = build_detection_output(
        "img1", "data/x.jpg", (1000, 800),
        boxes=[[10, 20, 100, 200], [-5, 20, 120, 210], [10, 400, 100, 5000]],
        confidences=[0.9, 0.8, 0.7],
    )
    assert [s.shelf_id for s in out.shelves] == ["S1", "S2"]
    assert len(out.items) == 3
    assert all(i.sku is None and i.slot_id is None for i in out.items)
    assert {i.shelf_id for i in out.items} == {"S1", "S2"}
    # boxes are clipped to the image
    assert all(0 <= i.bbox[0] and i.bbox[3] <= 800 for i in out.items)
    assert len({i.item_id for i in out.items}) == 3
    # round-trips through the contract
    assert type(out).model_validate_json(out.model_dump_json()) == out


def test_build_detection_output_no_detections():
    out = build_detection_output("img2", "data/y.jpg", (640, 480), [], [])
    assert out.items == [] and out.shelves == []
