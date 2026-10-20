import pytest

from shelfsense.vision.gaps import cluster_rows, find_gaps, low_stock_rows


def row_of(xs, y1=0, y2=100, w=50):
    return [[x, y1, x + w, y2] for x in xs]


def test_cluster_rows_top_to_bottom_left_to_right():
    boxes = row_of([100, 0, 50], 0, 100) + row_of([0, 50], 300, 400)
    rows = cluster_rows(boxes)
    assert [len(r.boxes) for r in rows] == [3, 2]
    assert [b[0] for b in rows[0].boxes] == [0, 50, 100]
    assert rows[0].index == 0 and rows[1].index == 1


def test_cluster_rows_empty():
    assert cluster_rows([]) == []
    assert find_gaps([]) == ([], [])


def test_full_row_has_no_gaps_and_full_fill():
    rows, gaps = find_gaps(row_of([0, 50, 100, 150]))
    assert gaps == []
    assert rows[0].fill_ratio == pytest.approx(1.0)


def test_missing_pack_in_the_middle_is_a_gap():
    boxes = row_of([0, 50, 150, 200])  # the pack at x=100 is missing
    _, gaps = find_gaps(boxes)
    assert len(gaps) == 1
    g = gaps[0]
    assert g.bbox == [100, 0, 150, 100] and g.estimated_missing == 1


def test_wide_gap_estimates_several_missing_packs():
    _, gaps = find_gaps(row_of([0, 50, 250, 300]))
    assert len(gaps) == 1 and gaps[0].estimated_missing == 3


def test_short_row_gets_a_gap_at_its_end_using_other_rows_for_the_bay_edge():
    boxes = row_of([0, 50, 100, 150], 0, 100) + row_of([0, 50], 300, 400)
    rows, gaps = find_gaps(boxes)
    assert [g.row for g in gaps] == [1]
    assert gaps[0].bbox[0] == 100 and gaps[0].bbox[2] == 200
    assert rows[1].fill_ratio == pytest.approx(0.5)
    assert [r.index for r in low_stock_rows(rows)] == [1]


def test_normal_tiny_spacing_is_not_a_gap():
    _, gaps = find_gaps([[0, 0, 48, 100], [52, 0, 100, 100], [104, 0, 152, 100]])
    assert gaps == []


def test_overlapping_boxes_do_not_create_negative_gaps():
    _, gaps = find_gaps([[0, 0, 60, 100], [40, 0, 100, 100]])
    assert gaps == []


def test_confidence_grows_with_gap_width():
    _, small = find_gaps(row_of([0, 100, 150]))  # 50 wide gap (1 pack)
    _, large = find_gaps(row_of([0, 250, 300]))  # 200 wide gap
    assert large[0].confidence > small[0].confidence
    assert 0.5 <= small[0].confidence <= 0.95
