from pathlib import Path

import pytest

from shelfsense.advisor.edit import apply_rows, changes, plan_rows, swap_products
from shelfsense.advisor.plan import plan
from shelfsense.contracts import load_catalogue, load_detection_output, load_planogram
from shelfsense.report.compliance import compute_report

BAY = Path(__file__).resolve().parents[1] / "data" / "demo_bays" / "store1_47"
pytestmark = pytest.mark.skipif(not BAY.exists(), reason="no demo bays")


def setup():
    return (
        load_detection_output(BAY / "detection_compliant.json"), load_planogram(BAY / "planogram_compliant.json"),
        load_catalogue(BAY / "catalogue.csv"),
    )


def score(det, pg, cat):
    return compute_report(det, pg, cat, group_items=True)


def test_unedited_rows_rebuild_the_same_plan_and_score():
    det, pg, cat = setup()
    assert apply_rows(pg, plan_rows(pg)) == pg
    assert score(det, apply_rows(pg, plan_rows(pg)), cat).score == score(det, pg, cat).score == 98


def test_swapping_two_slots_creates_a_swap_the_planner_merges():
    det, pg, cat = setup()
    rows = swap_products(plan_rows(pg), "S1-01", "S1-02")
    edited = apply_rows(pg, rows)
    rep = score(det, edited, cat)
    assert rep.score < 98
    steps = plan(rep, cat, edited)
    assert any(s.text.startswith("Swap") and "S1-01" in s.text and "S1-02" in s.text for s in steps)


def test_raising_facings_creates_low_stock():
    det, pg, cat = setup()
    rows = plan_rows(pg)
    row = next(r for r in rows if r["slot"] == "S2-01")
    row["facings"], row["minimum"] = row["facings"] + 2, row["facings"] + 1
    rep = score(det, apply_rows(pg, rows), cat)
    assert any(v.kind.value == "low_stock" and v.slot_id == "S2-01" for v in rep.violations)


def test_edits_cannot_make_an_invalid_plan():
    _, pg, _ = setup()
    rows = plan_rows(pg)
    rows[0]["facings"], rows[0]["minimum"] = 0, 9
    slot = apply_rows(pg, rows).shelves[0].slots[0]
    assert slot.facings == 1 and 1 <= slot.min_facings <= slot.facings


def test_changes_lists_only_what_was_edited():
    _, pg, _ = setup()
    rows = plan_rows(pg)
    assert changes(plan_rows(pg), rows) == []
    out = changes(plan_rows(pg), swap_products(rows, "S1-01", "S1-02"))
    assert len(out) == 2 and all("now expects" in line for line in out)


def test_swap_does_not_mutate_its_input():
    _, pg, _ = setup()
    rows = plan_rows(pg)
    swap_products(rows, "S1-01", "S1-02")
    assert rows == plan_rows(pg)
