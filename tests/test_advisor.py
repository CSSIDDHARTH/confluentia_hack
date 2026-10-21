from pathlib import Path

import pytest

from shelfsense.advisor.plan import plan
from shelfsense.contracts import load_catalogue, load_detection_output, load_planogram
from shelfsense.report.compliance import compute_report

BAY = Path(__file__).resolve().parents[1] / "data" / "demo_bays" / "store1_47"
pytestmark = pytest.mark.skipif(not BAY.exists(), reason="no demo bays")


def report(kind):
    cat = load_catalogue(BAY / "catalogue.csv")
    r = compute_report(
        load_detection_output(BAY / f"detection_{kind}.json"), load_planogram(BAY / f"planogram_{kind}.json"), cat, group_items=True
    )
    return r, cat


def test_every_violation_lands_in_exactly_one_step():
    r, cat = report("scenario")
    ids = [i for s in plan(r, cat) for i in s.violation_ids]
    assert sorted(ids) == sorted(v.violation_id for v in r.violations)


def test_planted_swap_becomes_one_step_naming_both_slots():
    r, cat = report("scenario")
    swaps = [s for s in plan(r, cat) if s.text.startswith("Swap")]
    assert len(swaps) == 1 and "S1-01" in swaps[0].text and "S1-02" in swaps[0].text
    assert len(swaps[0].violation_ids) == 4 and len(plan(r, cat)) < len(r.violations)


def test_a_wrong_product_in_a_slot_missing_the_right_one_is_one_step():
    r, cat = report("scenario")
    step = next(s for s in plan(r, cat) if "S3-01" in s.text and "Move" in s.text)
    assert "Then restock" in step.text


def test_steps_are_ordered_by_severity():
    r, cat = report("scenario")
    sev = [s.severity for s in plan(r, cat)]
    assert sev == sorted(sev, reverse=True)


def test_a_clean_shelf_has_no_steps():
    r, cat = report("compliant")
    assert len(plan(r.model_copy(update={"violations": []}), cat)) == 0


def test_what_to_do_page_shows_fewer_steps_than_violation_lines():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    page = str(Path(__file__).resolve().parents[1] / "app" / "pages" / "2_Shift_briefing.py")
    at = AppTest.from_file(page, default_timeout=120).run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["Compliance score"] == "31 / 100" and int(m["Steps for staff"]) < int(m["Violation lines"])
