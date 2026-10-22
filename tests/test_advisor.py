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


def test_a_misplaced_product_says_where_it_belongs():
    r, cat = report("scenario")
    pg = load_planogram(BAY / "planogram_scenario.json")
    step = next(s for s in plan(r, cat, pg) if "S3-01" in s.text and "Move" in s.text)
    assert "Cereals #138 belongs in S1-01." in step.text and step.text.index("belongs in") < step.text.index("Then restock")
    # without the planogram the text is unchanged
    assert "belongs in" not in next(s for s in plan(r, cat) if "S3-01" in s.text and "Move" in s.text).text


def test_a_product_the_plan_has_no_other_slot_for_is_said_so():
    r, cat = report("scenario")
    pg = load_planogram(BAY / "planogram_scenario.json")
    for shelf in pg.shelves:
        for slot in shelf.slots:
            if slot.sku == "G13-138":
                slot.sku = "G99-0"
    step = next(s for s in plan(r, cat, pg) if "S3-01" in s.text and "Move" in s.text)
    assert "no other slot for Cereals #138" in step.text


def test_swaps_are_unaffected_by_the_planogram():
    r, cat = report("scenario")
    pg = load_planogram(BAY / "planogram_scenario.json")
    assert [s.text for s in plan(r, cat) if s.text.startswith("Swap")] == [s.text for s in plan(r, cat, pg) if s.text.startswith("Swap")]


def test_steps_are_ordered_by_severity():
    r, cat = report("scenario")
    sev = [s.severity for s in plan(r, cat)]
    assert sev == sorted(sev, reverse=True)


def test_a_clean_shelf_has_no_steps():
    r, cat = report("compliant")
    assert len(plan(r.model_copy(update={"violations": []}), cat)) == 0


def _page():
    """The dashboard on demo shelf store1_47, showing the saved reading."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    page = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")
    at = AppTest.from_file(page, default_timeout=600).run()
    _by_label(at.selectbox, "Real shelf (held-out Grozi photo)").select("store1_47").run()
    if any(t.key == "show-saved" for t in at.toggle):
        at.toggle(key="show-saved").set_value(True).run()
    return at


def _plan(at, value):
    _by_label(at.radio, "Planogram").set_value(value).run()


def _metrics(at):
    return {x.label: x.value for x in at.metric}


def _by_label(elements, label):
    return next(e for e in elements if e.label == label)


def test_the_page_shows_fewer_steps_than_violation_lines(monkeypatch):
    monkeypatch.setenv("SHELFSENSE_SLM", "0")
    at = _page()
    _plan(at, "scenario")
    assert not at.exception
    lines = next(int(e.label.split()[1]) for e in at.expander if e.label.startswith("All ") and "violation lines" in e.label)
    assert _metrics(at)["Compliance score"] == "27 / 100" and int(_metrics(at)["Steps for staff"]) < lines


def test_edit_mode_starts_at_the_real_plan_and_a_swap_lowers_the_score(monkeypatch):
    monkeypatch.setenv("SHELFSENSE_SLM", "0")
    at = _page()
    _plan(at, "edit")
    assert not at.exception and _metrics(at)["Compliance score"] == "94 / 100"
    _by_label(at.selectbox, "Swap this slot").select("S1-01").run()
    _by_label(at.selectbox, "with this slot").select("S1-02").run()
    _by_label(at.button, "Swap their products").click().run()
    assert not at.exception
    assert int(_metrics(at)["Compliance score"].split()[0]) < 94
    assert any("Swap" in m.value for m in at.markdown)
    assert "vs the real plan" in [m.delta for m in at.metric][0]
    _by_label(at.button, "Reset to the shelf's plan").click().run()
    assert _metrics(at)["Compliance score"] == "94 / 100"


def test_live_run_reproduces_the_saved_score_and_reports_its_timing(monkeypatch):
    from shelfsense.vision import live

    if not live.available():
        pytest.skip("needs the Grozi product index and the detectors")
    monkeypatch.setenv("SHELFSENSE_SLM", "0")
    at = _page()
    _plan(at, "compliant")
    saved = _metrics(at)["Compliance score"]
    _by_label(at.button, "Run it live").click().run()
    assert not at.exception
    assert _metrics(at)["Compliance score"] == saved
    assert any("Live run on" in s.value and "in total" in s.value for s in at.success)
