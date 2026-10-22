import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")
BAYS = sorted(p.name for p in (ROOT / "data" / "demo_bays").glob("*") if (p / "detection_scenario.json").exists())


def scores(at) -> list[int]:
    """The score rings on the page, in order."""
    return [int(m) for h in at.html for m in re.findall(r'data-score="(\d+)"', h.value)]


def app(monkeypatch, timeout: int = 120):
    monkeypatch.setenv("SHELFSENSE_SLM", "0")  # template summary: the model is checked by scripts/eval_briefing.py
    return AppTest.from_file(APP, default_timeout=timeout).run()


def open_shelf(at, bay: str, plan: str, saved: bool = True):
    at.button(key=f"open-{bay}").click().run()
    at.segmented_control(key="plan-shelf").set_value(plan).run()
    if saved and any(t.key == "show-saved" for t in at.toggle):
        at.toggle(key="show-saved").set_value(True).run()
    return at


def test_home_shows_one_card_per_shelf_on_both_plans(monkeypatch):
    at = app(monkeypatch)
    assert not at.exception
    assert [m.label for m in at.metric] == ["Shelves checked", "Money at stake", "To-dos"]
    current = scores(at)
    assert len(current) == len(BAYS)
    at.segmented_control(key="plan-home").set_value("scenario").run()
    assert not at.exception
    assert sum(scores(at)) < sum(current)  # head office's new plan does not match the shelves yet


def test_every_shelf_and_plan_renders(monkeypatch):
    assert BAYS, "no demo bays"
    for bay in BAYS:
        for plan in ("compliant", "scenario", "edit"):
            at = open_shelf(app(monkeypatch), bay, plan)
            assert not at.exception, (bay, plan)
            assert len(scores(at)) == 1, (bay, plan)
            assert [t.label for t in at.tabs] == ["To-do list", "Prices", "Shift summary", "Details"]
            assert [m.label for m in at.metric] == ["Money at stake", "To-dos"]


def test_before_a_check_only_the_photo_and_the_plan_are_shown(monkeypatch):
    from shelfsense.vision import live

    if not live.available():
        pytest.skip("without the live chain the page shows the saved check at once")
    at = open_shelf(app(monkeypatch), BAYS[0], "scenario", saved=False)
    assert not at.exception
    assert not at.metric and not scores(at)
    assert any(b.label == "Check the shelf now" for b in at.button)


def test_back_and_scan_navigation(monkeypatch):
    at = open_shelf(app(monkeypatch), BAYS[0], "compliant")
    at.button(key="back").click().run()
    assert [m.label for m in at.metric][:1] == ["Shelves checked"]
    at.button(key="open-scan").click().run()
    assert not at.exception and any(m.label == "Overall fill" for m in at.metric)


def test_developer_inputs_show_every_mock(monkeypatch):
    at = app(monkeypatch)
    next(t for t in at.toggle if t.label == "Developer inputs").set_value(True).run()
    mocks = at.selectbox[0].options
    assert any(m.startswith("Low-stock demo") for m in mocks)
    for mock in mocks:
        at.selectbox[0].set_value(mock).run()
        assert not at.exception, mock
        assert len(scores(at)) == 1


def test_live_check_with_prices_reproduces_the_saved_audit(monkeypatch):
    from shelfsense.vision import live

    if not (live.available() and live.prices_available()):
        pytest.skip("needs the Grozi product index, the detectors, the label detector and tesseract")
    at = open_shelf(app(monkeypatch, timeout=600), "store4_9", "scenario")
    saved = scores(at)
    next(b for b in at.button if b.label == "Check the shelf now").click().run()
    assert not at.exception
    assert any("Checked live in" in h.value for h in at.html)
    assert scores(at) == saved
