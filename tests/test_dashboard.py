from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")


def _check(at, what):
    assert not at.exception, what
    assert at.metric[0].label == "Compliance score"
    assert any(h.value == "Shift briefing" for h in at.subheader)


def _app():
    """The audit page showing the saved reading (with the live chain present it otherwise waits for Run it live)."""
    at = AppTest.from_file(APP, default_timeout=600).run()
    if any(t.key == "show-saved" for t in at.toggle):
        at.toggle(key="show-saved").set_value(True).run()
    return at


def test_before_a_live_run_only_the_photo_is_shown(monkeypatch):
    from shelfsense.vision import live

    if not live.available():
        pytest.skip("without the live chain the page shows the saved reading at once")
    monkeypatch.setenv("SHELFSENSE_SLM", "0")
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert not at.metric and any(b.label == "Run it live" for b in at.button)


def test_every_real_shelf_and_plan_renders(monkeypatch):
    monkeypatch.setenv("SHELFSENSE_SLM", "0")  # template summary: the model is checked by scripts/eval_briefing.py
    at = _app()
    _check(at, "first load")
    shelf_box = lambda: next(b for b in at.selectbox if b.label == "Real shelf (held-out Grozi photo)")  # noqa: E731
    shelves = shelf_box().options
    assert shelves, "no demo bays"
    for shelf in shelves:
        for plan in ("compliant", "scenario", "edit"):
            shelf_box().set_value(shelf)
            next(r for r in at.radio if r.label == "Planogram").set_value(plan).run()
            _check(at, (shelf, plan))
            assert at.metric[1].label == "Value at stake (estimate)"


def test_developer_inputs_show_every_mock(monkeypatch):
    monkeypatch.setenv("SHELFSENSE_SLM", "0")
    at = AppTest.from_file(APP, default_timeout=120).run()
    next(t for t in at.toggle if t.label == "Developer inputs").set_value(True).run()
    mocks = at.selectbox[0].options
    assert any(m.startswith("Low-stock demo") for m in mocks)
    for mock in mocks:
        at.selectbox[0].set_value(mock).run()
        _check(at, mock)


def test_live_run_with_prices_reproduces_the_saved_audit(monkeypatch):
    from shelfsense.vision import live

    if not (live.available() and live.prices_available()):
        pytest.skip("needs the Grozi product index, the detectors, the label detector and tesseract")
    monkeypatch.setenv("SHELFSENSE_SLM", "0")
    at = _app()
    at.selectbox[0].set_value("store4_9")
    next(r for r in at.radio if r.label == "Planogram").set_value("scenario").run()
    saved = at.metric[0].value
    next(b for b in at.button if b.label == "Run it live").click().run()
    assert not at.exception
    assert any("Live run on" in s.value and "read price labels" in s.value for s in at.success)
    assert at.metric[0].value == saved
