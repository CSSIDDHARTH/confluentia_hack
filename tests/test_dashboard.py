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


def test_every_real_shelf_and_plan_renders(monkeypatch):
    monkeypatch.setenv("SHELFSENSE_SLM", "0")  # template summary: the model is checked by scripts/eval_briefing.py
    at = AppTest.from_file(APP, default_timeout=120).run()
    _check(at, "first load")
    shelves = at.selectbox[0].options
    assert shelves, "no demo bays"
    for shelf in shelves:
        for plan in ("compliant", "scenario"):
            at.selectbox[0].set_value(shelf)
            at.radio[0].set_value(plan).run()
            _check(at, (shelf, plan))
            assert at.metric[1].label == "Value at stake (estimate)"


def test_developer_inputs_show_every_mock(monkeypatch):
    monkeypatch.setenv("SHELFSENSE_SLM", "0")
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle[0].set_value(True).run()
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
    at = AppTest.from_file(APP, default_timeout=600).run()
    at.selectbox[0].set_value("store4_9")
    at.radio[0].set_value("scenario").run()
    saved = at.metric[0].value
    next(b for b in at.button if b.label == "Run it live").click().run()
    assert not at.exception
    assert any("Live run on" in s.value and "read price labels" in s.value for s in at.success)
    assert at.metric[0].value == saved
