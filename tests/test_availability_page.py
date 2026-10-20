from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "app" / "pages" / "1_Shelf_availability.py")
DEMO = ROOT / "data" / "availability_demo"

pytestmark = pytest.mark.skipif(not any(DEMO.glob("*.json")), reason="no saved availability results")


def metrics(at):
    return {m.label: m.value for m in at.metric}


def test_page_renders_saved_shelf_without_errors():
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    assert not at.exception
    m = metrics(at)
    assert {"Overall fill", "Empty spaces", "Packs to refill", "Low-stock rows", "Packs found"} <= m.keys()
    assert int(m["Packs found"]) > 50


def test_sliders_recompute_from_saved_packs():
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    base = int(metrics(at)["Empty spaces"])
    # a very large minimum gap leaves (almost) nothing counted as empty; a tiny one finds more
    at.slider[0].set_value(2.0).run()
    strict = int(metrics(at)["Empty spaces"])
    at.slider[0].set_value(0.4).run()
    loose = int(metrics(at)["Empty spaces"])
    assert strict <= base <= loose and strict < loose
    assert not at.exception


def test_every_saved_shelf_loads():
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    for name in at.selectbox[0].options:
        at.selectbox[0].select(name).run()
        assert not at.exception, name
