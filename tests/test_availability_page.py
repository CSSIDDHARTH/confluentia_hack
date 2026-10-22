from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")
DEMO = ROOT / "data" / "availability_demo"

pytestmark = pytest.mark.skipif(not any(DEMO.glob("*.json")), reason="no saved availability results")


def metrics(at):
    return {m.label: m.value for m in at.metric}


def photo_mode(timeout: int = 120):
    """The dashboard's "Check any shelf photo" page (the availability view)."""
    at = AppTest.from_file(APP, default_timeout=timeout).run()
    at.button(key="open-scan").click().run()
    return at


def test_page_renders_saved_shelf_without_errors():
    at = photo_mode()
    assert not at.exception
    m = metrics(at)
    assert {"Overall fill", "Empty spaces", "Packs to refill", "Low-stock rows", "Packs found"} <= m.keys()
    assert int(m["Packs found"]) > 50


def test_sliders_recompute_from_saved_packs():
    at = photo_mode()
    base = int(metrics(at)["Empty spaces"])
    # the first slider is the empty-space confidence: a high bar keeps fewer spaces, a low one more
    at.slider[0].set_value(0.8).run()
    strict = int(metrics(at)["Empty spaces"])
    at.slider[0].set_value(0.1).run()
    loose = int(metrics(at)["Empty spaces"])
    assert strict <= base <= loose and strict < loose
    assert not at.exception


def test_every_saved_shelf_loads():
    at = photo_mode()
    for name in at.pills(key="sample").options:
        at.pills(key="sample").set_value(name).run()
        assert not at.exception, name


def _live_detector_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return (ROOT / "models" / "pack_detector.pt").exists()


@pytest.mark.skipif(not _live_detector_available(), reason="needs ultralytics and models/pack_detector.pt")
def test_uploading_a_photo_runs_the_detector_and_shows_results():
    photo = ROOT / "data" / "test_images" / "shelf_06_coffee_us.jpg"  # tracked in the repo
    at = photo_mode(600)
    at.segmented_control(key="scan-source").set_value("Upload a photo").run()
    at.get("file_uploader")[0].upload(photo.name, photo.read_bytes(), "image/jpeg")
    at.run()
    assert not at.exception
    assert int(metrics(at)["Packs found"]) > 50
    assert len(at.get("download_button")) == 3
