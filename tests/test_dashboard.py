from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")


def test_every_preset_renders():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    presets = at.selectbox[0].options
    assert any(p.startswith("Real bay") for p in presets)
    for preset in presets:
        at.selectbox[0].set_value(preset).run()
        assert not at.exception, preset
        assert at.metric[0].label == "Compliance score"
