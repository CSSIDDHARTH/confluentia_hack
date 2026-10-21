"""Run the whole vision chain on one shelf photo, live: blocks, naming, plan alignment, packs, empty spaces.

Used by the dashboard's "run it live" button so the planogram demo does not depend on saved results. Loading
the models is the slow part (seconds); `run` after that takes a second or two on a laptop GPU.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from shelfsense.contracts import DetectionOutput, Planogram
from shelfsense.vision.bay import analyze_bay
from shelfsense.vision.detect import VOID_CONF_FLOOR, ProductDetector
from shelfsense.vision.sku_match import Embedder, SkuMatcher

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "data" / "grozi" / "sku_index.npz"
ADAPTER = ROOT / "models" / "sku_adapter.npy"
MODELS = ROOT / "models"


@dataclass
class LiveResult:
    det: DetectionOutput
    timings: dict[str, float]  # seconds per stage
    total: float
    device: str


def available() -> bool:
    """The chain needs the Grozi reference index (built by scripts/build_sku_index.py) and all three detectors."""
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return INDEX.exists() and ADAPTER.exists() and all((MODELS / f).exists() for f in ("product_detector.pt", "pack_detector.pt", "void_detector.pt"))


class LiveChain:
    def __init__(self, device: str | None = None):
        self.blocks = ProductDetector(MODELS / "product_detector.pt", device=device)
        self.packs = ProductDetector(MODELS / "pack_detector.pt", imgsz=1024, max_det=600, device=device)
        self.voids = ProductDetector(MODELS / "void_detector.pt", conf=VOID_CONF_FLOOR, imgsz=640, max_det=150, device=device)
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.matcher = SkuMatcher.load(INDEX, ADAPTER, embedder=Embedder(self.device))

    def run(self, image_path: str | Path, planogram: Planogram, shown_path: str | None = None) -> LiveResult:
        timings: dict[str, float] = {}
        t = time.perf_counter()
        cands = {s.sku for sh in planogram.shelves for s in sh.slots}
        det = analyze_bay(image_path, self.blocks, self.packs, self.matcher, planogram, cands, shown_path=shown_path, void_detector=self.voids, timings=timings)
        return LiveResult(det, timings, time.perf_counter() - t, self.device)
