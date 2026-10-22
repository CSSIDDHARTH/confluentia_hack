"""Run the whole vision chain on one shelf photo, live: blocks, naming, plan alignment, packs, empty spaces,
and optionally the price labels (detect, read, tie to slots).

Used by the dashboard's "run it live" button so the planogram demo does not depend on saved results. Loading
the models is the slow part (seconds); `run` after that takes a second or two on a laptop GPU.
"""

import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

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


def bay_photo(bay_dir: Path) -> tuple[Path, bool]:
    """The photo to read live for a demo bay: the original full-size Grozi photo named in its README.txt if it is
    here (the saved results were made from it), else the 1600 px copy stored with the bay. Second value: whether
    it is the original."""
    m = re.search(r"(data/grozi/\S+\.jpg)", (bay_dir / "README.txt").read_text())
    original = ROOT / m.group(1) if m else None
    return (original, True) if original and original.exists() else (bay_dir / "photo.jpg", False)


def prices_available() -> bool:
    """Reading prices live needs the label detector's weights and the tesseract binary."""
    from shelfsense.pricing.label_detector import DEFAULT_WEIGHTS

    return DEFAULT_WEIGHTS.exists() and shutil.which("tesseract") is not None


class LiveChain:
    def __init__(self, device: str | None = None):
        self.blocks = ProductDetector(MODELS / "product_detector.pt", device=device)
        self.packs = ProductDetector(MODELS / "pack_detector.pt", imgsz=1024, max_det=600, device=device)
        self.voids = ProductDetector(MODELS / "void_detector.pt", conf=VOID_CONF_FLOOR, imgsz=640, max_det=150, device=device)
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.matcher = SkuMatcher.load(INDEX, ADAPTER, embedder=Embedder(self.device))

    def run(
        self, image_path: str | Path, planogram: Planogram, shown_path: str | None = None, read_prices: bool = False,
        on_stage: Callable[[str, float], None] | None = None,
    ) -> LiveResult:
        """read_prices: also find the price labels, read them (Tesseract, plus EasyOCR if installed) and tie each to a slot.
        on_stage(stage, seconds) is called as each stage finishes."""
        timings: dict[str, float] = {}
        t = time.perf_counter()
        cands = {s.sku for sh in planogram.shelves for s in sh.slots}
        det = analyze_bay(
            image_path, self.blocks, self.packs, self.matcher, planogram, cands, shown_path=shown_path,
            void_detector=self.voids, timings=timings, on_stage=on_stage,
        )
        if read_prices:
            from shelfsense.pricing.label_assign import assign_labels_to_slots
            from shelfsense.pricing.label_detector import detect_labels

            t_labels = time.perf_counter()
            det.labels = assign_labels_to_slots(detect_labels(Image.open(image_path)), det, planogram)
            timings["read price labels"] = time.perf_counter() - t_labels
            if on_stage is not None:
                on_stage("read price labels", timings["read price labels"])
        return LiveResult(det, timings, time.perf_counter() - t, self.device)


_shared: LiveChain | None = None


def shared() -> LiveChain:
    """One LiveChain per process, so the dashboard's pages do not each load the models onto a small GPU."""
    global _shared
    if _shared is None:
        _shared = LiveChain()
    return _shared
