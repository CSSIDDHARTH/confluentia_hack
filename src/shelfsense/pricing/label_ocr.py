"""Label OCR: read a cropped shelf-edge label image and extract its
price, the "label detection, OCR, price parsing" phase CLAUDE.md gives
Person B.

This module is the producer of the text/price a `contracts.Label`
carries -- `price_check.py` consumes already-filled labels; this one
fills them, given a crop. Label *detection* (finding where on the shelf
image the tag is) isn't implemented here: that needs a trained detector
(data/price_tags/ is for that). `label_from_crop()` takes a bbox someone
else already found (ground truth in tests, eventually a detector's) and
turns the crop into a `Label`.

Three pieces:
- `read_label_text()`: OCR via Tesseract (through pytesseract). Needs
  the `tesseract` binary on PATH (`sudo apt install tesseract-ocr`,
  `brew install tesseract`, ...) -- raises a clear error if it's
  missing, rather than failing confusingly inside pytesseract.
- `read_label_price()`: the price itself. Runs several OCR variants,
  takes the tallest price-like word from each (the price is the biggest
  print on a shelf label; dates, article codes and unit prices are
  small), and returns the price most variants agree on, with the share
  of agreeing variants as its confidence.
- `parse_price()`: pure text -> float, no OCR engine needed. Handles
  the price-tag formats actually seen in tests/fixtures/price_tags/
  (see its manifest.json): plain "$10.99", comma-decimal "0,70",
  period-as-thousands "2.140" (no cents shown), a whole-unit currency
  word/symbol stuck to the number ("118円"), extra text/codes around
  the price ("3/$10.99", barcode digits, SKU codes). Ambiguous bare
  numbers with no separator and no currency marker (e.g. OCR reads
  "965" off a tag that's actually "9,65" with the comma lost) return
  None rather than guess -- these should surface as unreadable_price,
  same as a genuine OCR failure.
"""

from __future__ import annotations

import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageFilter, ImageOps

from shelfsense.contracts import Label

# Label crops tend to be small (the dataset in tests/fixtures/price_tags/
# averages ~200px wide) and often tilted; Tesseract's default settings
# read these as empty far more often than not. Upscaling + contrast
# stretch + a page-segmentation mode for "one block of text" (rather
# than the default "assume a page layout") recovers real signal on most
# of them -- see tests/test_label_ocr.py's accuracy tests and
# data/README.md.
_TARGET_WIDTH = 500
_TESSERACT_CONFIG = "--psm 6"

# (preprocessing kwargs, page-segmentation mode) for read_label_price().
# No single setting reads every label: blur, glare and print size each
# break a different one. Measured on tests/fixtures/grozi_labels/: one
# variant alone 68%, these five voting 94% (data/README.md).
_PRICE_VARIANTS = [
    ({}, 6),
    ({}, 11),
    ({"sharpen": True}, 6),
    ({"width": 900}, 6),
    ({"width": 900}, 11),
]


# EasyOCR is a second engine that votes alongside Tesseract: on small, soft
# shelf labels it reads the big price print far more reliably, though it
# returns nothing on the blurriest ones. Optional: without the package only
# Tesseract votes. Each variant's weight is how many Tesseract variants it counts as.
_EASYOCR_SCALES = (1, 2)
# Without a GPU the 2x pass is most of the cost: on the 31 Grozi labels, on 4 CPU cores, both scales read
# 29 correct (25 confident) at 2.2 s a label, scale 1 alone 29 correct (24 confident) at 0.7 s, none wrong
# either way. So a CPU-only device reads at scale 1 only.
_EASYOCR_CPU_SCALES = (1,)
_EASYOCR_WEIGHT = 1.5
_easyocr_reader = None


def _easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
        except ImportError:
            _easyocr_reader = False
        else:
            # Never download at run time: without internet (a demo venue) the first read would hang.
            # Fetch the models once with `python scripts/download_data.py easyocr`.
            import torch

            try:
                _easyocr_reader = easyocr.Reader(
                    ["en"], gpu=torch.cuda.is_available(), verbose=False, download_enabled=False
                )
            except (FileNotFoundError, OSError, RuntimeError):
                _easyocr_reader = False
    return _easyocr_reader or None


def _easyocr_price(image: Image.Image, scale: int) -> float | None:
    """The price read from the tallest price-like word EasyOCR finds."""
    import numpy as np

    reader = _easyocr()
    grey = ImageOps.autocontrast(image.convert("L"), cutoff=1)
    if scale > 1:
        grey = grey.resize((grey.width * scale, grey.height * scale), Image.BICUBIC)
    best: tuple[float, float] | None = None
    for box, word, conf in reader.readtext(np.array(grey), allowlist="0123456789.,", text_threshold=0.5):
        if conf < 0.5 or not _MARKED_RE.search(word):
            continue
        price = parse_price(word.strip())
        height = box[2][1] - box[0][1]
        if price is not None and (best is None or height > best[0]):
            best = (height, price)
    return best[1] if best else None


def _preprocess(image: Image.Image, width: int = _TARGET_WIDTH, sharpen: bool = False) -> Image.Image:
    grey = image.convert("L")
    scale = max(1, width // grey.width)
    if scale > 1:
        grey = grey.resize((grey.width * scale, grey.height * scale), Image.LANCZOS)
    grey = ImageOps.autocontrast(grey)
    if sharpen:
        grey = grey.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return grey

_NUMBER_RE = re.compile(r"\d[\d.,]*\d|\d")
# How a price can be grouped: "1.234,56" / "1,234" (thousands groups,
# optional cents) or "12.99" / "2.140" / "70". Rejects dates
# ("25.09.2012") and article codes ("1070.644.000.00") outright.
_PRICE_SHAPE_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?|\d+(?:[.,]\d{2,3})?")
# OCR often reads a price's decimal point as a colon ("2:90").
_COLON_DECIMAL_RE = re.compile(r"(\d):(\d{2})(?!\d)")
_CURRENCY_SYMBOLS = "$€£¥₹"
# Currencies/denominations commonly printed with no decimal places, so a
# bare multi-digit number next to one of these is a whole-unit price,
# not an ambiguous OCR read.
_WHOLE_UNIT_MARKERS = ("円", "¥", "JPY", "YEN", "WON", "KRW")
# A word read_label_price() will consider: a decimal/thousands separator
# between digits, a currency symbol right before digits, or a whole-unit
# marker right after them.
_MARKED_RE = re.compile(
    r"\d[.,:]\d|[" + _CURRENCY_SYMBOLS + r"]\d|\d\s?(?:" + "|".join(_WHOLE_UNIT_MARKERS) + ")"
)


class _OneThreadEnv(Mapping):
    """os.environ as it is at each call, plus OMP_THREAD_LIMIT=1: the environment Tesseract runs in.

    Tesseract starts one OpenMP thread per core and they spin while waiting. Five reads side by side on a
    4-core device then take minutes instead of under a second (measured: 87 s vs 0.4 s), so each Tesseract
    process gets one thread. Only Tesseract's subprocesses see this; torch in this process is untouched."""

    def _env(self) -> dict[str, str]:
        return {**os.environ, "OMP_THREAD_LIMIT": "1"}

    def __getitem__(self, key):
        return self._env()[key]

    def __iter__(self):
        return iter(self._env())

    def __len__(self):
        return len(self._env())


def _require_tesseract() -> None:
    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "label OCR needs the `tesseract` binary on PATH "
            "(e.g. `sudo apt install tesseract-ocr`, `brew install "
            "tesseract`); `pip install pytesseract` alone only installs "
            "the Python wrapper."
        )
    import pytesseract.pytesseract as runner

    runner.environ = _OneThreadEnv()  # pytesseract passes this module global to every tesseract subprocess


def read_label_text(image: Image.Image) -> str:
    """OCR a label crop to text via Tesseract."""
    _require_tesseract()
    import pytesseract  # deferred: only needed when OCR actually runs

    return pytesseract.image_to_string(_preprocess(image), config=_TESSERACT_CONFIG).strip()


def _tallest_price(image: Image.Image, psm: int) -> float | None:
    """The price read from the tallest price-like word Tesseract finds."""
    import pytesseract

    data = pytesseract.image_to_data(
        image, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    best: tuple[int, float] | None = None
    for word, height in zip(data["text"], data["height"]):
        # A lone "4" or "30" is usually a stray read of part of a word,
        # so a word only counts with a separator or currency marker.
        if not _MARKED_RE.search(word):
            continue
        # "(100ml = 1.62)": a unit price, never the label's price
        if word.startswith("(") or word.endswith(")"):
            continue
        price = parse_price(word.strip())
        if price is not None and (best is None or height > best[0]):
            best = (height, price)
    return best[1] if best else None


def read_label_price(image: Image.Image) -> tuple[float | None, float]:
    """Read the price off a label crop: (price, confidence).

    confidence is the (weighted) share of OCR variants that agree on the
    returned price (0.0 with price None when no variant reads one).
    """
    _require_tesseract()
    votes: Counter[float] = Counter()
    total = float(len(_PRICE_VARIANTS))
    use_easyocr = _easyocr() is not None
    # Each Tesseract variant is its own process, so they run side by side, and EasyOCR (one call at a
    # time: its reader is shared) runs alongside them. Same reads as one after another, about 3x faster.
    with ThreadPoolExecutor(max_workers=len(_PRICE_VARIANTS) + 1) as pool:
        tesseract = [pool.submit(_tallest_price, _preprocess(image, **kw), psm) for kw, psm in _PRICE_VARIANTS]
        scales = _EASYOCR_SCALES if use_easyocr and _easyocr().device != "cpu" else _EASYOCR_CPU_SCALES
        easy = pool.submit(lambda: [_easyocr_price(image, s) for s in scales]) if use_easyocr else None
        for job in tesseract:
            if (price := job.result()) is not None:
                votes[price] += 1
        if easy is not None:
            total += _EASYOCR_WEIGHT * len(scales)
            for price in easy.result():
                if price is not None:
                    votes[price] += _EASYOCR_WEIGHT
    if not votes:
        return None, 0.0
    price, weight = votes.most_common(1)[0]
    return price, weight / total


def _score(text: str, match: re.Match) -> tuple[int, int]:
    before = text[max(0, match.start() - 2) : match.start()]
    after = text[match.end() : match.end() + 4]
    has_currency = any(c in before for c in _CURRENCY_SYMBOLS) or any(
        marker in after for marker in _WHOLE_UNIT_MARKERS
    )
    has_separator = "." in match.group() or "," in match.group()
    has_cents = re.search(r"[.,]\d{2}$", match.group()) is not None
    # (does this look like a price, how long/specific is the match):
    # currency/whole-unit adjacency and a decimal separator both strongly
    # outrank raw digit count, so a 13-digit barcode never beats "$10.99",
    # and cents outrank thousands groups, so "3.95" beats "510.431.000".
    score = int(has_currency) * 100 + int(has_separator) * 50 + int(has_cents) * 25
    return (score, len(match.group()))


def parse_price(text: str) -> float | None:
    """Pick the most price-like number in `text` and parse it to a
    float, or None if nothing in it can be confidently read as a price.
    """
    if not text:
        return None

    text = _COLON_DECIMAL_RE.sub(r"\1.\2", text)
    candidates = [m for m in _NUMBER_RE.finditer(text) if _PRICE_SHAPE_RE.fullmatch(m.group())]
    if not candidates:
        return None

    best = max(candidates, key=lambda m: _score(text, m))
    raw = best.group()

    separators = [m.start() for m in re.finditer(r"[.,]", raw)]
    if not separators:
        if len(raw) <= 2:
            return float(raw)
        tail = text[best.end() : best.end() + 4]
        if any(marker in tail for marker in _WHOLE_UNIT_MARKERS):
            return float(raw)
        return None  # e.g. "965" with no separator or currency marker: ambiguous

    last_sep = separators[-1]
    digits_after = len(raw) - last_sep - 1
    if digits_after == 2:
        integer_part = re.sub(r"[.,]", "", raw[:last_sep])
        decimal_part = raw[last_sep + 1 :]
        return float(f"{integer_part}.{decimal_part}")
    # 3 digits after the last separator: period/comma used as a thousands
    # separator, not a decimal point (_PRICE_SHAPE_RE allows nothing else)
    return float(re.sub(r"[.,]", "", raw))


def label_from_crop(
    label_id: str,
    bbox: list[float],
    *,
    shelf_id: str | None,
    slot_id: str | None,
    sku: str | None,
    image: Image.Image,
) -> Label:
    """OCR a label crop into a contracts.Label, ready for
    price_check.check_prices(). The price and ocr_confidence come from
    read_label_price(); ocr_text is the plain single-pass read, kept for
    display and debugging.
    """
    price, confidence = read_label_price(image)
    return Label(
        label_id=label_id,
        bbox=bbox,
        shelf_id=shelf_id,
        slot_id=slot_id,
        sku=sku,
        ocr_text=read_label_text(image),
        price=price,
        ocr_confidence=confidence,
    )
