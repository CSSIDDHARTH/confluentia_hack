"""Label OCR: read a cropped shelf-edge label image and extract its
price, the "label detection, OCR, price parsing" phase CLAUDE.md gives
Person B.

This module is the producer of the text/price a `contracts.Label`
carries -- `price_check.py` consumes already-filled labels; this one
fills them, given a crop. Label *detection* (finding where on the shelf
image the tag is) isn't implemented here: that needs a trained detector,
out of scope for this pass. `label_from_crop()` takes a bbox someone
else already found (ground truth in tests, eventually the vision
stage's) and turns the crop into a `Label`.

Two independent pieces:
- `read_label_text()`: OCR via Tesseract (through pytesseract). Needs
  the `tesseract` binary on PATH (`sudo apt install tesseract-ocr`,
  `brew install tesseract`, ...) -- raises a clear error if it's
  missing, rather than failing confusingly inside pytesseract.
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

import re
import shutil

from PIL import Image

from shelfsense.contracts import Label

_NUMBER_RE = re.compile(r"\d[\d.,]*\d|\d")
_CURRENCY_SYMBOLS = "$€£¥₹"
# Currencies/denominations commonly printed with no decimal places, so a
# bare multi-digit number next to one of these is a whole-unit price,
# not an ambiguous OCR read.
_WHOLE_UNIT_MARKERS = ("円", "¥", "JPY", "YEN", "WON", "KRW")


def read_label_text(image: Image.Image) -> str:
    """OCR a label crop to text via Tesseract."""
    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "read_label_text() needs the `tesseract` binary on PATH "
            "(e.g. `sudo apt install tesseract-ocr`, `brew install "
            "tesseract`); `pip install pytesseract` alone only installs "
            "the Python wrapper."
        )
    import pytesseract  # deferred: only needed when OCR actually runs

    return pytesseract.image_to_string(image).strip()


def _score(text: str, match: re.Match) -> tuple[int, int]:
    before = text[max(0, match.start() - 2) : match.start()]
    after = text[match.end() : match.end() + 4]
    has_currency = any(c in before for c in _CURRENCY_SYMBOLS) or any(
        marker in after for marker in _WHOLE_UNIT_MARKERS
    )
    has_separator = "." in match.group() or "," in match.group()
    # (does this look like a price, how long/specific is the match):
    # currency/whole-unit adjacency and a decimal separator both strongly
    # outrank raw digit count, so a 13-digit barcode never beats "$10.99".
    return (int(has_currency) * 100 + int(has_separator) * 50, len(match.group()))


def parse_price(text: str) -> float | None:
    """Pick the most price-like number in `text` and parse it to a
    float, or None if nothing in it can be confidently read as a price.
    """
    if not text:
        return None

    candidates = list(_NUMBER_RE.finditer(text))
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
    if digits_after == 3:
        # period/comma used as a thousands separator, not a decimal point
        return float(re.sub(r"[.,]", "", raw))
    return None


def label_from_crop(
    label_id: str,
    bbox: list[float],
    *,
    shelf_id: str | None,
    slot_id: str | None,
    sku: str | None,
    image: Image.Image,
) -> Label:
    """OCR + parse a label crop into a contracts.Label, ready for
    price_check.check_prices(). ocr_confidence is left at a fixed
    placeholder; Tesseract doesn't expose a simple scalar confidence
    from image_to_string (image_to_data does, per-word, if ever needed).
    """
    text = read_label_text(image)
    price = parse_price(text)
    return Label(
        label_id=label_id,
        bbox=bbox,
        shelf_id=shelf_id,
        slot_id=slot_id,
        sku=sku,
        ocr_text=text,
        price=price,
        ocr_confidence=0.9 if price is not None else 0.3,
    )
