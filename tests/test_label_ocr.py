import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from shelfsense.pricing.label_ocr import label_from_crop, parse_price, read_label_text

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "price_tags"


# ---------- parse_price: pure text -> float, no OCR engine needed ----------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("$3.49", 3.49),
        ("3.49", 3.49),
        ("$10.99", 10.99),
        ("0.70", 0.70),
        ("RM 15.90", 15.90),
        ("8,65", 8.65),  # European comma-decimal
        ("1,234.56", 1234.56),  # thousands comma + decimal point
    ],
)
def test_parse_price_simple_decimal_formats(text, expected):
    assert parse_price(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2.140", 2140.0),  # period as thousands separator, no cents shown
        ("1.505", 1505.0),
        ("2.235", 2235.0),
    ],
)
def test_parse_price_period_as_thousands_separator(text, expected):
    assert parse_price(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("118円", 118.0),
        ("94円", 94.0),
        ("258円", 258.0),
        ("218円", 218.0),
    ],
)
def test_parse_price_whole_unit_currency_marker(text, expected):
    assert parse_price(text) == pytest.approx(expected)


def test_parse_price_picks_the_price_not_surrounding_numbers():
    # "3/$10.99" -- the "3" (a quantity) must not win over the price.
    assert parse_price("WEEKLY SPECIAL\nHONEY BUNCHES OF OATS\n3/$10.99") == pytest.approx(10.99)


def test_parse_price_ignores_a_barcode_next_to_the_price():
    text = "7891234567890123\nSKU 445\n$4.99"
    assert parse_price(text) == pytest.approx(4.99)


def test_parse_price_bare_ambiguous_number_returns_none():
    # No separator, no currency marker -- can't tell where the decimal
    # point should go (this is a real OCR failure mode, not a bug: a
    # misread "9,65" can come back as bare "965").
    assert parse_price("965") is None


def test_parse_price_empty_or_no_digits_returns_none():
    assert parse_price("") is None
    assert parse_price("SALE") is None


def test_parse_price_short_bare_number_is_accepted():
    # <=2 digits has nowhere to hide a lost decimal point either way.
    assert parse_price("70") == pytest.approx(70.0)


# ---------- read_label_text / label_from_crop: need the tesseract binary ----------

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None
SKIP_REASON = (
    "tesseract binary not on PATH (`sudo apt install tesseract-ocr`); "
    "parse_price() above is still fully tested without it"
)


def test_read_label_text_raises_a_clear_error_without_tesseract():
    if TESSERACT_AVAILABLE:
        pytest.skip("tesseract is installed; this tests the missing-binary path")
    img = Image.open(FIXTURES / "002_0.png")
    with pytest.raises(RuntimeError, match="tesseract"):
        read_label_text(img)


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason=SKIP_REASON)
def test_ocr_accuracy_against_hand_transcribed_fixtures():
    # Assumption 7: OCR accuracy is measured against hand-transcribed
    # tags. Legible tags (manifest value is not None) should parse to
    # the same price our ground truth does; illegible ones (None) just
    # shouldn't crash.
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    legible = {name: truth for name, truth in manifest.items() if truth is not None}

    correct = 0
    for name, truth in legible.items():
        img = Image.open(FIXTURES / name)
        got_price = parse_price(read_label_text(img))
        if got_price == pytest.approx(parse_price(truth)):
            correct += 1

    accuracy = correct / len(legible)
    assert accuracy >= 0.5, f"OCR price accuracy {accuracy:.0%} on {len(legible)} hand-transcribed tags"


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason=SKIP_REASON)
def test_label_from_crop_builds_a_valid_label():
    img = Image.open(FIXTURES / "029_0.png")
    label = label_from_crop(
        "test-l1", [0, 0, img.width, img.height],
        shelf_id="S1", slot_id="S1-01", sku="SKU-1001", image=img,
    )
    assert label.label_id == "test-l1"
    assert label.ocr_text  # something was read
