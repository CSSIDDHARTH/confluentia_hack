import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from shelfsense.pricing.label_ocr import label_from_crop, parse_price, read_label_price, read_label_text

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "price_tags"
GROZI_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "grozi_labels"


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


@pytest.mark.parametrize("text", ["25.09.2012", "1070.644.000.00", "1102.308.000.00"])
def test_parse_price_rejects_dates_and_article_codes(text):
    # Grozi labels print a date and a dotted article code next to the
    # price; neither has the shape of a price.
    assert parse_price(text) is None


def test_parse_price_reads_a_colon_as_a_decimal_point():
    assert parse_price("2:90") == pytest.approx(2.90)


def test_parse_price_prefers_the_price_over_a_dotted_code():
    assert parse_price("510.431.000\n3.95") == pytest.approx(3.95)


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
    # tags. This is that measurement, not a guarantee -- plain Tesseract
    # (no custom training/preprocessing beyond label_ocr._preprocess) on
    # these small, tilted, barcode-cluttered real crops scores low
    # (measured ~15%: see data/README.md's Label OCR section). The
    # floor here is set from that measurement, so it catches a real
    # regression (e.g. breaking _preprocess) without being a brittle
    # pass/fail bar for an inherently hard, un-tuned baseline. Improving
    # the number for real needs better crops (more margin, deskewed) or
    # a scene-text model, not a threshold change.
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    legible = {name: truth for name, truth in manifest.items() if truth is not None}

    correct = 0
    results = []
    for name, truth in legible.items():
        img = Image.open(FIXTURES / name)
        ocr_text = read_label_text(img)
        got_price = parse_price(ocr_text)
        want_price = parse_price(truth)
        ok = got_price == pytest.approx(want_price) if want_price is not None else got_price is None
        correct += ok
        results.append(f"  {name}: truth={truth!r} ocr={ocr_text!r} got={got_price} want={want_price} {'OK' if ok else 'X'}")

    accuracy = correct / len(legible)
    detail = "\n".join(results)
    assert accuracy >= 0.15, f"OCR price accuracy {accuracy:.0%} on {len(legible)} hand-transcribed tags\n{detail}"


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason=SKIP_REASON)
def test_price_accuracy_on_real_grozi_labels():
    # Real Swiss supermarket labels from Grozi-3.2K, boxed and transcribed
    # by hand (scripts/prep_grozi_label_fixtures.py). Measured 28/31 with
    # read_label_price() and none of the 3 misses returned a wrong price:
    # an unreadable label is a cheap re-check, a wrong price is a false
    # mismatch alert. The floors leave room for tesseract version drift.
    manifest = json.loads((GROZI_FIXTURES / "manifest.json").read_text())

    correct = wrong = 0
    results = []
    for name, truth in manifest.items():
        price, confidence = read_label_price(Image.open(GROZI_FIXTURES / name))
        ok = price == pytest.approx(float(truth)) if price is not None else False
        correct += ok
        wrong += price is not None and not ok
        results.append(f"  {name}: truth={truth} got={price} conf={confidence:.1f} {'OK' if ok else 'X'}")

    detail = "\n".join(results)
    assert correct / len(manifest) >= 0.8, f"{correct}/{len(manifest)} correct\n{detail}"
    assert wrong <= 1, f"{wrong} labels read as a wrong price\n{detail}"


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason=SKIP_REASON)
def test_label_from_crop_builds_a_valid_label():
    img = Image.open(FIXTURES / "029_0.png")
    label = label_from_crop(
        "test-l1", [0, 0, img.width, img.height],
        shelf_id="S1", slot_id="S1-01", sku="SKU-1001", image=img,
    )
    assert label.label_id == "test-l1"
    assert label.ocr_text  # something was read


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason=SKIP_REASON)
def test_label_from_crop_reads_price_and_confidence():
    img = Image.open(GROZI_FIXTURES / "s2_10_0.jpg")  # "2.70", sharp
    label = label_from_crop(
        "test-l2", [0, 0, img.width, img.height],
        shelf_id="S1", slot_id="S1-01", sku="SKU-1001", image=img,
    )
    assert label.price == pytest.approx(2.70)
    assert 0.5 <= label.ocr_confidence <= 1.0
