# data/

Everything here is git-ignored except `data/mock/`, `data/test_images/`
and this file.

| Folder | Contents | Source |
|---|---|---|
| `mock/` | Synthetic shelf images matching their `.json` detection outputs (and `contracts/detection_output.example.json`) | `python scripts/make_mock_image.py`, `python scripts/gen_scenario.py` |
| `test_images/` | The fixed 8 images every phase is evaluated on. Real shelf photos, no annotations (detection/vision testing only -- see `test_images/README.md`). | SKU-110K, see below |
| `supermarket_shelves/` | Not tracked (~140MB, git-ignored). 45 real shelf images with `Product`/`Price` bounding-box annotations (Supervisely format, `meta.json` + `annotations/*.jpg.json`). Used to build `tests/fixtures/price_tags/` (see `scripts/prep_price_tag_fixtures.py`); also useful later for training/testing real label *detection*, which isn't built yet. | [Humans in the Loop](https://humansintheloop.org/resources/datasets/supermarket-shelves-dataset/), CC0. Requires filling in a form on their site to get the download link -- ask whoever last did this if you need it again. |
| `grozi/` | Not fetched. Grocery Products / Grozi-3.2K + GP-180 planograms -- Drive-hosted, not a plain download | https://github.com/tobiagru/ObjectDetectionGroceryProducts, https://alessiotonioni.github.io/publication/planogram |

**`grocery_dataset/` (Varol & Kuzu, 2014,
https://github.com/gulvarol/grocerydataset) is not used.** Despite the
name, it's entirely cigarette-shelf photos with graphic tobacco health
warnings -- unsuitable for this project. Raise with the team before
relying on it; CLAUDE.md's assumption 8 should probably be corrected.

Later phases: Roboflow empty-shelf sets (gaps).

## Label OCR

`shelfsense.pricing.label_ocr.read_label_text()` needs the `tesseract`
binary (`sudo apt install tesseract-ocr`), not just `pip install
pytesseract` (that's just the Python wrapper, see pyproject.toml's `ocr`
extra). It isn't installed in every dev environment, so
`tests/test_label_ocr.py`'s OCR-against-real-images tests skip
gracefully without it; `parse_price()`'s text-parsing tests don't need
it and always run.

**Measured accuracy is low (~15%)** with tesseract installed, against
the 13 legible hand-transcribed crops in `tests/fixtures/price_tags/`
-- plain Tesseract (upscaled + contrast-stretched, `--psm 6`, no custom
training) on small, tilted, barcode-cluttered real photos. It's not
broken; this is what an un-tuned generic OCR engine actually gets on
hard real-world tags, which is itself the point of assumption 7's
"measured on hand-transcribed ... tags". The synthetic scenarios
(`scripts/gen_scenario.py`) remain the reliable path for demoing
price-check/compliance logic; improving real OCR for real needs either
larger/deskewed crops (a label-*detection* step feeding tighter boxes)
or a scene-text-tuned model, not just parse_price() tweaks.
