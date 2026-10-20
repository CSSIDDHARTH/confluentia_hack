# data/

Everything here is git-ignored except `data/mock/`, `data/test_images/`
and this file. Fetch the two main datasets with
`python scripts/download_data.py all` (needs `pip install py7zr`, about 2 GB).

| Folder | Contents | Used for | Source / licence |
|---|---|---|---|
| `mock/` | Synthetic shelf images matching their `.json` detection outputs (and `contracts/detection_output.example.json`) | Dashboard and report development | `python scripts/make_mock_image.py`, `python scripts/gen_scenario.py` |
| `test_images/` | The fixed 8 images every phase is evaluated on. Real shelf photos, no annotations (detection/vision testing only -- see `test_images/README.md`). | Vision checks | SKU-110K, academic use only |
| `grozi/` | **Grocery Products / Grozi-3.2K.** 680 real Swiss supermarket shelf photos with product boxes, studio reference images and planograms, and visible shelf-edge price labels. Details below. | Phases 1-5 (detection, SKU matching, planogram diff, price labels) | Academic research only |
| `price_tags/` | **Open Food Facts price-tag detection.** 2,229 real shop photos (1,982 train / 247 val), 1 class `price-tag`. Mix of full shelves and close-ups. Details below. | Phase 5 (label detector) | CC BY-SA 4.0 |
| `sku110k_subset/`, `sku110k_yolo/` | 800 train + 200 val SKU-110K shelf photos (one box per pack, ~147 per image), fetched as parquet and converted to YOLO format. Fetch with `python scripts/download_data.py sku110k`, convert with `python scripts/prep_sku110k_yolo.py` (needs `pip install pyarrow`). | Phase 4 pack detector | SKU-110K: academic and non-commercial use only; mirrored on Hugging Face as `adnankhan-11/smart-retail-shelf-auditing-v1` |
| `demo_bays/` | Five held-out Grozi shelves with a compliant planogram, a scenario planogram with planted changes, a synthetic catalogue, detection outputs and overlays for each. Built by `python scripts/make_demo_bays.py` (needs `data/grozi`, `models/`, and `python scripts/build_sku_index.py`). Load one in the dashboard via Advanced: custom paths. | Demo | Derived from Grozi, academic use only |
| `supermarket_shelves/` | Not tracked (~140MB, git-ignored). 45 real shelf images with `Product`/`Price` bounding-box annotations (Supervisely format, `meta.json` + `annotations/*.jpg.json`). Used to build `tests/fixtures/price_tags/` (see `scripts/prep_price_tag_fixtures.py`). | Label OCR fixtures | [Humans in the Loop](https://humansintheloop.org/resources/datasets/supermarket-shelves-dataset/), CC0. Requires filling in a form on their site to get the download link -- ask whoever last did this if you need it again. |

## grozi/

- `Testing/store{1..5}/images/N.jpg`: 680 shelf photos (3264×2448).
- `Testing/store{1..5}/annotation/N.csv`: `class,label,left_x,right_x,top_y,bottom_y`, coordinates relative (0–1). 2,668 boxes, 909 distinct products, `label = -1` means the product is not in the reference set (238 boxes).
  - **Each box covers a product block** (all facings of one product), not individual packs.
- `Testing/store{1..5}/annotation/N_plano.json`: planogram as a neighbour graph. `oggetti` lists the products (by `img_path` of their reference image), and `grafo` gives each product's n/ne/e/se/s/sw/w/nw neighbour index (-1 = none).
- `Training/<Category>/<Sub>/N.jpg`: 7,513 studio reference images, one per product. A test box's (`class`, `label`) maps to `Training/<classes[class]>/<label>.jpg`, where `classes` comes from `Training/TrainingClassesIndex.mat` (`classes`/`indices`, 1-based; read with `scipy.io.loadmat(..., simplify_cells=True)`). Some files use `.JPG`, so match the extension case-insensitively. 2,428 of 2,430 labelled boxes resolve.
- `Training/Background/`: 71 shelf and price-tag images with no products.
- `planos/`, `bb/`: planograms and absolute-pixel box polygons for a 123-image subset (`s<store>_<image>`), from Tonioni et al.'s planogram-compliance work.

## price_tags/

Parquet with columns `image_id`, `image.bytes`, `width`, `height`, `objects.bbox` (`[y_min, x_min, y_max, x_max]`, relative), `objects.category_name`.

## Not used

- `grocery_dataset/` (Varol & Kuzu, 2014, https://github.com/gulvarol/grocerydataset): entirely cigarette-shelf photos with graphic tobacco health warnings, and 79% of boxes are "other brand". Unsuitable for this project.
- `D-Shatnev/price-tag-detection` (Hugging Face): mostly motion-blurred video frames from non-grocery stores.
- SKU-110K in full (~12 GB): single class `object`, no product IDs. Only the 8 `test_images/` are used.

Later phases: Roboflow empty-shelf sets (gaps).

## Label OCR

`shelfsense.pricing.label_ocr.read_label_text()` needs the `tesseract`
binary (`sudo apt install tesseract-ocr`), not just `pip install
pytesseract` (that's just the Python wrapper, see pyproject.toml's `ocr`
extra). It isn't installed in every dev environment, so
`tests/test_label_ocr.py`'s OCR-against-real-images tests skip
gracefully without it; `parse_price()`'s text-parsing tests don't need
it and always run.
