# data/

Everything here is git-ignored except `data/mock/`, `data/test_images/`
and this file.

| Folder | Contents | Source |
|---|---|---|
| `mock/` | Synthetic shelf images matching their `.json` detection outputs (and `contracts/detection_output.example.json`) | `python scripts/make_mock_image.py`, `python scripts/gen_scenario.py` |
| `test_images/` | The fixed 8 images every phase is evaluated on. Real shelf photos, no annotations (detection/vision testing only -- see `test_images/README.md`). | SKU-110K, see below |
| `grozi/` | Not fetched. Grocery Products / Grozi-3.2K + GP-180 planograms -- Drive-hosted, not a plain download | https://github.com/tobiagru/ObjectDetectionGroceryProducts, https://alessiotonioni.github.io/publication/planogram |

**`grocery_dataset/` (Varol & Kuzu, 2014,
https://github.com/gulvarol/grocerydataset) is not used.** Despite the
name, it's entirely cigarette-shelf photos with graphic tobacco health
warnings -- unsuitable for this project. Raise with the team before
relying on it; CLAUDE.md's assumption 8 should probably be corrected.

Later phases: Roboflow empty-shelf sets (gaps), Kaggle / Roboflow
price-tag sets (OCR).
