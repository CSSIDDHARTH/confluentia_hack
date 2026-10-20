# data/

Everything here is git-ignored except `data/mock/` and this file.

| Folder | Contents | Source |
|---|---|---|
| `mock/` | Synthetic shelf image matching `contracts/detection_output.example.json` | `python scripts/make_mock_image.py` |
| `grocery_dataset/` | Grocery Dataset (Varol & Kuzu, 2014): 354 shelf images, boxes and brands, ~3 GB. Research use only. | https://github.com/gulvarol/grocerydataset |
| `grozi/` | Grocery Products / Grozi-3.2K: 680 shelf images plus studio reference images; GP-180 planograms | https://github.com/tobiagru/ObjectDetectionGroceryProducts, https://alessiotonioni.github.io/publication/planogram |
| `test_images/` | The fixed 5–10 images every phase is evaluated on | Picked from the above |

Later phases: SKU-110K (detector), Roboflow empty-shelf sets (gaps), Kaggle / Roboflow price-tag sets (OCR).
