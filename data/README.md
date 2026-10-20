# data/

Everything here is git-ignored except `data/mock/` and this file. Fetch with `python scripts/download_data.py all`.

| Folder | Contents | Used for | Licence |
|---|---|---|---|
| `mock/` | Synthetic shelf matching `contracts/detection_output.example.json` (`python scripts/make_mock_image.py`) | Dashboard development | — |
| `grozi/` | **Grocery Products / Grozi-3.2K.** Real Swiss supermarket shelves (Migros/Coop) with visible shelf-edge price labels. | Phases 1–5 | Academic research only |
| `price_tags/` | **Open Food Facts price-tag detection.** 2,229 real shop photos (1,982 train / 247 val), 1 class `price-tag`. Mix of full shelves and close-ups. | Phase 5 (label detector) | CC BY-SA 4.0 |

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

- Grocery Dataset (Varol & Kuzu): every shelf is a tobacco gantry, 79% of boxes are "other brand".
- `D-Shatnev/price-tag-detection`: mostly motion-blurred video frames from non-grocery stores.
