# data/test_images/

The fixed set every phase is evaluated on (per CLAUDE.md's working
agreements). 8 real shelf photos, hand-picked for being roughly
front-facing single-bay shots (assumption 1), with visible shelf-edge
labels where possible, and reasonable product density.

## Source

[SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) (Goldman, Herzig,
Eisenschtat, Goldberger & Hassner, *"Precise Detection in Densely Packed
Scenes"*, CVPR 2019) — already named in CLAUDE.md assumption 8.

> The dataset is provided for the exclusive use by the recipient and
> solely for academic and non-commercial purposes.

The full archive is ~12GB (`SKU110K_fixed.tar.gz`,
`http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz`).
These 8 were picked from a ~44MB streamed subset of its `test/` split
(stopped early via HTTP range + streaming `tar`, rather than pulling the
full archive) and renamed from `test_<n>.jpg` to describe their content.
No annotations are included — SKU-110K ships bounding boxes only, no
SKU/price labels, so these are for vision/detection testing, not full
pipeline (planogram/catalogue) testing; they don't carry a
`detection_output.json`.

## Note on assumption 8's other dataset

The "Grocery Dataset" also named in assumption 8
(`gulvarol/grocerydataset`) turned out, on inspection, to be entirely
cigarette-shelf photos carrying graphic Turkish tobacco health warnings
(diseased organs, fetuses, etc.) — not food groceries, and not
appropriate for this demo. None of it was kept. Flagging so the
assumption gets corrected/discussed with the team.

## Files

| File | Store / locale | Category |
|---|---|---|
| `shelf_01_health_beauty_uk.jpg` | UK | Toothpaste / deodorant |
| `shelf_02_toothpaste_uk.jpg` | UK | Toothpaste |
| `shelf_03_deodorant_us.jpg` | US | Deodorant |
| `shelf_04_babycare_uk.jpg` | UK | Baby care |
| `shelf_05_footcare_de.jpg` | Germany | Foot care (Fußpflege) |
| `shelf_06_coffee_us.jpg` | US | Coffee |
| `shelf_07_skincare_cn.jpg` | China | Men's skincare |
| `shelf_08_tampons_uk.jpg` | UK | Feminine hygiene |
