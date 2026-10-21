# Handoff: vision (Person A) to pricing and report (Person B)

Everything is on branch `pai`. Nothing here needs your code to change except the two small edits listed first, which are already made on `pai`: please read them before merging.

## Two edits I made in your files

1. `src/shelfsense/report/compliance.py`: `_slot_violations` and `compute_report` take a new argument `group_items: bool = False`. Default behaviour is unchanged and your tests pass unchanged.
2. `app/dashboard.py` line 111: `compute_report(det, planogram, catalogue, group_items=True)`. Your three mock presets score exactly as before (73, 66, 98).

Why: vision now emits one item per **pack** (each carries its block's SKU and slot), so a misplaced block of 8 packs produced 8 violations and the demo scenario scored 0. With `group_items=True` misplaced and unrecognised items are reported once per slot and product, and the scenario scores 31.

## What the detection output now looks like

- `items`: one per visible pack. `sku`, `sku_confidence`, `shelf_id`, `slot_id` are filled (sku is null below 0.5 confidence).
- Items in a slot = facings, so low stock and stock-outs follow from your existing rule.
- `gaps`: empty spaces with `estimated_missing_facings` and the slot they sit in.
- `labels` is **empty**: that is yours. The five demo bays in `data/demo_bays/` are real Grozi shelf photos with visible price labels, so running your label detector and OCR over `data/demo_bays/<bay>/photo.jpg` and filling `labels` would let the dashboard show price mismatches on the same shelves as the planogram demo. Catalogue prices there are synthetic (`data/demo_bays/<bay>/catalogue.csv`).

## New things you can show

- `app/pages/1_Shelf_availability.py`: wide-shelf mode, no planogram. Saved shelves or live photo upload; sliders recompute instantly. If it shows a "retake" warning, the photo had fewer than 8 packs.
- Demo shelves for the planogram view: in the dashboard, Advanced: custom paths, use `data/demo_bays/store1_47/detection_compliant.json`, `planogram_compliant.json`, `catalogue.csv` (score 98) and the `_scenario` versions (score 31). Full run order and likely judge questions: `docs/DEMO.md`.

## Running it

```bash
PYTHONPATH=src .venv/bin/streamlit run app/dashboard.py     # live upload needs the .venv recipe in CLAUDE.md
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest    # 105 pass, 2 skipped without tesseract
```

## Please say the limits out loud

Prices are synthetic, violations are planted, demo shelves are the best held-out ones, and empty-space F1 is about 0.63 (0.6-0.7 depending on the images). They are all in `docs/WRITEUP.md`; the same list should appear in your part of the pitch.
