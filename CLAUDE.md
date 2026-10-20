# shelfSense AI

AI retail compliance agent for a hackathon (see `problem_statement.md`). It takes shelf images and reports planogram violations, price-label mismatches and stock-outs, with corrective actions and a compliance score.

Two-person team. Every phase must end with something runnable and visible; extend working code rather than building ahead.

## Layout

```
contracts/            Example files = source of truth for the data contracts
  planogram.example.json
  catalogue.example.csv
  detection_output.example.json   A -> B handoff; also the mock for one image
src/shelfsense/
  contracts.py        Pydantic models + loaders mirroring contracts/ (shared)
  vision/             Person A: detection, SKU recognition, shelf/slot assignment, gaps
  pricing/            Person B: label detection, OCR, price parsing
  report/             Person B: compliance scoring, recommendations, report export
app/dashboard.py      Person B: Streamlit dashboard (planogram view)
app/pages/            Extra dashboard pages, shown in the sidebar automatically (Streamlit multipage):
                      1_Shelf_availability.py = wide-shelf mode: empty space, per-row fill, refill list, no planogram
scripts/              Data download, mock generation, utilities
tests/                pytest; test_contracts.py checks the contracts agree with each other
data/                 Git-ignored except data/mock/; fetch with `python scripts/download_data.py all` (see data/README.md)
```

## Setup and run

```bash
# Use your own Python env if it already has the packages, or make a venv:
#   python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # contracts, dashboard, tests
pip install -e ".[vision]"         # Person A only: ultralytics + opencv (skip if already installed)
python scripts/make_mock_image.py     # renders data/mock/mock_shelf_001.png
pytest                                 # must pass before every merge
streamlit run app/dashboard.py
```

Dashboard with live photo upload (the Shelf availability page runs the detector on an uploaded photo when `ultralytics` and `models/pack_detector.pt` are present). If your system Python has torch and ultralytics but not Streamlit, add Streamlit in a venv that reuses them, so nothing in your system packages changes:

```bash
python3 -m venv --system-site-packages .venv
PYTHONPATH= .venv/bin/pip install streamlit "pyarrow<18"   # pyarrow 18+ needs NumPy 2
PYTHONPATH=src .venv/bin/streamlit run app/dashboard.py     # then open the Shelf availability page
```

If ROS is sourced in your shell, its pytest plugins crash on load. Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest` or `unset PYTHONPATH` first.

## Data contracts

Always load and save through `shelfsense.contracts`. Don't parse the JSON by hand.

- **bbox** is `[x1, y1, x2, y2]` in absolute pixels of the original image, with the origin top-left.
- **Shelf `index` 0** is the top shelf. **Slot `position` 0** is the leftmost slot. `slot_id` is unique across a planogram.
- **Planogram:** shelves → slots, and each slot has an expected `sku`, `facings` and `min_facings`. It is ordinal (order and counts only), with no physical measurements.
- **Catalogue:** CSV keyed by `sku` (name, brand, category, size, price, currency).
- **Detection output:** one JSON per image. The vision stage fills `shelves`, `items` and `gaps`, and assigns each one a `shelf_id`/`slot_id`. The pricing stage fills `labels`. `sku: null` means detected but not recognised. `produced_by` records which stage and version wrote it.

## Assumptions

These must appear in the final submission. Add to the list as you go.

1. Input is a roughly front-facing photo of one shelf bay (fixture). Video is handled by sampling frames.
2. Each image is tagged with its planogram ID (in practice the store app would know which bay is being photographed).
3. Planograms are ordinal (shelf, slot order, facings), because the public planograms (Grozi, `data/grozi/**/N_plano.json`) only give relative neighbour positions.
4. Each slot has one shelf-edge label, below and left-aligned with the slot's first facing.
5. A slot is low-stock when matching facings < `min_facings`, and a stock-out when it has none.
6. Catalogue prices are the source of truth, and a label is a mismatch if its price differs from the catalogue price. Promotions are out of scope for now.
7. No public dataset has planograms with violations, real prices, or price-tag text. Violations are created by editing compliant shelves, prices are synthetic, and OCR accuracy is measured on hand-transcribed or synthetic tags.
8. Data: Grozi-3.2K (Swiss supermarket shelves with product boxes, reference images and planograms; academic use only) and Open Food Facts price tags (CC BY-SA 4.0). See `data/README.md`.
9. Grozi boxes cover a whole product block (all facings of a product), so facings come from a second, pack-level detector (see 12): each visible pack inside a block is one facing, and stacked packs count separately. Every pack becomes its own item carrying its block's SKU and slot, so the report's "items in a slot = facings" rule gives stock-outs and low stock. Misplaced and unrecognised items are reported once per slot and product (`compute_report(..., group_items=True)`), not once per pack.
10. The demo planograms (`data/demo_bays/<bay>/planogram_compliant.json`) are written from the human labels of held-out Grozi shelves, with facings taken from the pack counts on the photo. Each has a `planogram_scenario.json` with three planted planned changes: two products swapped, one slot expecting a product that is absent (taken from a different category), and one product's facings raised by two (a promotion reset the shelf has not caught up with). No public data has real planogram violations (assumption 7), and we do not edit photos: erasing packs by inpainting or flat fill looked fake.
11. Product identification is restricted to the planogram's own SKUs. Closed-set matching tends to confirm a lookalike product the plan expects, so planted absent products come from a different category. Measured on held-out stores, the whole chain (detect, name, align, count) scores 98, 88 and 86 on the compliant plan for the three best shelves of at least four blocks, and 80 or lower for the rest, so the demo bays are chosen by that ranking, not at random. A stress test with 20 random extra candidate products per bay scored lower and its results depended on which products were drawn.
12. Availability (phase 4) uses a second, pack-level detector (YOLO11s trained on an 800-image SKU-110K subset, mAP50 0.89 on 200 held-out images). An empty space is a stretch of a shelf row that no pack covers but that is at least 0.8 of that row's median pack width; missing facings = stretch width / median pack width. A row is low stock when packs cover less than 60% of the bay's width. These thresholds are untuned, and gap accuracy is not measured because no public data labels empty space; it is judged by eye on `data/test_images/`. Gaps are only as good as the pack detector, so missed packs show up as false gaps.

## Working agreements

- **Ownership:** Person A (vision) owns `src/shelfsense/vision/` and phases 1–4: detection, SKU recognition, shelf/slot assignment, gaps. Person B (pricing and report) owns `pricing/`, `report/` and `app/`, plus phases 5–6: label OCR, price checks, scoring, recommendations, dashboard. Get the owner's agreement before editing their folder.
- **Before Person A's real output exists**, Person B builds against `contracts/detection_output.example.json`.
- **Shared files** (`contracts/`, `src/shelfsense/contracts.py`, `CLAUDE.md`): tell your teammate before changing them, update the example file and the model together, and keep `pytest` green.
- **Branches:** one branch per person (`a/<topic>`, `b/<topic>`), merged into `main` at the end of every phase. `main` must always run.
- **Evaluation:** every phase is checked on the same images in `data/test_images/`.
- **Commits:** commit each working phase so there is always a demo-able version to fall back to.
