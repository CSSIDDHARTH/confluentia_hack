# shelfSense AI

AI retail compliance agent for a hackathon (see `problem_statement.md`). It takes shelf images and reports planogram violations, price-label mismatches and stock-outs, with corrective actions and a compliance score.

Started as a two-person team; since 2026-10-22 all work happens in one workspace. Every phase must end with something runnable and visible; extend working code rather than building ahead.

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
  report/             Person B: compliance scoring, recommendations, report export, value at stake (impact.py), shift briefing
  advisor/            Person A: merges a report's violation lines into ordered steps (plan.py)
app/dashboard.py      The shelf audit: pick a real shelf and plan, "Run it live" (vision + price labels), score, value at stake,
                      steps for staff, SLM summary. Mock shelves and custom paths are behind the "Developer inputs" switch
app/pages/            Extra dashboard pages, shown in the sidebar automatically (Streamlit multipage):
                      1_Shelf_availability.py = wide-shelf mode: empty space, per-row fill, refill list, no planogram
                      2_What_to_do.py = the planogram report's violations merged into steps for staff; "Edit the plan yourself" lets a user change the plan and watch the score
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
10. The demo planograms (`data/demo_bays/<bay>/planogram_compliant.json`) are written from the human labels of held-out Grozi shelves, with facings taken from the pack counts on the photo (so a high score on the correct plan checks naming and alignment, not counting: the facings are our own counts). Each has a `planogram_scenario.json` with three planted planned changes: two products swapped, one slot expecting a product that is absent (taken from a different category), and one product's facings raised by two (a promotion reset the shelf has not caught up with). No public data has real planogram violations (assumption 7), and we do not edit photos: erasing packs by inpainting or flat fill looked fake.
11. Product identification is restricted to the planogram's own SKUs. Closed-set matching tends to confirm a lookalike product the plan expects, so planted absent products come from a different category. Measured on held-out stores, the whole chain (detect, name, align, count) scores 98, 88 and 86 on the compliant plan for the three best shelves of at least four blocks, and 80 or lower for the rest, so the demo bays are chosen by that ranking, not at random. Scores are sensitive to input resolution on these small shelves: reading a 1600 px copy instead of the 3264 px original changed 5 of 10 scores by up to 17 points, so the live run reads the original. A stress test with 20 random extra candidate products per bay scored lower and its results depended on which products were drawn.
12. Availability (phase 4) uses a second, pack-level detector (YOLO11s trained on an 800-image SKU-110K subset, mAP50 0.89 on 200 held-out images). An empty space is a stretch of a shelf row that no pack covers but that is at least 0.8 of that row's median pack width; missing facings = stretch width / median pack width. A row is low stock when packs cover less than 60% of the bay's width. That geometry is now the fallback (the planogram demo shelves in `data/demo_bays/` also use the void detector). Empty spaces come from a third detector (`models/void_detector.pt`, YOLO11s started from the pack detector and trained on the 727 train photos of the Roboflow Universe "Empty shelf detection" set, CC BY 4.0, 416 px, one class `Void`, in `data/empty_shelf/`): `scripts/eval_voids.py` measures precision 0.66, recall 0.62, F1 0.64 against the hand-labelled spots on 94 held-out test photos (IoU 0.3; confidence 0.3 chosen on the valid split; 50 near-duplicates of SKU-110K images removed). The geometry method scores F1 0.27 on the same labels, and our earlier 0.63 for it compared it against gaps derived from pack boxes, which overstated it. A void's missing packs = its width / its row's median pack width; with the void detector a row's fill is the share of bay width not covered by voids, and the row is low stock below 60%; this fill is not measured against labels. Tried and rejected, with measured results: larger detection image sizes (F1 0.59-0.63 vs 0.67 at 1024 on the same half), a texture check to drop false gaps (no F1 gain), and retraining the pack detector on 2x the data (mAP50 0.897 vs 0.888 but gap F1 0.56 vs 0.63, against pack-derived gaps). Gaps are only as good as the pack detector, so missed packs show up as false gaps.
13. The price-label detector (YOLOv8n) is trained only on Open Food Facts price tags (`data/price_tags/`), with no Grozi shelves in training. It was tested on 31 hand-boxed Grozi labels: all 31 found, 30 prices read correctly, 0 wrong, 1 unreadable (Tesseract alone: 27 correct, 2 wrong); with Tesseract 5.3.4 on the dev laptop, 29 correct and 0 wrong. Seven Tesseract variants vote; the last two sharpen the original pixels strongly first (26 instead of 25 of the 31 read confidently and right; on the demo shelves' 15 whole labels 9 right and 0 wrong instead of 8 and 1). A detected box that touches the photo's edge and is not label-shaped (width/height outside 1.5-4.5; the 31 hand-boxed labels are 1.8-3.6) is a label the photo cuts off, or not a label, and is skipped instead of reported as unreadable. Each Tesseract process runs with one thread (`OMP_THREAD_LIMIT=1`; otherwise five side by side on four cores took 87 s instead of 0.4 s), and without a GPU EasyOCR reads once instead of also at 2x (same 29 correct, 0 wrong, 3x faster on four cores). The whole audit with prices takes a median 2.2 s on the laptop GPU and 4.8 s on four CPU cores (`scripts/benchmark_chain.py --prices`). Prices are voted on by Tesseract and, if installed, EasyOCR (`pip install -e ".[ocr-easy]"`, models fetched once with `python scripts/download_data.py easyocr`; it never downloads at run time, so an offline demo cannot hang); EasyOCR returns nothing on the blurriest labels. A label price that differs from the catalogue on an OCR confidence below 0.5 is not reported as a price mismatch: it counts as an unreadable price ("check label by eye", weight 2), so a shaky read cannot cost a mismatch's 5 points or claim money at stake.
14. A label is tied to a slot by position: it goes on the same shelf, to the slot whose first facing is closest in x, with at most one label per slot. A label not near any slot's left edge but under a free slot's detected products goes to that slot (a slot that starts at the photo's edge has its real left edge out of frame). A slot's span comes from the vision stage's items and gaps in it, so a slot with nothing detected cannot receive a label.
15. Demo-bay prices: Grozi has no catalogue prices, so each demo bay's `catalogue.csv` takes its prices from the labels read with OCR confidence of at least 0.5 (`scripts/add_demo_labels.py`), and `catalogue_scenario.csv` raises one of them by 10% as the planted price mismatch. Labels that cannot be read are still reported as unreadable, so the compliant plan does not score 100. Of the 24 boxes the detector finds on the five bays, 9 are labels cut off by the photo's edge (or not labels) and are skipped. Of the 15 whole labels, 8 read confidently and right (7 of them tied to a slot; the other sits under a misplaced product), 1 reads right but unsure (store1_47), and 6 are motion-blurred (all 5 on store1_44, 1 on store1_47): a person can read them, but no OCR setting we tried could (downscaling, sharpening, binarising), so they are reported as unreadable. Labels are tied to slots separately in each detection file, against that file's own reading and plan, exactly as the live run does: on all 10 bay/plan pairs the live run (vision and prices) reproduces the saved score and every label's price and slot. With the planted plan, store1_36 shows the planted 10% price rise (label 2.70, catalogue 2.97). On store4_2 and store4_9 the only confidently read label is in a slot the planted plan swaps, so the firm mismatch there is the swap's: the old product's price under the new product, a real product-label mismatch (staff must move or reprint the label), and their planted price rise is not visible.
16. Shift briefing (`report/briefing.py`): the task list is built in code (shelves top first; within a shelf stock-out, price mismatch, misplaced, low stock, then label and recognition checks) and every step is a violation's own recommended action. A small language model (Qwen2.5-1.5B-Instruct, 4-bit GGUF, 1.1 GB, CPU via llama.cpp; `python scripts/download_data.py slm`, `pip install -e ".[slm]"`) writes only a 2-sentence summary of the three most urgent problems. It is not allowed to restate prices or counts, because when asked to it swapped them. Its summary is checked: only those three slots, each with the problem it really has, no other numbers, no "compliant", not cut off; otherwise the template summary is shown. On the 13 dashboard scenarios (`scripts/eval_briefing.py`) 8 to 12 summaries pass depending on sampling, every rejection read by hand was a real error, and a summary takes about 5 s on this laptop's CPU when nothing else is running.
17. Value at stake (`report/impact.py`) is an estimate for putting work in order, not accounting: a stock-out or low stock is priced at its missing packs (planned facings minus matching packs seen) times the catalogue price, and a firm price mismatch at the price gap times the slot's planned facings. Misplaced, unrecognised and unreadable items are not priced. No sales rates are used (Grozi has none and the prices come from assumption 15).

## Working agreements

- **Ownership:** the Person A / Person B split above (vision vs pricing, report and app) describes who built what. Since 2026-10-22 all work happens in one workspace, so any folder may be edited.
- **Shared files** (`contracts/`, `src/shelfsense/contracts.py`): update the example file and the model together, and keep `pytest` green.
- **Branches:** work on `pai`, merged into `main` when a phase works. `main` must always run.
- **Evaluation:** every phase is checked on the same images in `data/test_images/`.
- **Commits:** commit each working phase so there is always a demo-able version to fall back to.
