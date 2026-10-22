# shelfSense AI

**One photo of a shelf bay in; what is wrong with it, what it costs, and what to do first out.**

Store staff check shelves by eye against a planogram (the planned layout) and a price list: right product, right place, right count, right price, nothing empty. It is slow, it gets skipped, and every gap or wrong label costs sales or customer trust. shelfSense does that check from one photo, on a small computer next to the shelf, and gives a store associate a short ordered list of fixes and the shift lead a two-sentence summary.

## What it checks

| Brief asks for | What shelfSense does |
|---|---|
| Planogram compliance | Finds each product block, names it, aligns the shelf against the plan row by row, counts facings: misplaced, missing, swapped products and low stock, with where each misplaced product belongs |
| Labels and prices | Finds the shelf-edge price labels, reads them with two OCR engines voting, ties each label to its slot, compares with the price list. A wrong price is claimed only when the read is confident; otherwise "check by eye" |
| Availability | Finds empty shelf space with a detector trained on hand-labelled empty spots, estimates missing packs, flags low-stock rows. Works on any shop photo, no planogram needed |
| Report | Compliance score, every violation with its corrective action, money at stake, violations merged into a few steps (a swap is one step, not four), JSON/CSV export |
| Small language model | Qwen2.5-1.5B (4-bit, CPU) writes only the shift summary; a checker rejects any summary with a wrong slot, problem or number |
| Edge | The whole audit runs headless on four CPU cores with no GPU (`scripts/audit.py`), measured below |

## How it works

```
photo ─► product blocks ─► name each block ─► align to plan ─► count packs ─► empty space ─► price labels ─► report ─► steps + summary
         YOLO11s (Grozi)   DINOv2 + adapter   Needleman-Wunsch  YOLO11s        YOLO11s          YOLOv8n +        score,      rules decide,
                           (plan's products)  (per shelf row)   (SKU-110K)     (empty-shelf     Tesseract +      actions,    SLM writes the
                                                                               photos)          EasyOCR vote     value at    summary, a
                                                                                                                 stake       checker verifies
```

Code decides every violation and step; the language model only writes, and its writing is checked. Details and every measurement: [docs/WRITEUP.md](docs/WRITEUP.md).

## Results (held-out data)

| What | Result |
|---|---|
| Whole chain on all 35 held-out shelves, correct plan (placement) | median 55; best three 98, 93, 88 (our demo shelves are the top five) |
| Demo shelves with planted plan changes | 31 to 49, with every planted change listed |
| Product naming, plan-restricted / open set | 81% / 49% top-1 |
| Pack detector | mAP50 0.89 on 200 held-out SKU-110K images; counting with bottle pieces merged: F1 0.845, median count error 5.7% |
| Empty space vs 639 hand-labelled spots, 94 photos | precision 0.66, recall 0.62, F1 0.64 (our first method: F1 0.27) |
| Price labels, 31 hand-boxed Grozi labels | 31 found, 30 read correctly, 0 wrong, 1 unreadable (29 with this laptop's Tesseract 5.3.4) |
| Price labels on the demo shelves | 15 whole labels: 8 read confidently and right, 0 wrong; 6 motion-blurred |
| Live run vs saved results, 5 shelves x 2 plans | identical scores, identical price and slot for every label |
| Shift summary by the SLM, 13 scenarios | 8 to 12 pass the checker (depends on sampling); every rejection read by hand was a real error |
| Speed, whole audit with prices (median, 3264 px photo) | 2.2 s on a laptop GPU; 4.8 s on 4 CPU cores with no GPU (vision 1.4 s, price labels 3.5 s) |

## Assumptions and limits

Stated in full in [CLAUDE.md](CLAUDE.md#assumptions) (17 assumptions) and [docs/WRITEUP.md](docs/WRITEUP.md#assumptions-and-limits-stated-not-hidden). The ones that matter most:

- One roughly front-facing photo per shelf bay, tagged with its planogram. Planograms are ordinal (order and facings, no measurements).
- **No public data has real planogram violations or shelf prices.** Violations are planted by changing the plan of a real photo (never by editing the photo), and on "Edit the plan yourself" a judge can plant their own. Demo prices come from the labels read confidently on the photo, with one raised by 10%.
- Demo shelves are the best five of 35 held-out shelves; the median shelf scores 55 (look-alike products, misaligned blocks). Plain gaps between bottles can be missed by the empty-space detector. Of their 15 whole price labels, 8 read confidently and right; 6 are motion-blurred and reported as unreadable (retake the photo). Labels cut off by the photo's edge are skipped.
- Value at stake is an estimate for ordering work (missing packs times price, price gaps times facings); there are no sales rates.
- Edge speed is measured on a laptop CPU capped at 4 cores; no Jetson or phone was available.
- Grozi-3.2K and SKU-110K are academic-use only; the empty-shelf photos are CC BY 4.0; Open Food Facts price tags are CC BY-SA 4.0.

## Run it

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate   # or your own env
pip install -e ".[dev,vision,ocr,ocr-easy,slm]" py7zr
sudo apt install tesseract-ocr                     # price reading
python scripts/download_data.py grozi easyocr slm  # shelf photos + product images, OCR models, language model
python scripts/build_sku_index.py                  # product index for naming
streamlit run app/dashboard.py                     # the shelf audit; pages: What to do, Shelf availability
```

Headless, as a shelf device would run it:

```bash
PYTHONPATH=src python3 scripts/audit.py data/demo_bays/store4_9 --plan scenario
CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 0-3 python3 scripts/audit.py data/demo_bays/store4_9 --plan scenario --cpu
```

Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest` (160 pass, 1 skipped with Tesseract and EasyOCR installed). Without the data and models, the mock shelves still run: switch on **Developer inputs** in the dashboard.

Demo script and likely questions: [docs/DEMO.md](docs/DEMO.md). Project layout, data contracts and working agreements: [CLAUDE.md](CLAUDE.md).
