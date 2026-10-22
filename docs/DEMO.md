# Demo script (about 6 minutes)

Presenter notes. One presenter, three pages, in this order.

## Before you go on stage

```bash
cd ~/confluentia_ml
PYTHONPATH=src .venv/bin/streamlit run app/dashboard.py     # http://localhost:8501
```

- Tesseract must be on PATH for live price reading (`sudo apt install tesseract-ocr`); without it the audit page says so and reads placement only. EasyOCR's models must already be in `~/.EasyOCR/model` (`python scripts/download_data.py easyocr`); it never downloads during the demo.
- Open the app and press **Run it live** once on each page, so the models are loaded (the first press takes a few seconds; later ones about 2 s with prices, a third of a second without).
- Have 2-3 fresh shop photos on the desktop for the availability page, **already tried once**. Good photos: shelves of packaged goods, whole bay in view, 1-2 m, straight on. Avoid fruit and vegetables, close-ups, and price tags alone (the page then shows a "retake" warning).
- Fallback if the live run fails: every page shows the saved reading of the same photo, which the live run reproduces exactly. Nothing needs the internet.

## 1. The problem, in one breath (30 s)

"A store has a planned layout, a planogram, and a price list. Staff walk the aisle checking by eye: right product, right place, right count, right price, nothing empty. We do that from one photo, on a small computer at the shelf, and tell staff what to fix first."

## 2. The shelf audit (2 min)

Main page (**dashboard**), shelf `store4_9`, plan "Store changed the plan (planted)" (the default).

Say: "Real supermarket photo from a store the models never trained on. The store changed its plan: two products swapped, one product the shelf does not have, a promotion that needs two more facings, and one price raised."

1. Press **Run it live**. It reads the photo from scratch: product blocks, naming, alignment with the plan, packs, empty spaces and the price labels, with the seconds per stage. The score stays **43**: the live run reproduces the saved result on all ten shelf/plan pairs.
2. Point at the metrics: score 43, **CHF 37.84 at stake** (missing packs times price, plus price gaps; an estimate, no sales data), 5 steps for staff.
3. Read the first step: "Move Oil-Vinegar #50 out of S1-02 ... then restock Oil-Vinegar #53". Each step says how much money it puts back on the shelf.
4. Price: "Reprint label label_000: shows 2.20, catalogue price is 3.04." The swap moved a product under another product's label. On `store1_47` show "Check label label_001 by eye": OCR read 5.50 but was not sure, so it **does not** claim a mismatch. Say: "We only call a price wrong when the OCR is confident; otherwise we ask a human."
5. Shift briefing: the summary for the shift lead is written by a 1.5B-parameter language model on the CPU and checked against the report; if it names a wrong slot or number, the template is shown instead, and the page says which one you are reading.

For the planted 10% price rise itself, switch to `store1_36`: "Reprint label label_000: shows 2.70, catalogue price is 2.97."

## 3. Let a judge plant the violation (90 s, the strongest part)

Sidebar, **What to do**, shelf `store1_47`. This page scores placement and facings only.

1. Plan "Matches the shelf": **98**. Plan "Store changed the plan (planted)": **31**; the report's 8 violation lines become 4 steps, the swap is one step ("Swap Cereals #88 with Cereals #138"). Open the expander to show the 8 lines it replaced.
2. Choose **Edit the plan yourself**: "Those changes were ours. Now you pick." A judge swaps two slots, changes a slot's expected product, or raises *facings planned* and *minimum* together (a promotion reset). Score and steps update at once. Be clear: the photo is read once; only the plan changes.

Other shelves on this page (correct plan, then planted): store4_9 88 then 48, store1_36 93 then 41. Do not use store1_44 or store4_2 here; they score 80 on the correct plan.

## 4. Empty space on any photo (60 s)

Sidebar, **Shelf availability**. No planogram needed.

1. Saved shelf `shelf_02_toothpaste_uk`: 9 empty spaces, 16 packs to refill, fill 89%. Drag the "Empty-space confidence" slider: results recompute instantly.
2. **Upload a fresh photo**: refill list and annotated image, live. Download the CSV.

## 5. Close: edge (30 s)

"The same audit runs headless on four CPU cores with no GPU: `scripts/audit.py` reads the photo and writes one JSON report, which is what a shelf camera would send to the store server." Quote the measured numbers in `docs/WRITEUP.md`, section "Edge fit".

## Questions you will be asked, and honest answers

| Question | Answer |
|---|---|
| How accurate is it? | Empty space against hand-labelled spots on 94 held-out photos: precision 0.66, recall 0.62, F1 0.64. Product naming: 81% when the plan limits the candidates, 49% with no hint. Price labels: on 31 hand-boxed Grozi labels, all found, 30 read correctly, none wrong. |
| Why do most demo labels say "check by eye"? | Of 15 whole labels on the demo shelves, 8 read confidently and right, none wrong. 6 are motion-blurred (store1_44): readable by a person, not by OCR; the fix is a steadier photo. Labels cut off by the photo's edge are skipped. The label detector never saw a Grozi shelf in training. We report an unsure read as "check by eye", never as a wrong price. |
| Are the prices real? | No public data has shelf photos with a price list. The demo price list takes each price from the label read confidently, and the planted plan raises one by 10%. So it tests reading and matching, not a real store's pricing errors. |
| Are the violations real? | No public dataset has real planogram violations, so a violation is a plan that disagrees with a real photo. We planted some, and on "Edit the plan yourself" you plant your own. We do not edit photos, because erased packs looked fake. |
| What is "value at stake"? | Missing packs times their price, plus each confident price gap times the slot's facings. An estimate for putting work in order; there are no sales rates in the data. |
| Did you use a small language model? | Yes, for one job: Qwen2.5-1.5B (4-bit, 1.1 GB, CPU) writes the two-sentence shift summary, and a checker rejects any summary with a wrong slot, problem or number. We tested it for deciding and wording the steps too: it told staff to restock two products that were only swapped, so rules decide the steps. |
| Why these shelves? Is it only good on five? | They are the best five of 35 held-out shelves (98, 93, 88, 80, 80). Honestly, the median shelf scores 55: look-alike products get confused and blocks misaligned, and three shelves fail outright. The planogram files for the others are a script away; we show the ones where every planted change is clearly visible. |
| It missed the empty space on the oil-vinegar shelf. | Yes: the empty-space detector was trained on small photos of other shops, and plain gaps between bottles are its weak spot. We tried inferring gaps from the space between bottles; it caught this one but made 253 false alarms on our labelled test photos, so we did not ship it. |
| Is the 98 inflated? | Partly: the correct plan's facings are our own pack counts on the same photo, so 98 checks naming and alignment, not counting. On a downsized copy of the photo the score moved by up to 17 points on small shelves, so we read the original. |
| What fails? | Lookalike products can be confused; small false empty spaces on gappy-looking products; blurry labels. Loose produce does not work; close-ups get a "retake" warning. |
| What did you try that did not work? | Larger input sizes, a texture check on gaps, test-time averaging, merging detectors, a pack detector retrained on twice the data (better mAP, worse gaps), and the language model deciding the steps. All measured, none shipped. Biggest lesson: our proxy metric said 0.63 and real labels said 0.27, so we switched to real labels. |
| Can it run on the edge? | Measured on four CPU cores with no GPU; see "Edge fit" in `docs/WRITEUP.md`. Real Jetson or phone timings are untested. |
| Why not a big vision-language model? | Small specialised models run at the shelf, cost nothing per image and can be measured. |
