# Demo script (about 6 minutes)

Presenter notes. One presenter, one web page: Home, then a shelf, then "Check any shelf photo".

## Before you go on stage

```bash
cd ~/confluentia_ml
PYTHONPATH=src .venv/bin/streamlit run app/dashboard.py     # http://localhost:8501
```

- Tesseract must be on PATH for live price reading (`sudo apt install tesseract-ocr`); EasyOCR's models must already be in `~/.EasyOCR/model` (`python scripts/download_data.py easyocr`); nothing downloads during the demo.
- Open the app, open a shelf and press **Check the shelf now** once, so the models are loaded (the first check takes a few seconds; later ones about 2 s). Then press F5 to start clean on Home.
- Browser: full screen (F11), zoom 90 to 100%. Close other tabs (a second tab is fine, but it shares the models).
- Have 2-3 fresh shop photos on the desktop for "Check any shelf photo", **already tried once**. Good photos: shelves of packaged goods, whole bay in view, 1-2 m, straight on.
- Fallback if a live check fails: the **Show the saved check** switch shows the saved result of the same photo, which the live check reproduces exactly. Nothing needs the internet.

## 1. The problem, in one breath (30 s)

"A store has a planned layout, a planogram, and a price list. Staff walk the aisle checking by eye: right product, right place, right count, right price, nothing empty. We do that from one photo, on a small computer at the shelf, and tell staff what to fix first."

## 2. Home: the whole store at a glance (30 s)

**Today's shelf check**, plan **Current plan**: three shelves, scores 100, 94 and 93, CHF 4.80 at stake. "These are real supermarket photos from a store our models never saw, checked against the plan they were stocked to."

Click **New plan from head office**: the rings turn red (28, 33, 27), **CHF 239.82** at stake, 18 to-dos. "Overnight head office changed the plan: two products swapped, one product the shelves do not carry, a promotion that needs more space, and a price rise. Nobody has touched the shelves yet."

## 3. One shelf, checked live (2 min)

**Open shelf** on **Oil & vinegar** (store 4, photo 9).

1. Photo on the left, the plan on the right, still grey. Press **Check the shelf now**: each step ticks off with its time (found the products, recognised each one, matched them to the plan, counted the packs, looked for empty space, read the price labels), about 2 s in total.
2. Score **28**, **CHF 93.98 at stake**, **4 to-dos**. The photo and the plan show the same slots in the same colours: red "wrong product", orange "low stock". The price label is tagged red **2.20 ≠ 5.57**.
3. **To-do list**: the swap is one to-do, "Swap Oil-Vinegar #53 with Oil-Vinegar #50: each belongs in the other's slot", although the report has four findings for it (Details tab). Press **Show on photo** on it: everything else fades, and the two slots stay bright on the photo and on the plan.
4. Switch to **Current plan** and **Check the shelf now** again: **100**, every bottle green, "Nothing to do".
5. **Prices** tab on **Canned food** (New plan): the planted rise, "label reads 2.70, price list 2.97", tagged *wrong price*. On **Breakfast cereals**, a label read as 5.50 but unsure is tagged *check by eye*. "We only call a price wrong when the text readers are confident; otherwise we ask a person."
6. **Shift summary** tab: two lines for the shift lead, written by Qwen 2.5 (1.5B) on this laptop; the chip under it says whether the checker accepted it. If it was rejected, say so: that is the checker doing its job.

## 4. Let a judge plant the violation (90 s, the strongest part)

**Breakfast cereals**, plan **Try your own plan**: it starts at **94**, the real plan. "Those changes were ours. Now you pick." A judge swaps two slots (for example S1-01 with S1-02: the score drops to 54, "−40 vs current plan" under the ring, and the to-do "Swap Cereals #88 with Cereals #138" appears), changes what a slot should hold in the table, or raises *facings planned* and *low stock below* together (a promotion reset). Be clear: the photo is read once; only the plan changes.

## 5. Any photo, no plan (60 s)

**← All shelves**, then **Check a photo**.

1. Sample **Coffee (US)** (selected first): fill 88%, 5 empty spaces, about 11 packs to refill, row S5 flagged **low stock** ("Shelf S5: empty space on the left, about 6 packs wide. Refill it."). Open **Sensitivity** and drag the empty-space slider: results recompute instantly.
2. **Upload a photo** (your own, or `data/test_images/shelf_02_toothpaste_uk.jpg`: 9 empty spaces): refill list and annotated photo, live.

## 6. Close: edge (30 s)

"The same audit runs headless on four CPU cores with no GPU: `scripts/audit.py` reads the photo and writes one JSON report, which is what a shelf camera would send to the store server." Quote the measured numbers in `docs/WRITEUP.md`, section "Edge fit".

## Questions you will be asked, and honest answers

| Question | Answer |
|---|---|
| How accurate is it? | Empty space against hand-labelled spots on 94 held-out photos: precision 0.66, recall 0.62, F1 0.64. Product naming: 81% when the plan limits the candidates, 49% with no hint. Price labels: on 31 hand-boxed Grozi labels, all found, 30 read correctly, none wrong. |
| Why does a label say "check"? | Of 9 whole labels on the demo shelves, 7 read confidently and right, none wrong. One is read right but unsure, one is motion-blurred; on blurred photos the fix is a steadier photo. Labels cut off by the photo's edge are skipped. The label detector never saw a Grozi shelf in training. We report an unsure read as "check by eye", never as a wrong price. |
| Why does Canned food show a wrong price even on "Current plan"? | The dataset gives the 340 g and 490 g tins of the same pineapple slices one product ID, so the price list can only hold one price (1.95) and the 490 g label (2.75) mismatches. In a real catalogue they are two products. |
| Are the prices real? | No public data has shelf photos with a price list. The demo price list takes each price from the label read confidently, and the planted plan raises one by 10%. So it tests reading and matching, not a real store's pricing errors. |
| Are the violations real? | No public dataset has real planogram violations, so a violation is a plan that disagrees with a real photo. We planted some, and on "Edit the plan yourself" you plant your own. We do not edit photos, because erased packs looked fake. |
| What is "value at stake"? | Missing packs times their price, plus each confident price gap times the slot's facings. An estimate for putting work in order; there are no sales rates in the data. |
| Did you use a small language model? | Yes, for one job: Qwen2.5-1.5B (4-bit, 1.1 GB, CPU) writes the two-sentence shift summary, and a checker rejects any summary with a wrong slot, problem or number. We tested it for deciding and wording the steps too: it told staff to restock two products that were only swapped, so rules decide the steps. |
| Why these shelves? Is it only good on three? | They are the best three of 35 held-out shelves (100, 98, 98). Honestly, the median shelf scores 53: look-alike products get confused and blocks misaligned, and three shelves fail outright. The planogram files for the others are a script away; we show the ones where every planted change is clearly visible. |
| It missed the empty space on the oil-vinegar shelf. | Yes: the empty-space detector was trained on small photos of other shops, and plain gaps between bottles are its weak spot. We tried inferring gaps from the space between bottles; it caught this one but made 253 false alarms on our labelled test photos, so we did not ship it. |
| Is the 98 inflated? | Partly: the correct plan's facings are our own pack counts on the same photo, so 98 checks naming and alignment, not counting. On a downsized copy of the photo the score moved by up to 17 points on small shelves, so we read the original. |
| What fails? | Lookalike products can be confused; small false empty spaces on gappy-looking products; blurry labels. Loose produce does not work; close-ups get a "retake" warning. |
| What did you try that did not work? | Larger input sizes, a texture check on gaps, test-time averaging, merging detectors, a pack detector retrained on twice the data (better mAP, worse gaps), and the language model deciding the steps. All measured, none shipped. Biggest lesson: our proxy metric said 0.63 and real labels said 0.27, so we switched to real labels. |
| Can it run on the edge? | Measured on four CPU cores with no GPU; see "Edge fit" in `docs/WRITEUP.md`. Real Jetson or phone timings are untested. |
| Why not a big vision-language model? | Small specialised models run at the shelf, cost nothing per image and can be measured. |
