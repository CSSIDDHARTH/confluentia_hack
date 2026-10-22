# shelfSense AI: how it works and what we measured

Written for judges. Covers the whole chain: photo to products, placement, counts, empty space, price labels, and the report staff act on.

## What it does

From one shelf photo it answers: is the right product in the right place, in the right quantity, and where is the shelf empty?

```
photo -> product-block detector -> product naming -> align to planogram -> count facings -> empty space -> report
         (YOLO11s, Grozi)          (DINOv2 + adapter)  (sequence alignment)  (pack detector)   (void detector)
                                                         pack detector: YOLO11s trained on SKU-110K
                                                         void detector: YOLO11s trained on 727 hand-labelled empty-shelf photos
```

1. **Detect** product blocks (one box per product, all its facings) with a YOLO11s detector trained on Grozi supermarket shelves.
2. **Name** each block by comparing its DINOv2 embedding with 8,133 studio reference images, restricted to the products the planogram expects. A small linear adapter, trained on shelf crops, closes the gap between studio and shelf photos. Low-confidence matches (below 0.5) are marked unknown for review.
3. **Align** the detected blocks to the planogram row by row with Needleman-Wunsch sequence alignment, so a swap, a missing product or an extra product shows up as a specific mismatch.
4. **Count** facings with a second YOLO11s detector that finds individual packs (trained on SKU-110K). Each pack becomes an item that carries its block's product and slot, so the report's rule "items in slot = facings" yields low stock and stock-outs.
5. **Find empty space** with a third YOLO11s detector trained directly on hand-labelled empty shelf spots. Rows still come from the vertical coverage of packs; each empty space goes to its nearest row, its width over that row's median pack width is the number of missing facings, and a row more than 40% covered by empty spaces is low stock. (Our first method inferred gaps from missing packs; it is kept as the fallback when the void weights are absent, and scored far lower, see below.)
6. **Read the price labels** with a YOLOv8n label detector (trained on Open Food Facts price tags), then vote on each price with seven Tesseract variants and EasyOCR. Boxes the photo's edge cuts off are skipped. Each label goes to the slot whose first facing it sits under. A label whose price differs from the catalogue is a price mismatch only when the OCR vote is confident (at least 0.5); otherwise staff are told to check it by eye.
7. **Report**: a compliance score with each violation and a corrective action, the money at stake (missing packs times price, price gap times facings: an estimate for ordering work), and the violations merged into a few steps for staff.

The demo does not depend on saved results: the dashboard has a **Check the shelf now** button that reads the photo from scratch (all three detectors, naming, alignment and the price labels) and reports the seconds each stage took. On the three demo shelves it reproduces all 6 saved scores exactly, and every label's price and slot.

A second mode, *shelf availability*, skips the planogram and gives empty spaces, per-row fill and a refill list for any shelf photo, uploaded live.

## From violation lines to steps

The compliance report lists one line per violation, so one swap of two products is four lines (a stock-out and a misplaced item in each slot). `advisor/plan.py` merges them with rules: a swap becomes one step naming both slots, a wrong product in a slot that lacks the right one becomes one "move out, then restock" step, a misplaced product is told where it belongs ("Cereals #138 belongs in S1-01", from the planogram, or "the plan has no other slot for it"), and steps are ordered by severity. Across the 6 demo reports (3 shelves, correct plan and planted changes; placement only) 24 violation lines become 14 steps, and each violation lands in exactly one step. The planted swap becomes one "Swap" step on store4_9 and store1_47; on store1_36 it stays as two moves.

The dashboard also has an **Edit the plan yourself** mode (`advisor/edit.py`): a person changes what a slot expects, its facings and minimum, or swaps two slots, and the score and steps recompute from the unchanged photo reading. This is how a violation gets created without us planting it. It tests the comparison, not the detector, since the photo is read once.

### Where the small language model is used, and where it is not

The brief invites a small language model (under 20B parameters) where it fits. The rule is: **code decides, the model writes, a checker verifies.** We tested Qwen2.5-1.5B and 0.5B, run locally, in three roles and kept one.

- **Deciding what to do** from the raw violation lines: every slot and number it wrote existed in the report (10 of 10 shelves), but it told staff to restock two products that were simply in each other's slots (the right instruction is a swap), named the wrong item to verify, and padded a one-violation shelf with generic advice. Its answers pass a "mentions only known slots and numbers" check, so that check does not catch wrong meaning.
- **Rewording the rule-based steps**, with every line checked to name the same slots and products and add no numbers: the 1.5B model kept 22 of 34 lines, took a median 21 s per shelf on 16 CPU cores (5 tokens/s) and does not fit on the 4 GB GPU next to the desktop; the 0.5B model (2 s on GPU) kept 2 of 34. Kept lines were still worse than the rules: "Move Cereals #138 from S3-01 to S3-01", "9 facing slots".

So the rules decide and phrase the steps, which is exact and instant.

- **Writing the shift summary** (shipped, `report/briefing.py`): Qwen2.5-1.5B-Instruct (4-bit GGUF, 1.1 GB, CPU through llama.cpp) writes two sentences about the three most urgent problems for the shift lead. It is not allowed to restate prices or counts, because when asked to it swapped them. A checker accepts the summary only if it names exactly those three slots, each with the problem it really has, adds no other numbers, never calls the shelf compliant and is not cut off; otherwise the template summary is shown, and the dashboard says which one you are reading. On the 13 dashboard scenarios (`scripts/eval_briefing.py`) 8 to 12 summaries pass depending on sampling, every rejection we read by hand was a real error, and a summary takes about 5 s on the laptop CPU.

## Measured results

All on held-out data (different stores or different images from training).

| Part | Result |
|---|---|
| Product-block detector | trained on stores 2, 3, 5; evaluated on stores 1, 4 |
| Pack detector | mAP50 0.89 on 200 held-out SKU-110K images |
| Naming, planogram-restricted | 0.81 top-1 |
| Naming with 20 extra distractors | 0.79 |
| Naming, open set (no planogram) | 0.49 |
| Whole chain (detect, name, align, count), correct plan, 35 held-out shelves | median 53; best three 100, 98, 98; three shelves score 0 |
| Planted-change shelves (the 3 demo shelves) | scores fall to 31 to 46 with every planted change listed (placement only; 27 to 33 with prices and label checks) |
| Pack counting, 200 held-out SKU-110K photos, 29,876 hand-boxed packs | F1 0.845, median count error 5.7% with bottle pieces merged (0.836 and 8.1% without) |
| Price labels, 31 hand-boxed labels on Grozi shelves | 31 found, 30 prices read correctly, 0 wrong, 1 unreadable (Tesseract alone: 27 correct, 2 wrong) |
| Price labels on the three demo shelves | 9 whole labels: 7 read confidently and right, 0 wrong, 1 unsure, 1 motion-blurred (on an earlier five-shelf set: 6 of 15 motion-blurred) |
| Live run vs saved results | all 6 shelf/plan pairs: same score, same price and slot for every label |
| **Empty space vs real hand-labelled empty spots**, 94 held-out photos, 639 labelled spots | precision 0.66, recall 0.62, **F1 0.64** (void detector, confidence 0.3 chosen on a separate split) |
| Same photos, our first method (gaps inferred from missing packs) | precision 0.29, recall 0.25, F1 0.27 |

## Assumptions and limits (stated, not hidden)

- **The 98 on the correct plan is partly circular.** That plan's facings are the pack counts of our own pack detector on the same photo, so a high score checks naming, plan alignment and slot assignment, not counting accuracy. Counting is judged only by the pack detector's mAP50 of 0.89 on SKU-110K.
- **Scores are sensitive to input resolution on small shelves.** Reading the stored 1600 px copy of a demo photo instead of the 3264 px original changed 5 of the 10 scores, by up to 17 points (`store1_44`: 80 to 63). These shelves have 4 to 6 product blocks, so one different detection moves the score by 7 to 15 points. The live page therefore reads the original photo when it is present.

- Front-facing photo of one shelf bay; each image is tagged with its planogram.
- Planograms are ordinal (shelf, slot order, facings); public planograms carry no physical measurements.
- **No public dataset has real planogram violations, real prices or price-tag text.** The violations are planted by editing the planogram of a real photo. The demo price list takes each price from the label read confidently on the photo, and the planted plan raises one by 10%; so a demo price mismatch tests reading and matching, not a real store's pricing error. We did not edit photos, because erased packs looked fake.
- **Some labels cannot be read.** On an earlier five-shelf demo set, 6 of 15 whole labels were motion-blurred: a person can read them, but no OCR setting we tried could (downscaling, sharpening, binarising); the fix is a steadier photo. The label detector never saw a Grozi shelf in training. An unreadable label costs 2 points and asks for a check by eye; it never claims a mismatch or money at stake.
- Demo shelves are the best five of 35 held-out shelves. The median shelf scores 55 on its correct plan: look-alike products get confused and blocks get misaligned, and on three shelves the chain fails outright (score 0).
- **Plain gaps between bottles can be missed.** The void detector was trained on 416 px photos of other shops; on store4_9 it rates a clear 1.5-bottle gap at 0.06. Our fix for it made things worse on the labelled photos (below), so it is not shipped.
- Use packaged-goods shelves photographed with the whole bay in view. Loose produce and close-ups are outside what the pack detector handles.
- The empty-space labels are 416 px photos from one Roboflow Universe project (single annotator style, many photos from the same shop types as SKU-110K). Accuracy on very different shops is not measured. Spots the annotators left unlabelled count as false alarms, so the true precision may be higher than 0.66.
- Datasets: Grozi-3.2K and SKU-110K are academic-use only; the empty-shelf set is Policenta "Empty shelf detection" on Roboflow Universe, CC BY 4.0.

## Robustness on photos we did not train on

24 real phone photos of shops (Open Food Facts price-tag set, random draw, never used to train the pack detector), run through the live upload path and judged by eye. No ground truth, so this is a qualitative check, not an accuracy number. The table below is for the first method (gaps inferred from missing packs); the void detector is described after it.

| Kind of photo | Count | What happened |
|---|---|---|
| Shelves of packaged goods, whole bay in view | 9 | Packs found and counted sensibly. 4 of the 9 also had a false empty space or low-stock flag, usually a price-label strip read as a shelf row or a missed pack. |
| Loose produce (fruit, vegetables) | 6 | Fruit boxed as "packs"; fill percentages are not meaningful. Out of scope for a pack detector. |
| Close-ups, price tags, too few packs (under 8) | 8 | Now flagged "looks like a close-up, retake from 1-2 m" and no refill list is given. Before the guard these produced false "low stock" alerts. |
| Cluttered shelf with overlapping products | 1 | False empty spaces over products the detector missed. |

A second guard drops a "row" of at most 3 flat boxes under much taller packs (a price-label strip). It removed the false low-stock row in one of the two photos where it occurred, left the other, and changed neither the empty-space F1 (0.63) nor any of the 8 test shelves.

**With the void detector** (rerun on the 9 packaged-goods photos, by eye): the five full shelves now show no empty spaces and no low-stock rows (the first method put false gaps or a false "low stock" row on 4 of the 9). The other four show 1 to 4 empty spaces; the bare shelf on one photo and a gap at the end of a row on another look right, a few small ones on a tea shelf look doubtful. The remaining mistakes are small boxes on products that look gappy.

The 8-pack guard separated every close-up from every usable shelf in this sample and does not trigger on any of the 8 test shelves (125-189 packs each). Its threshold was set after looking at these 24 photos, so treat it as a sanity check, not a tuned result.

## Empty space against real labels

Until we had real labels, empty-space accuracy was measured against gaps derived from the labelled packs (below), which gave F1 0.63. That compared one gap finder against another. When we got photos with empty spots labelled by hand (727 train / 206 valid / 112 test, one class `Void`), the same gap finder scored **F1 0.27**, so the earlier number overstated it badly. `scripts/eval_voids.py` evaluates on the valid and test splits after removing 50 photos that are near-duplicates (16x16 thumbnail similarity above 0.9) of SKU-110K images our pack detector may have trained on.

| Method (94 clean test photos, 639 labelled spots) | Precision | Recall | F1 |
|---|---|---|---|
| Gaps inferred from missing packs (first method) | 0.29 | 0.25 | 0.27 |
| Void detector, confidence 0.1 | 0.39 | 0.83 | 0.53 |
| Void detector, confidence 0.2 | 0.54 | 0.72 | 0.62 |
| **Void detector, confidence 0.3 (shipped)** | 0.66 | 0.62 | **0.64** |
| Void detector, confidence 0.4 | 0.78 | 0.48 | 0.59 |

A match needs IoU of at least 0.3. The confidence 0.3 was chosen on the valid split (also F1 0.64 there, 174 photos), not on the test split. The detector starts from our pack detector's weights (it already knows shelves) and trains in about 15 minutes on the laptop GPU. The dashboard's confidence slider moves along this table.

The planogram demo shelves use the same void detector for their empty spaces (regenerating them changed no compliance score, because scores do not use gaps). Row fill now comes from the voids too (the share of the bay's width not covered by empty spaces). Pack coverage alone read a cropped top shelf as "45% stocked". This part is not measured against labels; it is checked by eye below.

## Where the first method's errors came from (against pack-derived gaps)

On the 100 images of the second half (402 correct gaps, 169 false, 226 missed), each error sorted by looking at the labelled packs:

| False gaps (169) | Count |
|---|---|
| Spot is empty in the annotation too, but the annotation-side gap finder does not call it a gap (bay-edge slivers, small position offsets) | 114 |
| The detector missed a real pack, so the spot looks empty | 51 |
| Edges differ from a real gap | 4 |

| Missed gaps (226) | Count |
|---|---|
| A detected box sits in a spot the annotation leaves empty | 106 |
| No predicted gap there | 78 |
| Edges differ from a predicted gap | 42 |

So a missed pack explains only about 30% of the false gaps. The larger share comes from comparing one gap finder against another, which also means the score understates how often a flagged spot is truly empty in the annotation. No real empty-shelf labels exist to measure that directly. Precision is 56% on one half of the images and 70% on the other, so differences of a few F1 points between variants are not reliable; we tested variants on all 200 images with a paired bootstrap where it mattered.

## What we tried and rejected, with numbers

| Idea | Outcome |
|---|---|
| Detection at 1280 and 1536 px | gap F1 0.59-0.63 vs 0.67 at 1024 on the same half of the images, so worse |
| Texture check to drop false gaps | no F1 gain |
| Flip copies, blended views, deeper adapter for naming | none beat 0.81 |
| Pack detector retrained on 2x data | mAP50 0.897 vs 0.888 but gap F1 0.56 vs 0.63, so rejected |
| Threshold tuning | F1 0.69 on one half, too small a gain to change defaults |
| Flip and multi-scale test-time averaging | F1 0.63 vs 0.63 over 200 images; 95% interval for the difference [-0.03, +0.02]; 3x slower |
| Merging the shipped and retrained detectors, or 1024 with 640 | F1 0.63-0.66 vs 0.66 on the held-out half, no gain |
| Edge-of-bay or gap-size rules | precision is only 6-9 points lower at the bay edge and does not vary clearly with gap size |
| Gaps between merged packs that a close look confirms empty, added to the void detector | caught store4_9's gap, but on the 94 labelled photos 16 true and 253 false empty spaces: F1 0.64 to 0.55 (0.60 for gaps of 2+ pack widths) |
| Merging any narrower box above a pack into it (bottle necks) | pack F1 0.806 vs 0.848 on SKU-110K (it merged real stacked packs); only a tight version ships: 0.845 |
| Using the gap confidence score to separate sure from possible gaps | the score is a function of gap width and does not rank correct gaps (precision 57%, 66%, 64%, 59% by quartile) |

The retrain shows that box accuracy alone is not a safe proxy for shelf-level quality, so we measure the end result.

## Edge fit (measured)

**The whole planogram chain** (`scripts/benchmark_chain.py`: product blocks, naming, alignment, packs, empty spaces; the five demo bays of an earlier build, store4_9, store1_36 and store1_47 among them, at their original 3264 px, median over 10 runs after a warm-up):

| Stage | Laptop GPU | 4 CPU cores |
|---|---|---|
| Find product blocks | 59 ms | 357 ms |
| Name each block | 73 ms | 508 ms |
| Align to the plan | 0 ms | 0 ms |
| Count packs | 77 ms | 440 ms |
| Find empty spaces | 51 ms | 193 ms |
| **Whole chain** | **334 ms** | **1,567 ms** |

So one shelf bay is audited in a third of a second on a laptop GPU and in about 1.6 s on four CPU cores with no GPU.

**With the price labels** (`scripts/benchmark_chain.py --prices`, same photos and runs): the whole audit takes a median **2.2 s on the laptop GPU and 4.8 s on four CPU cores with no GPU** (max 9.9 s, on the shelf with 7 labels). Reading the labels is the slow part (1.9 s and 3.5 s): each label is read by seven Tesseract variants side by side and by EasyOCR. Two things made it fit on four cores, both measured:

- Tesseract starts one thread per core and they spin while waiting, so five reads side by side on four cores took 87 s instead of 0.4 s. Each Tesseract process now gets one thread.
- Without a GPU, EasyOCR reads each label once instead of also at 2x scale. On the 31 hand-boxed Grozi labels on four cores: both scales 29 correct (25 confident) at 2.2 s a label, one scale 29 correct (24 confident) at 0.7 s, none wrong either way. On the demo shelves, a CPU-only run gives the GPU run's score on 9 of 10 shelf/plan pairs; on store4_2's planted plan one label's confidence falls below 0.5, so it is a "check by eye" instead of a price mismatch (27 instead of 24).

`scripts/audit.py --cpu` runs the same audit headless and writes one JSON (report, steps, value at stake, timings): what a shelf device would send to the store server. The first run after loading is slower (about 2 s on the GPU). Still untested: real Jetson or phone hardware and int8 quantisation.

The three detectors are YOLO11s (19 MB each as PyTorch weights, 38 MB as ONNX); the timings below are for the pack detector, and the void detector is the same size at 640 px (not timed separately), and the matcher is a ViT-S/14 embedder. Median time to run the pack detector on one photo (8 test photos, `scripts/benchmark_edge.py`), CPU runs pinned to 4 cores of a laptop CPU as a stand-in for a small edge device (no Jetson or phone was available):

| Input size | Laptop GPU (fp16) | CPU, 4 cores, PyTorch | CPU, 4 cores, ONNX Runtime | Empty-space F1 (200 images) |
|---|---|---|---|---|
| 1024 px (shipped) | 47 ms | 357 ms | 224 ms | 0.63 |
| 640 px | 36 ms | 170 ms | 107 ms | 0.60 |

(The last column is the first method's F1 against pack-derived gaps, kept to show the cost of a smaller input; it is not the real-label score.)

So a four-core CPU with no GPU runs the detector in about a quarter of a second per photo, or a tenth of a second at 640 px for 3 points less F1. A shelf audit needs one photo per bay, so this is fast enough without a GPU. Not measured: the 4 GB GPU is the only accelerator we had, so real Jetson or phone timings, int8 quantisation, and the naming step on CPU are untested.

## Reproduce

```bash
PYTHONPATH=src python3 scripts/eval_voids.py data/empty_shelf --void-weights models/void_detector.pt   # empty space vs real labels
PYTHONPATH=src python3 scripts/eval_gaps.py      # first method vs pack-derived gaps
PYTHONPATH=src python3 scripts/train_detector.py --model models/pack_detector.pt --data data/void_yolo/data.yaml --imgsz 640 --batch 8 --epochs 60 --name void_v1   # void detector, ~15 min
PYTHONPATH=src python3 scripts/eval_pipeline.py  # whole-chain scores
PYTHONPATH= taskset -c 0-3 .venv/bin/python scripts/benchmark_edge.py --threads 4   # edge timing, pack detector
PYTHONPATH=src taskset -c 0-3 python3 scripts/benchmark_chain.py --device cpu --threads 4   # edge timing, whole chain
PYTHONPATH=src python3 scripts/audit.py data/demo_bays/store4_9 --plan scenario   # the whole audit, headless
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest          # 160 pass, 1 skipped with tesseract and EasyOCR
```
