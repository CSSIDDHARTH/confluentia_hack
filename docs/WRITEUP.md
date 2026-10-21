# shelfSense AI: vision pipeline

Written for judges. Covers the vision half (photo to products, placement, counts, empty space); price verification and the dashboard report are described by Person B.

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
6. **Report**: a compliance score with each violation and a corrective action.

A second mode, *shelf availability*, skips the planogram and gives empty spaces, per-row fill and a refill list for any shelf photo, uploaded live.

## From violation lines to steps

The compliance report lists one line per violation, so one swap of two products is four lines (a stock-out and a misplaced item in each slot). `advisor/plan.py` merges them with rules: a swap becomes one step naming both slots, a wrong product in a slot that lacks the right one becomes one "move out, then restock" step, a misplaced product is told where it belongs ("Cereals #138 belongs in S1-01", from the planogram, or "the plan has no other slot for it"), and steps are ordered by severity. Across the 10 demo reports (5 shelves, correct plan and planted changes) 47 violation lines become 34 steps, and each violation lands in exactly one step. The planted swap shows up as a clean swap in the report on 2 of the 5 shelves, and the planner merged both. On the other 3 the vision output assigned a product block's packs across both slots, so the report shows low stock instead of two stock-outs and there is nothing to merge.

The "What to do" page also has an **Edit the plan yourself** mode (`advisor/edit.py`): a person changes what a slot expects, its facings and minimum, or swaps two slots, and the score and steps recompute from the unchanged photo reading. This is how a violation gets created without us planting it. It tests the comparison, not the detector, since the photo is read once.

### A small language model: tested, not used

The brief invites a small language model (under 20B parameters) where it fits. We tested Qwen2.5-1.5B and 0.5B, run locally, in two roles:

- **Deciding what to do** from the raw violation lines: every slot and number it wrote existed in the report (10 of 10 shelves), but it told staff to restock two products that were simply in each other's slots (the right instruction is a swap), named the wrong item to verify, and padded a one-violation shelf with generic advice. Its answers pass a "mentions only known slots and numbers" check, so that check does not catch wrong meaning.
- **Rewording the rule-based steps**, with every line checked to name the same slots and products and add no numbers: the 1.5B model kept 22 of 34 lines, took a median 21 s per shelf on 16 CPU cores (5 tokens/s) and does not fit on the 4 GB GPU next to the desktop; the 0.5B model (2 s on GPU) kept 2 of 34. Kept lines were still worse than the rules: "Move Cereals #138 from S3-01 to S3-01", "9 facing slots".

So the rules decide and phrase the steps, which is exact and instant. A larger model (3B or more) might word them well, but would need a semantic check we do not have, so we did not ship one.

## Measured results

All on held-out data (different stores or different images from training).

| Part | Result |
|---|---|
| Product-block detector | trained on stores 2, 3, 5; evaluated on stores 1, 4 |
| Pack detector | mAP50 0.89 on 200 held-out SKU-110K images |
| Naming, planogram-restricted | 0.81 top-1 |
| Naming with 20 extra distractors | 0.79 |
| Naming, open set (no planogram) | 0.49 |
| Whole chain (detect, name, align, count) on the best 3 held-out shelves | compliance score 98, 88, 86 on the correct plan |
| Planted-change shelves | scores fall to 31 to 48 with every planted change listed |
| **Empty space vs real hand-labelled empty spots**, 94 held-out photos, 639 labelled spots | precision 0.66, recall 0.62, **F1 0.64** (void detector, confidence 0.3 chosen on a separate split) |
| Same photos, our first method (gaps inferred from missing packs) | precision 0.29, recall 0.25, F1 0.27 |

## Assumptions and limits (stated, not hidden)

- Front-facing photo of one shelf bay; each image is tagged with its planogram.
- Planograms are ordinal (shelf, slot order, facings); public planograms carry no physical measurements.
- **No public dataset has real planogram violations, real prices or price-tag text.** The violations are planted by editing the planogram of a real photo, and prices are synthetic. We did not edit photos, because erased packs looked fake.
- Demo shelves are the best of the held-out set. Other held-out shelves score 80 or lower.
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
| Using the gap confidence score to separate sure from possible gaps | the score is a function of gap width and does not rank correct gaps (precision 57%, 66%, 64%, 59% by quartile) |

The retrain shows that box accuracy alone is not a safe proxy for shelf-level quality, so we measure the end result.

## Edge fit (measured)

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
PYTHONPATH= taskset -c 0-3 .venv/bin/python scripts/benchmark_edge.py --threads 4   # edge timing
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest          # 126 pass, 2 skipped (no tesseract)
```
