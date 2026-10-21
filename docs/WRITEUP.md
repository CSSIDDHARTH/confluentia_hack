# shelfSense AI: vision pipeline

Written for judges. Covers the vision half (photo to products, placement, counts, empty space); price verification and the dashboard report are described by Person B.

## What it does

From one shelf photo it answers: is the right product in the right place, in the right quantity, and where is the shelf empty?

```
photo -> product-block detector -> product naming -> align to planogram -> count facings -> empty space -> report
         (YOLO11s, Grozi)          (DINOv2 + adapter)  (sequence alignment)  (pack detector)   (row gaps)
                                                         pack detector: YOLO11s trained on SKU-110K
```

1. **Detect** product blocks (one box per product, all its facings) with a YOLO11s detector trained on Grozi supermarket shelves.
2. **Name** each block by comparing its DINOv2 embedding with 8,133 studio reference images, restricted to the products the planogram expects. A small linear adapter, trained on shelf crops, closes the gap between studio and shelf photos. Low-confidence matches (below 0.5) are marked unknown for review.
3. **Align** the detected blocks to the planogram row by row with Needleman-Wunsch sequence alignment, so a swap, a missing product or an extra product shows up as a specific mismatch.
4. **Count** facings with a second YOLO11s detector that finds individual packs (trained on SKU-110K). Each pack becomes an item that carries its block's product and slot, so the report's rule "items in slot = facings" yields low stock and stock-outs.
5. **Find empty space**: rows come from the vertical coverage of packs; a stretch of a row with no pack and at least 0.8 of the median pack width is an empty space, and its width over the median pack width is the number of missing facings. A row covered under 60% is low stock.
6. **Report**: a compliance score with each violation and a corrective action.

A second mode, *shelf availability*, skips the planogram and gives empty spaces, per-row fill and a refill list for any shelf photo, uploaded live.

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
| Empty space vs gaps derived from labelled packs | precision 0.70, recall 0.64, F1 0.67 (tuned on one half, reported on the other) |
| A sold-out product column | found 64% of the time |

## Assumptions and limits (stated, not hidden)

- Front-facing photo of one shelf bay; each image is tagged with its planogram.
- Planograms are ordinal (shelf, slot order, facings); public planograms carry no physical measurements.
- **No public dataset has real planogram violations, real prices or price-tag text.** The violations are planted by editing the planogram of a real photo, and prices are synthetic. We did not edit photos, because erased packs looked fake.
- Demo shelves are the best of the held-out set. Other held-out shelves score 80 or lower.
- Use packaged-goods shelves photographed with the whole bay in view. Loose produce and close-ups are outside what the pack detector handles.
- Empty-space accuracy is relative to gaps derived from human pack boxes, not to measured empty shelf. Missed packs appear as false gaps.
- Datasets are academic-use only (Grozi-3.2K, SKU-110K).

## Robustness on photos we did not train on

24 real phone photos of shops (Open Food Facts price-tag set, random draw, never used to train the pack detector), run through the live upload path and judged by eye. No ground truth, so this is a qualitative check, not an accuracy number.

| Kind of photo | Count | What happened |
|---|---|---|
| Shelves of packaged goods, whole bay in view | 9 | Packs found and counted sensibly. 4 of the 9 also had a false empty space or low-stock flag, usually a price-label strip read as a shelf row or a missed pack. |
| Loose produce (fruit, vegetables) | 6 | Fruit boxed as "packs"; fill percentages are not meaningful. Out of scope for a pack detector. |
| Close-ups, price tags, too few packs (under 8) | 8 | Now flagged "looks like a close-up, retake from 1-2 m" and no refill list is given. Before the guard these produced false "low stock" alerts. |
| Cluttered shelf with overlapping products | 1 | False empty spaces over products the detector missed. |

A second guard drops a "row" of at most 3 flat boxes under much taller packs (a price-label strip). It removed the false low-stock row in one of the two photos where it occurred, left the other, and changed neither the empty-space F1 (0.67) nor any of the 8 test shelves.

The 8-pack guard separated every close-up from every usable shelf in this sample and does not trigger on any of the 8 test shelves (125-189 packs each). Its threshold was set after looking at these 24 photos, so treat it as a sanity check, not a tuned result.

## What we tried and rejected, with numbers

| Idea | Outcome |
|---|---|
| Detection at 1280 and 1536 px | gap F1 0.59-0.63, worse than 1024 |
| Texture check to drop false gaps | no F1 gain |
| Flip copies, blended views, deeper adapter for naming | none beat 0.81 |
| Pack detector retrained on 2x data | mAP50 0.897 vs 0.888 but gap F1 0.55 vs 0.67, so rejected |
| Threshold tuning | F1 0.69, too small a gain to change defaults |

The retrain shows that box accuracy alone is not a safe proxy for shelf-level quality, so we measure the end result.

## Edge fit (measured)

Each detector is YOLO11s (19 MB as PyTorch weights, 38 MB as ONNX), and the matcher is a ViT-S/14 embedder. Median time to run the pack detector on one photo (8 test photos, `scripts/benchmark_edge.py`), CPU runs pinned to 4 cores of a laptop CPU as a stand-in for a small edge device (no Jetson or phone was available):

| Input size | Laptop GPU (fp16) | CPU, 4 cores, PyTorch | CPU, 4 cores, ONNX Runtime | Empty-space F1 |
|---|---|---|---|---|
| 1024 px (shipped) | 47 ms | 357 ms | 224 ms | 0.67 |
| 640 px | 36 ms | 170 ms | 107 ms | 0.62 |

So a four-core CPU with no GPU runs the detector in about a quarter of a second per photo, or a tenth of a second at 640 px for 5 points less F1. A shelf audit needs one photo per bay, so this is fast enough without a GPU. Not measured: the 4 GB GPU is the only accelerator we had, so real Jetson or phone timings, int8 quantisation, and the naming step on CPU are untested.

## Reproduce

```bash
PYTHONPATH=src python3 scripts/eval_gaps.py      # empty-space accuracy
PYTHONPATH=src python3 scripts/eval_pipeline.py  # whole-chain scores
PYTHONPATH= taskset -c 0-3 .venv/bin/python scripts/benchmark_edge.py --threads 4   # edge timing
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest          # 105 pass, 2 skipped (no tesseract)
```
