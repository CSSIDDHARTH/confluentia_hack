# Demo script (about 5 minutes)

Presenter notes for the vision half. Teammate adds the price-verification step where marked.

## Before you go on stage

```bash
cd ~/confluentia_ml
PYTHONPATH=src .venv/bin/streamlit run app/dashboard.py     # http://localhost:8501
```

- Open the app once and click through both pages so the detector is loaded (first upload otherwise waits a few seconds).
- Have 2-3 fresh shop photos on the desktop, **already tried once**. Pick the ones that worked. Good photos: shelves of packaged goods (bottles, jars, boxes, cans), whole bay in view, shot from 1-2 m, straight on. Avoid fruit and vegetables, close-ups, and price tags alone: the page then shows a "retake" warning instead of a refill list.
- Fallback if the live upload fails or the GPU is busy: the Shelf availability page works from saved results (no GPU), and the planogram demo needs only files. Nothing below needs the internet.

## 1. The problem, in one breath (30 s)

"A store has a planned layout, a planogram. Staff walk the aisle checking it by eye: right product, right place, right count, right price. We do that from a photo."

## 2. Planogram compliance (90 s)

Dashboard, sidebar, Advanced: custom paths. Enter, for the best shelf:

| Field | Path |
|---|---|
| Detection output | `data/demo_bays/store1_47/detection_compliant.json` |
| Planogram | `data/demo_bays/store1_47/planogram_compliant.json` |
| Catalogue | `data/demo_bays/store1_47/catalogue.csv` |

Say: "Real supermarket shelf. The system found the product blocks, named each one, lined them up against the plan and counted the facings. Score 98."

Then swap the two paths to the `_scenario` files:

"Now the same photo, but the store has changed the plan: two products swapped, one product the shelf does not have, and a promotion that needs two more facings. Score drops to **31** and it lists each violation with an action."

Other shelves (compliant score then scenario score): store4_9 88 then 48, store1_36 86 then 34. Do not show store1_44 or store4_2 live; they score 80 on the compliant plan.

## 3. Empty space on any photo (90 s)

Sidebar, **Shelf availability**.

1. Pick `shelf_02_toothpaste_uk` from the saved shelves: 7 empty spaces, 18 packs to refill, fill 79%.
2. Drag the "minimum gap" slider: results recompute instantly. "Stricter means fewer, surer gaps."
3. **Upload a fresh photo.** No planogram needed. It runs the detector live and gives the refill list and the annotated image. Download the CSV.

Say: "This is the wide-shelf mode: it works on any shop photo, because it needs no plan, only the packs it can see."

### Optional: what to do (30 s)

Sidebar, **What to do**. Pick `store1_47`, "Store changed the plan". The report's 8 violation lines become 4 steps; the swap is one step ("Swap Cereals #88 with Cereals #138"). Open the expander to show the 8 lines it replaced.

## 4. Price check (teammate, 60 s)

Teammate shows label reading and the price mismatches. Say plainly that the prices in the demo catalogue are synthetic.

## 5. Close (30 s)

"Detector, product matching, plan alignment, counting, empty-space detection and the report run on one 4 GB laptop GPU. And with no GPU at all, four CPU cores run the pack detector in about a quarter of a second per photo (measured)."

## Questions you will be asked, and honest answers

| Question | Answer |
|---|---|
| How accurate is it? | Empty-space detection: precision 0.63, recall 0.63, F1 0.63 on 200 held-out images (0.59 on one half, 0.67 on the other), measured against gaps derived from human-labelled pack boxes, not against real empty shelf. Product naming: 81% when the plan limits the candidates, 49% with no hint. |
| Are the violations real? | Planted. No public dataset has real planogram violations, so we edited the plan of a real photo. We say so rather than fake the photo. |
| Why these shelves? | Best of the held-out set: 98, 88 and 86 on the compliant plan. Others score 80 or lower. |
| Are the prices real? | No, synthetic catalogue. |
| What fails? | Missed packs show up as false gaps, and lookalike products can be confused. On 24 unseen phone photos, packaged shelves worked, with false flags on about 4 of 9; loose produce does not work; close-ups now get a "retake" warning instead of a wrong refill list. |
| What did you try that did not work? | Larger input sizes, a texture check on gaps, flip and view-blending for naming, and a detector retrained on twice the data (better mAP, worse gap quality). All measured, none shipped. |
| Can it run on the edge? | Measured: pack detector 224 ms per photo on 4 CPU cores with ONNX, 107 ms at lower resolution (5 points less F1). Real Jetson or phone timings are untested. |
| Did you use a small language model? | We tested Qwen2.5 0.5B and 1.5B locally. The 1.5B model took 21 s on CPU and wrote steps like "move X from S3-01 to S3-01"; the 0.5B model failed our checks on 32 of 34 lines. A rule-based planner is exact and instant, so that is what ships. |
| Why not a big vision-language model? | Small specialised models run on the shelf-edge device, cost nothing per image and are measurable. |
