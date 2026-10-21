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

1. Pick `shelf_02_toothpaste_uk` from the saved shelves: 9 empty spaces, 16 packs to refill, fill 89%.
2. Drag the "Empty-space confidence" slider: results recompute instantly. "Higher means fewer, surer empty spaces."
3. **Upload a fresh photo.** No planogram needed. It runs the detector live and gives the refill list and the annotated image. Download the CSV.

Say: "This is the wide-shelf mode: it works on any shop photo, because it needs no plan, only the packs it can see."

### Let a judge plant the violation (60 s, the strongest part)

Sidebar, **What to do**. First show `store1_47`, "Store changed the plan": the report's 8 violation lines become 4 steps; the swap is one step ("Swap Cereals #88 with Cereals #138") and each misplaced product says where it belongs ("Cereals #138 belongs in S1-01"). Open the expander to show the 8 lines it replaced.

Then choose **Edit the plan yourself** and say: "Those changes were ours. Now you pick." Ask a judge to swap two slots (pick them, press *Swap their products*), change a slot's expected product in the table, or raise *facings planned* and *minimum* together on a slot (a promotion reset). The score and steps update at once, with the change versus the real plan shown under the score. Be clear about what is fixed: the photo is read once; only the plan changes, so the score moves because the shelf no longer matches what the plan expects.

## 4. Price check (teammate, 60 s)

Teammate shows label reading and the price mismatches. Say plainly that the prices in the demo catalogue are synthetic.

## 5. Close (30 s)

"Detector, product matching, plan alignment, counting, empty-space detection and the report run on one 4 GB laptop GPU. And with no GPU at all, four CPU cores run the pack detector in about a quarter of a second per photo (measured)."

## Questions you will be asked, and honest answers

| Question | Answer |
|---|---|
| How accurate is it? | Empty-space detection, against hand-labelled empty spots on 94 held-out photos: precision 0.66, recall 0.62, F1 0.64. Product naming: 81% when the plan limits the candidates, 49% with no hint. |
| Where do the empty-space labels come from? | A public Roboflow set of 1,050 shelf photos with empty spots drawn by hand (CC BY 4.0). We trained on its train split and report on its test split. Our first method, inferring gaps from missing packs, scored 0.27 on the same labels, so we replaced it. |
| Are the violations real? | No public dataset has real planogram violations, so a violation is a plan that disagrees with a real photo. We planted some, and on the "Edit the plan yourself" page you plant your own and watch the score move. We do not edit the photo, because erased packs looked fake. |
| Why these shelves? | Best of the held-out set: 98, 88 and 86 on the compliant plan. Others score 80 or lower. |
| Are the prices real? | No, synthetic catalogue. |
| What fails? | Lookalike products can be confused, and a few small false empty spaces appear on gappy-looking products. On 24 unseen phone photos, packaged shelves worked (the 5 full shelves of 9 showed no false flags); loose produce does not work; close-ups get a "retake" warning instead of a wrong refill list. The empty-space labels are 416 px photos from one source, so other shops are untested. |
| What did you try that did not work? | Larger input sizes, a texture check on gaps, test-time averaging, merging detectors, flip and view-blending for naming, and a pack detector retrained on twice the data (better mAP, worse gap quality). All measured, none shipped. The biggest lesson: our proxy metric said 0.63 and real labels said 0.27, so we switched to real labels. |
| Can it run on the edge? | Measured: pack detector 224 ms per photo on 4 CPU cores with ONNX, 107 ms at lower resolution (5 points less F1). Real Jetson or phone timings are untested. |
| Did you use a small language model? | We tested Qwen2.5 0.5B and 1.5B locally. The 1.5B model took 21 s on CPU and wrote steps like "move X from S3-01 to S3-01"; the 0.5B model failed our checks on 32 of 34 lines. A rule-based planner is exact and instant, so that is what ships. |
| Why not a big vision-language model? | Small specialised models run on the shelf-edge device, cost nothing per image and are measurable. |
