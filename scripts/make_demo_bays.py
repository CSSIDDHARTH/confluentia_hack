"""Build demo bays from held-out Grozi shelves and rank how well the pipeline handles them.

    python scripts/make_demo_bays.py [--top 5]

One pass per photo covers everything (src/shelfsense/vision/bay.py): product blocks are named
and aligned to a planogram, packs are counted inside each block (its facings), and empty
space is found between packs. Each bay gets two planograms:

  compliant  written from the human labels, with facings taken from the pack counts on the
             photo, so a correct pipeline scores 100;
  scenario   a planned change: two neighbouring products swapped (misplaced), one slot now
             expecting a product that is not on the shelf (stock-out), and one product's
             facings raised by two (a promotion reset the shelf has not caught up with: low stock).

Shelves where the compliant plan scores highest are the safest to demo; the top ones are
written to data/demo_bays/<id>/ with a downsized copy of the photo so the dashboard works
without the full Grozi download.

Assumptions (also in CLAUDE.md): Grozi has no prices, so catalogue prices are synthetic and
deterministic; a facing is one visible pack (stacked packs count separately); the pack
detector is the SKU-110K one and is not tuned to Grozi.
"""

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image

from shelfsense.contracts import CatalogueItem, DetectionOutput, Planogram, Shelf, Slot, save_detection_output
from shelfsense.report.compliance import compute_report
from shelfsense.vision.bay import analyze_bay
from shelfsense.vision.detect import VOID_CONF_FLOOR, ProductDetector, group_into_rows
from shelfsense.vision.render import draw_bay_report as draw_report
from shelfsense.vision.grozi import VAL_STORES, labelled_boxes
from shelfsense.vision.sku_match import SkuMatcher, class_index

ROOT = Path(__file__).resolve().parents[1]
GROZI = ROOT / "data" / "grozi"
OUT = ROOT / "data" / "demo_bays"
MAX_SIDE = 1600  # longest side of the photo copy stored with each bay
MIN_FACINGS_SHARE = 0.6  # min_facings = 60% of the facings the plan expects
DISTRACTORS = 0  # extra random candidate products per bay (0 = the planogram's own SKUs only)


def all_labelled(img: Path) -> bool:
    """True when every annotated block on the image has a known product."""
    with open(img.parent.parent / "annotation" / f"{img.stem}.csv", newline="") as f:
        return all(r["label"] != "-1" for r in csv.DictReader(f))


def plan_from_boxes(plan_id: str, name: str, skus: list[str], boxes: list[list[float]]) -> Planogram:
    shelves = []
    for r, row in enumerate(group_into_rows(boxes)):
        slots = [
            Slot(slot_id=f"S{r + 1}-{p + 1:02d}", position=p, sku=skus[i], facings=1, min_facings=1)
            for p, i in enumerate(row)
        ]
        shelves.append(Shelf(shelf_id=f"S{r + 1}", index=r, slots=slots))
    return Planogram(planogram_id=plan_id, fixture_name=name, effective_from="2026-10-01", shelves=shelves)


def with_facings(plan: Planogram, det: DetectionOutput) -> Planogram:
    """Copy of the plan whose facings are the packs the photo actually shows in each slot."""
    p = plan.model_copy(deep=True)
    for slot in (s for sh in p.shelves for s in sh.slots):
        n = max(1, sum(1 for i in det.items if i.slot_id == slot.slot_id))
        slot.facings = n
        slot.min_facings = max(1, round(MIN_FACINGS_SHARE * n))
    return p


def make_scenario(plan: Planogram, pool: list[str], present: set[str]) -> tuple[Planogram, list[str]]:
    """Plant three planned changes: a swap, an absent product, and more facings than the shelf holds."""
    p = plan.model_copy(deep=True)
    notes, used = [], set()
    widest = max(p.shelves, key=lambda s: len(s.slots))
    if len(widest.slots) >= 2 and widest.slots[0].sku != widest.slots[1].sku:
        a, b = widest.slots[0], widest.slots[1]
        a.sku, b.sku = b.sku, a.sku
        used |= {a.slot_id, b.slot_id}
        notes.append(f"swapped {a.slot_id} and {b.slot_id}")
    for shelf in reversed(p.shelves):
        slot = next((s for s in reversed(shelf.slots) if s.slot_id not in used), None)
        if slot is None:
            continue
        # a different category on purpose: closed-set matching tends to 'confirm' a lookalike the plan
        # expects, so a same-category absent product would often go unflagged (see CLAUDE.md)
        other = sorted(s for s in pool if s[:3] != slot.sku[:3] and s not in present)
        if other:
            notes.append(f"{slot.slot_id} now expects {other[0]}, which is not on the shelf")
            slot.sku = other[0]
            used.add(slot.slot_id)
            break
    promo = max((s for sh in p.shelves for s in sh.slots if s.slot_id not in used), key=lambda s: s.facings, default=None)
    if promo is not None:
        promo.facings += 2
        promo.min_facings = promo.facings - 1
        notes.append(f"{promo.slot_id} promotion reset: now {promo.facings} facings planned")
    p.planogram_id = plan.planogram_id + "-scenario"
    p.fixture_name = plan.fixture_name + " (planned changes)"
    return p, notes


def synthetic_catalogue(skus: set[str], names: dict[int, str]) -> dict[str, CatalogueItem]:
    out = {}
    for sku in sorted(skus):
        cls = names.get(int(sku[1:3]), "?")
        h = int(hashlib.sha1(sku.encode()).hexdigest()[:6], 16)
        out[sku] = CatalogueItem(
            sku=sku, name=f"{cls.split('/')[-1]} #{sku[4:]}", brand="Grozi", category=cls, size="",
            price=round(1 + (h % 900) / 100, 2), currency="CHF",
        )
    return out


def write_catalogue(cat: dict[str, CatalogueItem], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CatalogueItem.model_fields))
        w.writeheader()
        for item in cat.values():
            w.writerow(item.model_dump())


def downscale(det: DetectionOutput, factor: float, shown_path: str) -> DetectionOutput:
    """Scale all coordinates so the output matches the stored (downsized) photo copy."""
    det = det.model_copy(deep=True)
    for obj in [*det.shelves, *det.items, *det.gaps, *det.labels]:
        obj.bbox = [round(v * factor, 1) for v in obj.bbox]
    det.image_width, det.image_height = round(det.image_width * factor), round(det.image_height * factor)
    det.image_path = shown_path
    return det


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    blocks = ProductDetector(ROOT / "models" / "product_detector.pt")
    packs = ProductDetector(ROOT / "models" / "pack_detector.pt", imgsz=1024, max_det=600)
    void_weights = ROOT / "models" / "void_detector.pt"
    voider = ProductDetector(void_weights, conf=VOID_CONF_FLOOR, imgsz=640, max_det=150) if void_weights.exists() else None
    matcher = SkuMatcher.load(GROZI / "sku_index.npz", ROOT / "models" / "sku_adapter.npy")
    names = class_index(GROZI)
    known = sorted(matcher.index.unique_skus)
    by_img: dict[Path, list] = {}
    for img, sku, rel in labelled_boxes(GROZI, VAL_STORES, set(known)):
        by_img.setdefault(img, []).append((sku, rel))

    ranked = []
    for img, items in by_img.items():
        if len(items) < 4 or not all_labelled(img):  # four blocks: room for all three planted changes
            continue
        w, h = Image.open(img).size
        boxes = [[l * w, t * h, r * w, b * h] for _, (l, t, r, b) in items]
        base = plan_from_boxes(f"PG-{img.parent.parent.name}-{img.stem}", f"{img.parent.parent.name} photo {img.stem}", [s for s, _ in items], boxes)
        present = {s for s, _ in items}
        # In deployment the candidates for a bay are the planogram's own products. DISTRACTORS adds that many random
        # extra products (seeded per photo) as a stress test: 0 = planogram only.
        distractors = set(random.Random(f"{img.parent.parent.name}-{img.stem}").sample(known, DISTRACTORS)) if DISTRACTORS else set()
        cands = {s.sku for sh in base.shelves for s in sh.slots} | distractors
        det0 = analyze_bay(img, blocks, packs, matcher, base, cands)
        plan = with_facings(base, det0)
        cat = synthetic_catalogue(present, names)
        report = compute_report(det0, plan, cat, group_items=True)
        ranked.append((report.score, len(det0.items), img, plan, present, distractors, cat))

    ranked.sort(key=lambda r: (-r[0], -r[1]))
    print(f"{len(ranked)} candidate shelves; compliant-plan score distribution: "
          f"{sum(r[0] == 100 for r in ranked)} perfect, {sum(r[0] >= 85 for r in ranked)} >= 85")

    if OUT.exists():
        shutil.rmtree(OUT)
    for score, n, img, plan, present, distractors, cat in ranked[: a.top]:
        bay = OUT / f"{img.parent.parent.name}_{img.stem}"
        bay.mkdir(parents=True)
        scenario, notes = make_scenario(plan, known, present)
        pool_cat = {**cat, **synthetic_catalogue({s.sku for sh in scenario.shelves for s in sh.slots}, names)}
        write_catalogue(pool_cat, bay / "catalogue.csv")

        photo = Image.open(img).convert("RGB")
        factor = min(1.0, MAX_SIDE / max(photo.size))
        small = photo.resize((round(photo.width * factor), round(photo.height * factor)), Image.LANCZOS)
        small.save(bay / "photo.jpg", quality=88)
        shown = str((bay / "photo.jpg").relative_to(ROOT))

        for tag, pl in (("compliant", plan), ("scenario", scenario)):
            (bay / f"planogram_{tag}.json").write_text(pl.model_dump_json(indent=2))
            cands = {s.sku for sh in pl.shelves for s in sh.slots} | distractors
            det = downscale(analyze_bay(img, blocks, packs, matcher, pl, cands, void_detector=voider), factor, shown)
            det.image_id = bay.name
            report = compute_report(det, pl, pool_cat, group_items=True)
            save_detection_output(det, bay / f"detection_{tag}.json")
            draw_report(small, det, report, tag).save(bay / f"overlay_{tag}.jpg", quality=85)
            print(f"  {bay.name:10s} {tag:9s} score {report.score:5.1f}  {report.counts}  packs {len(det.items)}, empty spaces {len(det.gaps)}")
        (bay / "README.txt").write_text(
            f"Held-out Grozi shelf {img.relative_to(ROOT)} (downsized copy: photo.jpg)\n"
            f"compliant: planogram from the human labels, facings from the pack counts on the photo\n"
            f"scenario : {'; '.join(notes) or 'no changes possible'}\n"
        )
        print(f"    scenario: {'; '.join(notes)}")


if __name__ == "__main__":
    main()
