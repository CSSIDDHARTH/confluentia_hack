"""Build demo bays from held-out Grozi shelves and rank how well the pipeline handles them.

    python scripts/make_demo_bays.py [--top 5]

For each held-out shelf photo whose products are all labelled, this writes a planogram from
the human labels (the "compliant" reference) and a scenario planogram with planted changes
(two products swapped; one slot expecting a product that is not on the shelf). The vision
pipeline (detect, identify, align) is run against both and scored with the report code.
The shelves where the compliant plan scores highest are the safest to demo; the top ones
are written to data/demo_bays/<id>/.

Assumptions (also in CLAUDE.md): Grozi has no prices, so catalogue prices are synthetic and
deterministic. Each slot holds one product block (facings 1), because Grozi boxes cover a
whole block of packs.
"""

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shelfsense.contracts import CatalogueItem, Planogram, Shelf, Slot, save_detection_output
from shelfsense.report.compliance import ViolationKind, compute_report
from shelfsense.vision.detect import ProductDetector, group_into_rows
from shelfsense.vision.grozi import VAL_STORES, labelled_boxes
from shelfsense.vision.pipeline import identify_items
from shelfsense.vision.planogram_align import assign_slots
from shelfsense.vision.sku_match import SkuMatcher, class_index

ROOT = Path(__file__).resolve().parents[1]
GROZI = ROOT / "data" / "grozi"
OUT = ROOT / "data" / "demo_bays"


def all_labelled(img: Path) -> bool:
    """True when every annotated block on the image has a known product."""
    ann = img.parent.parent / "annotation" / f"{img.stem}.csv"
    with open(ann, newline="") as f:
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


def make_scenario(plan: Planogram, pool: list[str], present: set[str]) -> tuple[Planogram, list[str]]:
    """Plant two changes: swap two neighbouring products, and make one slot expect an absent one."""
    p = plan.model_copy(deep=True)
    notes = []
    widest = max(p.shelves, key=lambda s: len(s.slots))
    used = set()
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
        # expects, so a same-category absent product would often go unflagged (see README)
        same_class = sorted(s for s in pool if s[:3] != slot.sku[:3] and s not in present)
        if same_class:
            notes.append(f"{slot.slot_id} now expects {same_class[0]}, which is not on the shelf")
            slot.sku = same_class[0]
            break
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


def run_pipeline(img, detector, matcher, plan, extra, rng):
    """Detect, identify (candidates = plan SKUs + 20 distractors) and align to the plan."""
    cands = {s.sku for sh in plan.shelves for s in sh.slots} | extra
    det = detector.detect(img.resolve(), rel_to=ROOT)
    det.image_id = f"{img.parent.parent.name}_{img.stem}"
    im = Image.open(img).convert("RGB")
    identify_items(det, im, matcher, cands)
    return assign_slots(det, plan), im


def draw_report(im: Image.Image, det, report, title: str) -> Image.Image:
    bad = {v.violation_id.split("-")[1]: v.kind for v in report.violations if v.violation_id.startswith("v-i")}
    colours = {ViolationKind.MISPLACED: (214, 40, 40), ViolationKind.UNRECOGNISED_ITEM: (240, 160, 32)}
    im = im.copy()
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(18, im.width // 55))
    lw = max(4, im.width // 400)
    for it in det.items:
        colour = colours.get(bad.get(it.item_id), (46, 158, 68))
        d.rectangle(it.bbox, outline=colour, width=lw)
        label = f"{it.slot_id or 'no slot'} {it.sku[4:] if it.sku else '?'}"
        tx, ty = it.bbox[0] + 8, it.bbox[1] + 8
        d.rectangle(d.textbbox((tx, ty), label, font=font), fill=(0, 0, 0))
        d.text((tx, ty), label, fill=(255, 255, 0), font=font)
    banner = f"{title}: score {report.score:.0f}  " + "  ".join(f"{k.replace('_', ' ')} {n}" for k, n in report.counts.items())
    strip = font.size + 24
    canvas = Image.new("RGB", (im.width, im.height + strip), (0, 0, 0))  # header sits above the photo
    canvas.paste(im, (0, strip))
    ImageDraw.Draw(canvas).text((10, 10), banner, fill=(255, 255, 255), font=font)
    canvas.thumbnail((1400, 1400))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    detector = ProductDetector(ROOT / "models" / "product_detector.pt")
    matcher = SkuMatcher.load(GROZI / "sku_index.npz", ROOT / "models" / "sku_adapter.npy")
    names = class_index(GROZI)
    known = sorted(matcher.index.unique_skus)
    rng = random.Random(0)

    by_img: dict[Path, list] = {}
    for img, sku, rel in labelled_boxes(GROZI, VAL_STORES, set(known)):
        by_img.setdefault(img, []).append((sku, rel))

    ranked = []
    for img, items in by_img.items():
        if len(items) < 3 or not all_labelled(img):
            continue
        w, h = Image.open(img).size
        boxes = [[l * w, t * h, r * w, b * h] for _, (l, t, r, b) in items]
        plan = plan_from_boxes(f"PG-{img.parent.parent.name}-{img.stem}", f"{img.parent.parent.name} photo {img.stem}", [s for s, _ in items], boxes)
        present = {s for s, _ in items}
        cat = synthetic_catalogue(present, names)
        distractors = set(rng.sample(known, 20))
        det, _ = run_pipeline(img, detector, matcher, plan, distractors, rng)
        report = compute_report(det, plan, cat)
        ranked.append((report.score, len(items), img, plan, present, distractors, cat))

    ranked.sort(key=lambda r: (-r[0], -r[1]))
    print(f"{len(ranked)} candidate shelves; compliant-plan score distribution: "
          f"{sum(r[0] == 100 for r in ranked)} perfect, {sum(r[0] >= 85 for r in ranked)} >= 85")

    if OUT.exists():
        shutil.rmtree(OUT)
    for score, n, img, plan, present, distractors, cat in ranked[: a.top]:
        bay = OUT / f"{img.parent.parent.name}_{img.stem}"
        bay.mkdir(parents=True)
        scenario, notes = make_scenario(plan, known, present)
        scen_cat = synthetic_catalogue(present | {s.sku for sh in scenario.shelves for s in sh.slots}, names)
        pool_cat = {**cat, **scen_cat}
        write_catalogue(pool_cat, bay / "catalogue.csv")
        for tag, pl in (("compliant", plan), ("scenario", scenario)):
            (bay / f"planogram_{tag}.json").write_text(pl.model_dump_json(indent=2))
            det, im = run_pipeline(img, detector, matcher, pl, distractors, rng)
            report = compute_report(det, pl, pool_cat)
            save_detection_output(det, bay / f"detection_{tag}.json")
            draw_report(im, det, report, tag).convert("RGB").save(bay / f"overlay_{tag}.jpg", quality=85)
            print(f"  {bay.name:12s} {tag:9s} score {report.score:5.1f}  {report.counts}")
        (bay / "README.txt").write_text(
            f"Held-out Grozi shelf {img.relative_to(ROOT)}\n"
            f"compliant: planogram written from the human labels\n"
            f"scenario : {'; '.join(notes) or 'no changes possible'}\n"
        )
        print(f"    scenario: {'; '.join(notes)}")


if __name__ == "__main__":
    main()
