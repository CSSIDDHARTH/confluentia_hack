"""Generate a synthetic DetectionOutput for a planogram, with per-slot
overrides, so new demo/test scenarios don't need their bboxes hand-edited.

Lays slots out left-to-right within each shelf band (same three shelf
bands as contracts/detection_output.example.json, for visual consistency
with the existing mock image), placing one item box per facing and one
shelf-edge label per slot (below and left-aligned with the first facing,
per CLAUDE.md assumption 4). Slots keep their planogram width, so missing
facings show up as empty space and are reported as gaps.

    python scripts/gen_scenario.py

Library usage (see __main__ below for the two scenarios this produces):
    build_scenario(planogram, catalogue, image_id, image_path, ...)
"""

from __future__ import annotations

from pathlib import Path

from shelfsense.contracts import (
    DetectionOutput,
    Gap,
    Item,
    Label,
    Planogram,
    ShelfRegion,
    load_catalogue,
    load_planogram,
    save_detection_output,
)

ROOT = Path(__file__).resolve().parents[1]

SHELF_BANDS = {0: (60, 310), 1: (340, 590), 2: (620, 870)}
SHELF_X1, SHELF_X2 = 20, 880
ITEM_W, ITEM_GAP, SLOT_GAP = 100, 10, 20
LABEL_W, LABEL_H = 90, 24


def build_scenario(
    planogram: Planogram,
    catalogue: dict,
    image_id: str,
    image_path: str,
    planogram_id: str,
    *,
    facings_override: dict[str, int] | None = None,
    misplace: dict[str, str] | None = None,
    unrecognised: set[str] | None = None,
    label_overrides: dict[str, dict] | None = None,
) -> DetectionOutput:
    """Build a DetectionOutput where, by default, every slot is exactly
    compliant (facings items matching the slot's sku, one label at the
    catalogue price). Overrides create the violations a scenario wants:

    - facings_override[slot_id] = n: place n matching items instead of
      the planogram's `facings` (0 for stock-out, < min_facings for
      low-stock).
    - misplace[slot_id] = sku: swap one of the slot's items to `sku`
      (misplaced-product violation).
    - unrecognised: slot_ids where one item's sku is set to None.
    - label_overrides[slot_id] = {"price": None}  -> unreadable price
                                  {"price_delta": x} -> price mismatch
                                  {"sku": None}  -> unknown product
                                  {"skip": True} -> no label at all
    """
    facings_override = facings_override or {}
    misplace = misplace or {}
    unrecognised = unrecognised or set()
    label_overrides = label_overrides or {}

    shelves: list[ShelfRegion] = []
    items: list[Item] = []
    gaps: list[Gap] = []
    labels: list[Label] = []
    item_n = 0
    gap_n = 0
    label_n = 0

    for shelf in planogram.shelves:
        y1, y2 = SHELF_BANDS[shelf.index]
        shelves.append(ShelfRegion(shelf_id=shelf.shelf_id, bbox=[SHELF_X1, y1, SHELF_X2, y2]))
        x = SHELF_X1 + 10

        for slot in shelf.slots:
            # Each slot keeps its planogram width; missing facings stay empty.
            slot_x1 = x
            n = min(facings_override.get(slot.slot_id, slot.facings), slot.facings)

            for i in range(n):
                item_n += 1
                sku = slot.sku
                if slot.slot_id in misplace and i == n - 1:
                    sku = misplace[slot.slot_id]
                if slot.slot_id in unrecognised and i == 0:
                    sku = None
                items.append(
                    Item(
                        item_id=f"i{item_n:02d}",
                        bbox=[x, y1 + 20, x + ITEM_W, y2 - 40],
                        det_confidence=0.95,
                        sku=sku,
                        sku_confidence=0.9 if sku else 0.4,
                        shelf_id=shelf.shelf_id,
                        slot_id=slot.slot_id,
                    )
                )
                x += ITEM_W + ITEM_GAP

            missing = slot.facings - n
            if missing:
                gap_n += 1
                gap_x2 = x + missing * (ITEM_W + ITEM_GAP) - ITEM_GAP
                gaps.append(
                    Gap(
                        gap_id=f"g{gap_n:02d}",
                        bbox=[x, y1 + 20, gap_x2, y2 - 40],
                        shelf_id=shelf.shelf_id,
                        slot_id=slot.slot_id,
                        confidence=0.9,
                        estimated_missing_facings=missing,
                    )
                )
                x = gap_x2 + ITEM_GAP

            x += SLOT_GAP

            override = label_overrides.get(slot.slot_id, {})
            if not override.get("skip"):
                price = catalogue[slot.sku].price
                if "price_delta" in override:
                    price = round(price + override["price_delta"], 2)
                elif "price" in override:
                    price = override["price"]
                label_n += 1
                labels.append(
                    Label(
                        label_id=f"l{label_n:02d}",
                        bbox=[slot_x1, y2 - LABEL_H - 8, slot_x1 + LABEL_W, y2 - 8],
                        shelf_id=shelf.shelf_id,
                        slot_id=slot.slot_id,
                        sku=override.get("sku", slot.sku),
                        ocr_text=f"${price:.2f}" if price is not None else "$?.??",
                        price=price,
                        ocr_confidence=0.9 if price is not None else 0.3,
                    )
                )

    return DetectionOutput(
        image_id=image_id,
        image_path=image_path,
        image_width=900,
        image_height=900,
        planogram_id=planogram_id,
        captured_at="2026-10-20T09:30:00Z",
        produced_by={"vision": "mock-0", "pricing": "mock-0", "generator": "scripts/gen_scenario.py"},
        shelves=shelves,
        items=items,
        gaps=gaps,
        labels=labels,
    )


if __name__ == "__main__":
    planogram = load_planogram(ROOT / "contracts/planogram.example.json")
    catalogue = load_catalogue(ROOT / "contracts/catalogue.example.csv")

    low_stock = build_scenario(
        planogram,
        catalogue,
        image_id="mock_shelf_002_lowstock",
        image_path="data/mock/mock_shelf_002_lowstock.png",
        planogram_id=planogram.planogram_id,
        facings_override={"S1-01": 1, "S2-02": 0},
        misplace={"S1-02": "SKU-2002"},
        unrecognised={"S3-01"},
        label_overrides={"S2-03": {"price_delta": 0.50}},
    )
    save_detection_output(low_stock, ROOT / "data/mock/mock_shelf_002_lowstock.json")
    print("wrote data/mock/mock_shelf_002_lowstock.json")

    near_clean = build_scenario(
        planogram,
        catalogue,
        image_id="mock_shelf_003_clean",
        image_path="data/mock/mock_shelf_003_clean.png",
        planogram_id=planogram.planogram_id,
        label_overrides={"S1-03": {"price": None}},
    )
    save_detection_output(near_clean, ROOT / "data/mock/mock_shelf_003_clean.json")
    print("wrote data/mock/mock_shelf_003_clean.json")
