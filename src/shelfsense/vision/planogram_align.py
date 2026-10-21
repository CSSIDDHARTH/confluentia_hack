"""Phase 3 vision stage: assign each detected item to a planogram slot.

For every shelf row the detected products (left to right) are aligned against the
planogram's expected products (left to right) with Needleman-Wunsch sequence
alignment, so:
  * a missing product leaves a slot with no item (the report flags a stock-out),
  * a wrong product paired with a slot is reported as misplaced,
  * an unrecognised item (sku None) is a soft mismatch rather than a hard one,
  * an extra item with no slot is attached to the nearest aligned slot.

Rows are matched to planogram shelves top to bottom. Items in rows beyond the
planogram's shelves keep slot_id None.
"""

from shelfsense.contracts import DetectionOutput, Item, Planogram, Slot

GAP_COST = 1.0
MISMATCH_COST = 1.0
UNKNOWN_COST = 0.4  # an unrecognised item could well be the expected product


def _sub_cost(item: Item, slot: Slot) -> float:
    if item.sku is None:
        return UNKNOWN_COST
    return 0.0 if item.sku == slot.sku else MISMATCH_COST


def align_row(items: list[Item], slots: list[Slot]) -> dict[str, str]:
    """Align items (left to right) to slots (left to right). Returns item_id -> slot_id.

    Ties prefer pairing an item with a slot (a substitution) over leaving both
    unmatched, so a swapped pair is reported as two misplaced items.
    """
    n, m = len(items), len(slots)
    if m == 0:
        return {}
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i * GAP_COST
    for j in range(1, m + 1):
        cost[0][j] = j * GAP_COST
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + _sub_cost(items[i - 1], slots[j - 1]),
                cost[i - 1][j] + GAP_COST,
                cost[i][j - 1] + GAP_COST,
            )

    paired: dict[int, int] = {}  # item index -> slot index
    i, j = n, m
    while i > 0 and j > 0:
        if cost[i][j] == cost[i - 1][j - 1] + _sub_cost(items[i - 1], slots[j - 1]):
            paired[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + GAP_COST:
            i -= 1
        else:
            j -= 1

    out: dict[str, str] = {}
    for idx, item in enumerate(items):
        if idx in paired:
            out[item.item_id] = slots[paired[idx]].slot_id
            continue
        # extra item: attach to the nearest paired neighbour (previous first, else next)
        prev = next((paired[k] for k in range(idx - 1, -1, -1) if k in paired), None)
        nxt = next((paired[k] for k in range(idx + 1, n) if k in paired), None)
        chosen = prev if prev is not None else nxt
        out[item.item_id] = slots[chosen if chosen is not None else 0].slot_id
    return out


def assign_slots(det: DetectionOutput, planogram: Planogram) -> DetectionOutput:
    """Set planogram_id and each item's shelf_id / slot_id from the alignment."""
    det.planogram_id = planogram.planogram_id
    shelves = sorted(planogram.shelves, key=lambda s: s.index)
    row_ids = [r.shelf_id for r in det.shelves]  # detector rows, top to bottom

    for k, row_id in enumerate(row_ids):
        row_items = sorted((i for i in det.items if i.shelf_id == row_id), key=lambda i: i.bbox[0])
        if k >= len(shelves):
            for item in row_items:
                item.slot_id = None
            continue
        shelf = shelves[k]
        mapping = align_row(row_items, sorted(shelf.slots, key=lambda s: s.position))
        for item in row_items:
            item.shelf_id = shelf.shelf_id
            item.slot_id = mapping[item.item_id]

    for region, row_id in zip(det.shelves, row_ids):
        k = row_ids.index(row_id)
        if k < len(shelves):
            region.shelf_id = shelves[k].shelf_id
    det.produced_by["alignment"] = "nw-v1"
    return det
