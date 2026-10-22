"""Tie detected price labels to planogram slots.

label_detector.detect_labels() returns labels with shelf_id/slot_id/sku
None. Given the vision stage's output (shelves, and items/gaps already
assigned to slots), this fills them in, using assumption 4: each slot has
one label, below and left-aligned with the slot's first facing.

The planogram has no slot geometry (it is ordinal), so a slot's left edge
is taken from the items and gaps the vision stage put in it. A slot with
nothing in it has no edge and cannot receive a label.
"""

from __future__ import annotations

from shelfsense.contracts import DetectionOutput, Label, Planogram


def _shelf_for(label: Label, det: DetectionOutput) -> str | None:
    """The shelf region holding the label's centre, else the vertically nearest one."""
    cx = (label.bbox[0] + label.bbox[2]) / 2
    cy = (label.bbox[1] + label.bbox[3]) / 2
    best: tuple[float, str] | None = None
    for shelf in det.shelves:
        x1, y1, x2, y2 = shelf.bbox
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return shelf.shelf_id
        dist = max(y1 - cy, cy - y2, 0)
        if best is None or dist < best[0]:
            best = (dist, shelf.shelf_id)
    return best[1] if best else None


def _slot_left_edges(det: DetectionOutput) -> dict[str, float]:
    return {slot_id: x1 for slot_id, (x1, _) in _slot_spans(det).items()}


def _slot_spans(det: DetectionOutput) -> dict[str, tuple[float, float]]:
    """slot_id -> (left, right) of the items and gaps the vision stage put in it."""
    spans: dict[str, tuple[float, float]] = {}
    for obj in [*det.items, *det.gaps]:
        if obj.slot_id is not None:
            x1, x2 = spans.get(obj.slot_id, (obj.bbox[0], obj.bbox[2]))
            spans[obj.slot_id] = (min(x1, obj.bbox[0]), max(x2, obj.bbox[2]))
    return spans


def assign_labels_to_slots(
    labels: list[Label], det: DetectionOutput, planogram: Planogram
) -> list[Label]:
    """Copies of `labels` with shelf_id, slot_id and sku filled in.

    Each slot takes at most one label (the closest by left edge, else one whose left edge lies under the
    slot's detected products); a label that finds no free slot on its shelf keeps shelf_id but gets
    slot_id and sku None, which check_prices() reports as an unknown product.
    """
    slots = {s.slot_id: (shelf.shelf_id, s.sku) for shelf in planogram.shelves for s in shelf.slots}
    edges = _slot_left_edges(det)

    # A label further than half the tightest slot spacing on its shelf from a
    # slot's edge belongs to a neighbour or to nothing, not to this slot.
    tolerance: dict[str, float] = {}
    for shelf in planogram.shelves:
        xs = sorted(edges[s.slot_id] for s in shelf.slots if s.slot_id in edges)
        tolerance[shelf.shelf_id] = min((b - a for a, b in zip(xs, xs[1:])), default=float("inf")) / 2

    shelf_of = {lb.label_id: _shelf_for(lb, det) for lb in labels}
    pairs = sorted(
        (abs(lb.bbox[0] - edges[slot_id]), lb.label_id, slot_id)
        for lb in labels
        for slot_id, (shelf_id, _) in slots.items()
        if shelf_id == shelf_of[lb.label_id]
        and slot_id in edges
        and abs(lb.bbox[0] - edges[slot_id]) <= tolerance[shelf_id]
    )
    chosen: dict[str, str] = {}
    taken: set[str] = set()
    for _, label_id, slot_id in pairs:
        if label_id not in chosen and slot_id not in taken:
            chosen[label_id] = slot_id
            taken.add(slot_id)

    # Second pass: a label not near any slot's left edge but under a free slot's products belongs to it.
    # A slot that starts at the photo's edge has its true left edge out of frame, so its label sits
    # mid-slot (store1_36: a slot from x=0 to 732 with its label at x=249).
    spans = _slot_spans(det)
    under = sorted(
        (lb.bbox[0] - spans[slot_id][0], lb.label_id, slot_id)
        for lb in labels
        if lb.label_id not in chosen
        for slot_id, (shelf_id, _) in slots.items()
        if shelf_id == shelf_of[lb.label_id] and slot_id in spans and spans[slot_id][0] <= lb.bbox[0] < spans[slot_id][1]
    )
    for _, label_id, slot_id in under:
        if label_id not in chosen and slot_id not in taken:
            chosen[label_id] = slot_id
            taken.add(slot_id)

    out = []
    for lb in labels:
        slot_id = chosen.get(lb.label_id)
        out.append(
            lb.model_copy(
                update={
                    "shelf_id": slots[slot_id][0] if slot_id else shelf_of[lb.label_id],
                    "slot_id": slot_id,
                    "sku": slots[slot_id][1] if slot_id else None,
                }
            )
        )
    return out
