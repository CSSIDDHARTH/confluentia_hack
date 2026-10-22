"""Money at stake per violation: an estimate for putting work in order, not accounting.

- stock-out / low stock: the packs the plan puts on display that are not there (planned facings minus the
  matching packs seen), times the catalogue price. That is the shelf value a shopper cannot pick up.
- price mismatch: the gap between label and catalogue price, times the planned facings: what the bay's packs
  would be over- or under-charged by if each sold at the label price.
- other kinds (misplaced, unreadable, unrecognised) are not priced.

No sales rates are used: Grozi has none and the catalogue prices are synthetic (assumption 15).
"""

from __future__ import annotations

from shelfsense.contracts import CatalogueItem, DetectionOutput, Planogram
from shelfsense.report.compliance import ComplianceReport, ViolationKind

K = ViolationKind


def at_stake(
    report: ComplianceReport,
    det: DetectionOutput,
    planogram: Planogram,
    catalogue: dict[str, CatalogueItem],
) -> dict[str, float]:
    """violation_id -> money at stake in the catalogue's currency, for the priced kinds only."""
    slots = {s.slot_id: s for shelf in planogram.shelves for s in shelf.slots}
    label_on = {lb.slot_id: lb for lb in det.labels if lb.slot_id is not None}
    out: dict[str, float] = {}
    for v in report.violations:
        item = catalogue.get(v.sku) if v.sku else None
        slot = slots.get(v.slot_id) if v.slot_id else None
        if item is None or slot is None:
            continue
        if v.kind in (K.STOCK_OUT, K.LOW_STOCK):
            seen = sum(i.slot_id == slot.slot_id and i.sku == slot.sku for i in det.items)
            out[v.violation_id] = round(max(slot.facings - seen, 0) * item.price, 2)
        elif v.kind == K.PRICE_MISMATCH and v.slot_id in label_on and label_on[v.slot_id].price is not None:
            out[v.violation_id] = round(abs(label_on[v.slot_id].price - item.price) * slot.facings, 2)
    return out


def currency(catalogue: dict[str, CatalogueItem]) -> str:
    """The catalogue's currency, or '' if it mixes several."""
    found = {c.currency for c in catalogue.values()}
    return found.pop() if len(found) == 1 else ""
