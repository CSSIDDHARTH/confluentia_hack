"""Edit a planogram the way a person would at the shelf: change what a slot expects, how many facings, or swap two.

The page keeps the plan as plain rows (one per slot) so it can show them in an editable table;
these helpers turn a Planogram into rows, swap two slots' products, and build the edited Planogram back.
The photo reading does not change, only what the plan expects, so the score moves because of the edit.
"""

from shelfsense.contracts import Planogram

Row = dict  # {"slot": "S1-01", "sku": "G13-88", "facings": 2, "minimum": 1}


def plan_rows(planogram: Planogram) -> list[Row]:
    return [
        {"slot": s.slot_id, "sku": s.sku, "facings": s.facings, "minimum": s.min_facings}
        for shelf in planogram.shelves for s in shelf.slots
    ]


def swap_products(rows: list[Row], slot_a: str, slot_b: str) -> list[Row]:
    """The two slots exchange their expected product (facings stay with the slot)."""
    out = [dict(r) for r in rows]
    a = next(r for r in out if r["slot"] == slot_a)
    b = next(r for r in out if r["slot"] == slot_b)
    a["sku"], b["sku"] = b["sku"], a["sku"]
    return out


def apply_rows(planogram: Planogram, rows: list[Row]) -> Planogram:
    """A copy of the plan with each slot's product and facings taken from `rows`.

    Facings are at least 1 and the minimum is kept between 1 and the facings, so an edit cannot make an invalid plan."""
    edited = planogram.model_copy(deep=True)
    by_slot = {r["slot"]: r for r in rows}
    for shelf in edited.shelves:
        for slot in shelf.slots:
            r = by_slot[slot.slot_id]
            slot.sku = r["sku"]
            slot.facings = max(1, int(r["facings"]))
            slot.min_facings = min(slot.facings, max(1, int(r["minimum"])))
    return edited


def changes(original: list[Row], rows: list[Row]) -> list[str]:
    """One plain line per slot the person changed."""
    was = {r["slot"]: r for r in original}
    out = []
    for r in rows:
        w = was[r["slot"]]
        if r["sku"] != w["sku"]:
            out.append(f"{r['slot']}: now expects {r['sku']} (was {w['sku']})")
        if r["facings"] != w["facings"] or r["minimum"] != w["minimum"]:
            out.append(f"{r['slot']}: facings {w['facings']} to {r['facings']}, minimum {w['minimum']} to {r['minimum']}")
    return out
