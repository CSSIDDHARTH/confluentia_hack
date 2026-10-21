"""Turn a compliance report's violation list into a short, ordered list of things to do.

Pure rules, no model: a swap of two products is four violations (a stock-out and a misplaced item per
slot) but one job, and a wrong product sitting in a slot that expects another is one job too. Getting
this merge right matters, so it is exact code, and the language model only rewords the result.
"""

from dataclasses import dataclass, field

from shelfsense.contracts import CatalogueItem
from shelfsense.report.compliance import WEIGHTS, ComplianceReport, Violation, ViolationKind

K = ViolationKind


@dataclass
class Step:
    text: str
    severity: float
    violation_ids: list[str] = field(default_factory=list)


def _name(catalogue: dict[str, CatalogueItem], sku: str | None) -> str:
    item = catalogue.get(sku) if sku else None
    return item.name if item else (sku or "unknown product")


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


def plan(report: ComplianceReport, catalogue: dict[str, CatalogueItem]) -> list[Step]:
    """Most severe step first. Every violation ends up in exactly one step."""
    by_slot: dict[tuple[K, str | None], list[Violation]] = {}
    for v in report.violations:
        by_slot.setdefault((v.kind, v.slot_id), []).append(v)
    used: set[str] = set()
    steps: list[Step] = []

    def take(*vs: Violation) -> list[str]:
        ids = [v.violation_id for v in vs]
        used.update(ids)
        return ids

    # a swap: the wrong product P in slot X is what slot Y expects, and Y's wrong product Q is what X expects
    for m1 in report.violations:
        if m1.kind != K.MISPLACED or m1.violation_id in used:
            continue
        for s_y in (v for (k, _), vs in by_slot.items() if k == K.STOCK_OUT for v in vs if v.sku == m1.sku and v.slot_id != m1.slot_id):
            for m2 in by_slot.get((K.MISPLACED, s_y.slot_id), []):
                s_x = next((v for v in by_slot.get((K.STOCK_OUT, m1.slot_id), []) if v.sku == m2.sku), None)
                if s_x is None or {m1.violation_id, m2.violation_id, s_x.violation_id, s_y.violation_id} & used:
                    continue
                text = f"Swap {_name(catalogue, m1.sku)} (now in {m1.slot_id}) with {_name(catalogue, m2.sku)} (now in {s_y.slot_id}): each belongs in the other's slot."
                steps.append(Step(text, max(WEIGHTS[K.STOCK_OUT], WEIGHTS[K.MISPLACED]), take(m1, m2, s_x, s_y)))
                break

    # a slot that holds the wrong product and also lacks the right one
    for (kind, slot), vs in list(by_slot.items()):
        if kind != K.STOCK_OUT:
            continue
        for s in vs:
            if s.violation_id in used:
                continue
            wrong = [m for m in by_slot.get((K.MISPLACED, slot), []) if m.violation_id not in used]
            if wrong:
                text = " ".join(m.recommended_action for m in wrong) + " Then " + _lower_first(s.recommended_action)
                steps.append(Step(text, WEIGHTS[K.STOCK_OUT] + WEIGHTS[K.MISPLACED], take(s, *wrong)))

    for v in report.violations:
        if v.violation_id not in used:
            steps.append(Step(v.recommended_action, WEIGHTS.get(v.kind, 0), take(v)))

    return sorted(steps, key=lambda s: -s.severity)
