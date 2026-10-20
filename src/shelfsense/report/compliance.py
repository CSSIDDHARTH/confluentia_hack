"""Compliance scoring and corrective-action report.

Combines the planogram, detection output and catalogue into one report:
a list of violations (each with a recommended corrective action) and an
overall compliance score for the image.

Violation categories, per CLAUDE.md's assumptions:
- stock_out / low_stock: assumption 5 (matching facings == 0 / < min_facings).
- misplaced: an item with a recognised SKU sitting in a slot that expects
  a different SKU (CLAUDE.md's "misplaced products").
- unrecognised_item: a detected item the vision stage could not identify
  (sku is None) occupying a slot.
- price_mismatch / unreadable_price / unknown_product: from
  `shelfsense.pricing.price_check` (assumption 6).

Scoring assumption (new -- not yet in CLAUDE.md; raise with teammate
before adding it there): start at 100 and subtract a fixed weight per
violation, floored at 0. Weights in WEIGHTS are a first pass, meant to be
tuned once there's feedback on how harshly each category should count.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from shelfsense.contracts import CatalogueItem, DetectionOutput, Planogram
from shelfsense.pricing.price_check import PriceIssueType, check_prices


class ViolationKind(str, Enum):
    STOCK_OUT = "stock_out"
    LOW_STOCK = "low_stock"
    MISPLACED = "misplaced"
    UNRECOGNISED_ITEM = "unrecognised_item"
    PRICE_MISMATCH = "price_mismatch"
    UNREADABLE_PRICE = "unreadable_price"
    UNKNOWN_PRODUCT = "unknown_product"


WEIGHTS: dict[ViolationKind, float] = {
    ViolationKind.STOCK_OUT: 15,
    ViolationKind.LOW_STOCK: 7,
    ViolationKind.MISPLACED: 5,
    ViolationKind.UNRECOGNISED_ITEM: 2,
    ViolationKind.PRICE_MISMATCH: 5,
    ViolationKind.UNREADABLE_PRICE: 2,
    ViolationKind.UNKNOWN_PRODUCT: 3,
}


class Violation(BaseModel):
    violation_id: str
    kind: ViolationKind
    shelf_id: str | None
    slot_id: str | None
    sku: str | None
    detail: str
    recommended_action: str


class ComplianceReport(BaseModel):
    image_id: str
    planogram_id: str
    score: float
    violations: list[Violation]
    counts: dict[str, int]


def _product_name(catalogue: dict[str, CatalogueItem], sku: str | None) -> str:
    if sku is None:
        return "unknown product"
    item = catalogue.get(sku)
    return item.name if item else sku


def _slot_violations(
    det: DetectionOutput, planogram: Planogram, catalogue: dict[str, CatalogueItem]
) -> list[Violation]:
    violations: list[Violation] = []

    for shelf in planogram.shelves:
        for slot in shelf.slots:
            in_slot = [i for i in det.items if i.slot_id == slot.slot_id]
            matching = [i for i in in_slot if i.sku == slot.sku]
            misplaced = [i for i in in_slot if i.sku is not None and i.sku != slot.sku]
            unrecognised = [i for i in in_slot if i.sku is None]

            expected_name = _product_name(catalogue, slot.sku)
            matching_facings = len(matching)

            if matching_facings == 0:
                violations.append(
                    Violation(
                        violation_id=f"v-{slot.slot_id}-stock_out",
                        kind=ViolationKind.STOCK_OUT,
                        shelf_id=shelf.shelf_id,
                        slot_id=slot.slot_id,
                        sku=slot.sku,
                        detail=(
                            f"Slot {slot.slot_id} expects {slot.sku} "
                            f"({expected_name}) but none are present."
                        ),
                        recommended_action=(
                            f"Restock {expected_name} at {slot.slot_id} "
                            f"({slot.facings} facings expected)."
                        ),
                    )
                )
            elif matching_facings < slot.min_facings:
                violations.append(
                    Violation(
                        violation_id=f"v-{slot.slot_id}-low_stock",
                        kind=ViolationKind.LOW_STOCK,
                        shelf_id=shelf.shelf_id,
                        slot_id=slot.slot_id,
                        sku=slot.sku,
                        detail=(
                            f"Slot {slot.slot_id} has {matching_facings} facing(s) "
                            f"of {slot.sku}, below the minimum of {slot.min_facings}."
                        ),
                        recommended_action=(
                            f"Replenish {expected_name} at {slot.slot_id} to at "
                            f"least {slot.min_facings} facing(s)."
                        ),
                    )
                )

            for item in misplaced:
                found_name = _product_name(catalogue, item.sku)
                violations.append(
                    Violation(
                        violation_id=f"v-{item.item_id}-misplaced",
                        kind=ViolationKind.MISPLACED,
                        shelf_id=shelf.shelf_id,
                        slot_id=slot.slot_id,
                        sku=item.sku,
                        detail=(
                            f"Item {item.item_id} ({item.sku}, {found_name}) found "
                            f"in slot {slot.slot_id}, which expects {slot.sku}."
                        ),
                        recommended_action=(
                            f"Move {found_name} out of {slot.slot_id}; that slot "
                            f"should hold {expected_name}."
                        ),
                    )
                )

            for item in unrecognised:
                violations.append(
                    Violation(
                        violation_id=f"v-{item.item_id}-unrecognised",
                        kind=ViolationKind.UNRECOGNISED_ITEM,
                        shelf_id=shelf.shelf_id,
                        slot_id=slot.slot_id,
                        sku=None,
                        detail=(
                            f"Item {item.item_id} in slot {slot.slot_id} could not "
                            "be identified."
                        ),
                        recommended_action=(
                            f"Manually verify the product at {slot.slot_id}; the "
                            "detector could not recognise its SKU."
                        ),
                    )
                )

    return violations


def _price_violations(
    det: DetectionOutput, catalogue: dict[str, CatalogueItem]
) -> list[Violation]:
    violations: list[Violation] = []

    for issue in check_prices(det, catalogue):
        if issue.issue_type == PriceIssueType.PRICE_MISMATCH:
            kind = ViolationKind.PRICE_MISMATCH
            action = (
                f"Reprint label {issue.label_id}: shows {issue.label_price:.2f}, "
                f"catalogue price is {issue.catalogue_price:.2f}."
            )
            if issue.low_confidence:
                action = (
                    f"Check label {issue.label_id} by eye (low OCR confidence); "
                    f"if it shows {issue.label_price:.2f}, reprint it at "
                    f"{issue.catalogue_price:.2f}."
                )
        elif issue.issue_type == PriceIssueType.UNREADABLE_PRICE:
            kind = ViolationKind.UNREADABLE_PRICE
            action = f"Inspect label {issue.label_id} in person; OCR could not read a price."
        else:
            kind = ViolationKind.UNKNOWN_PRODUCT
            action = f"Check label {issue.label_id}: it does not match a known catalogue SKU."

        violations.append(
            Violation(
                violation_id=f"v-{issue.label_id}-{issue.issue_type.value}",
                kind=kind,
                shelf_id=issue.shelf_id,
                slot_id=issue.slot_id,
                sku=issue.sku,
                detail=issue.detail,
                recommended_action=action,
            )
        )

    return violations


def compute_report(
    det: DetectionOutput, planogram: Planogram, catalogue: dict[str, CatalogueItem]
) -> ComplianceReport:
    """Build the full compliance report for one image."""
    violations = _slot_violations(det, planogram, catalogue) + _price_violations(det, catalogue)

    penalty = sum(WEIGHTS[v.kind] for v in violations)
    score = round(max(0.0, 100.0 - penalty), 1)

    counts: dict[str, int] = {}
    for v in violations:
        counts[v.kind.value] = counts.get(v.kind.value, 0) + 1

    return ComplianceReport(
        image_id=det.image_id,
        planogram_id=det.planogram_id,
        score=score,
        violations=violations,
        counts=counts,
    )
