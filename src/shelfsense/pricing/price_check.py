"""Price-check: compare shelf-edge label prices (from the detection output)
against the catalogue, and flag mismatches, unreadable prices and labels
with no matching product.

Depends only on `shelfsense.contracts` (shared) — safe to extend freely.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from shelfsense.contracts import CatalogueItem, DetectionOutput

# Prices are stored as floats (dollars); treat anything below a cent as
# floating-point noise rather than a real mismatch.
PRICE_TOLERANCE = 0.005

# OCR reads agreeing on fewer than this share of variants are often right but
# sometimes wrong (see label_ocr.read_label_price), so a mismatch built on one
# is still reported, flagged for a human to verify.
LOW_CONFIDENCE = 0.5


class PriceIssueType(str, Enum):
    PRICE_MISMATCH = "price_mismatch"       # label price != catalogue price
    UNREADABLE_PRICE = "unreadable_price"   # OCR could not parse a price
    UNKNOWN_PRODUCT = "unknown_product"     # label has no SKU, or an unknown one


class PriceIssue(BaseModel):
    label_id: str
    shelf_id: str | None
    slot_id: str | None
    sku: str | None
    ocr_text: str
    issue_type: PriceIssueType
    label_price: float | None
    catalogue_price: float | None
    difference: float | None  # label_price - catalogue_price, when both are known
    detail: str
    low_confidence: bool = False  # price_mismatch resting on a shaky OCR read


def check_prices(
    det: DetectionOutput, catalogue: dict[str, CatalogueItem]
) -> list[PriceIssue]:
    """Return one PriceIssue per label that fails a price check.

    Checks, per label, in this order:
    1. unreadable_price — OCR produced no parsed price.
    2. unknown_product  — label has no SKU, or a SKU absent from the catalogue.
    3. price_mismatch   — label price differs from the catalogue price.
    Labels that pass all three produce no issue.
    """
    issues: list[PriceIssue] = []

    for label in det.labels:
        cat_item = catalogue.get(label.sku) if label.sku else None

        if label.price is None:
            issues.append(
                PriceIssue(
                    label_id=label.label_id,
                    shelf_id=label.shelf_id,
                    slot_id=label.slot_id,
                    sku=label.sku,
                    ocr_text=label.ocr_text,
                    issue_type=PriceIssueType.UNREADABLE_PRICE,
                    label_price=None,
                    catalogue_price=cat_item.price if cat_item else None,
                    difference=None,
                    detail=(
                        f"Label {label.label_id!r}: OCR text {label.ocr_text!r} "
                        "could not be parsed as a price."
                    ),
                )
            )
            continue

        if cat_item is None:
            reason = (
                "label is not linked to any SKU"
                if label.sku is None
                else f"SKU {label.sku!r} is not in the catalogue"
            )
            issues.append(
                PriceIssue(
                    label_id=label.label_id,
                    shelf_id=label.shelf_id,
                    slot_id=label.slot_id,
                    sku=label.sku,
                    ocr_text=label.ocr_text,
                    issue_type=PriceIssueType.UNKNOWN_PRODUCT,
                    label_price=label.price,
                    catalogue_price=None,
                    difference=None,
                    detail=f"Label {label.label_id!r}: {reason}.",
                )
            )
            continue

        difference = round(label.price - cat_item.price, 2)
        if abs(difference) >= PRICE_TOLERANCE:
            shaky = label.ocr_confidence < LOW_CONFIDENCE
            issues.append(
                PriceIssue(
                    label_id=label.label_id,
                    shelf_id=label.shelf_id,
                    slot_id=label.slot_id,
                    sku=label.sku,
                    ocr_text=label.ocr_text,
                    issue_type=PriceIssueType.PRICE_MISMATCH,
                    label_price=label.price,
                    catalogue_price=cat_item.price,
                    difference=difference,
                    detail=(
                        f"Label {label.label_id!r} shows {label.price:.2f} but the "
                        f"catalogue price for {label.sku} is {cat_item.price:.2f}."
                        + (" Low OCR confidence: verify the label by eye." if shaky else "")
                    ),
                    low_confidence=shaky,
                )
            )

    return issues
