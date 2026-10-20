"""Shared data contracts between the vision pipeline (Person A) and the
pricing / report layer (Person B).

The example files in `contracts/` are the source of truth; these models
mirror them. Change both together, and tell your teammate first.

Conventions:
- bbox is [x1, y1, x2, y2] in absolute pixels of the original image,
  origin top-left.
- Shelf `index` 0 is the top shelf. Slot `position` 0 is the leftmost slot.
- Planograms are ordinal (shelf/slot order + facings), not metric.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "0.1"

BBox = list[float]


class _HasBBox(BaseModel):
    bbox: BBox

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: BBox) -> BBox:
        if len(v) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox must have x2 > x1 and y2 > y1, got {v}")
        return v


# ---------- Planogram ----------

class Slot(BaseModel):
    slot_id: str
    position: int = Field(ge=0)
    sku: str
    facings: int = Field(ge=1)
    min_facings: int = Field(ge=0)

    @model_validator(mode="after")
    def _min_le_facings(self) -> Slot:
        if self.min_facings > self.facings:
            raise ValueError(f"{self.slot_id}: min_facings > facings")
        return self


class Shelf(BaseModel):
    shelf_id: str
    index: int = Field(ge=0)
    slots: list[Slot]


class Planogram(BaseModel):
    schema_version: str = SCHEMA_VERSION
    planogram_id: str
    fixture_name: str
    effective_from: str
    shelves: list[Shelf]

    @model_validator(mode="after")
    def _unique_slot_ids(self) -> Planogram:
        ids = [s.slot_id for sh in self.shelves for s in sh.slots]
        if len(ids) != len(set(ids)):
            raise ValueError("slot_id values must be unique across the planogram")
        return self

    def slots_by_id(self) -> dict[str, Slot]:
        return {s.slot_id: s for sh in self.shelves for s in sh.slots}


# ---------- Catalogue ----------

class CatalogueItem(BaseModel):
    sku: str
    name: str
    brand: str
    category: str
    size: str
    price: float = Field(ge=0)
    currency: str


# ---------- Detection output (A -> B handoff) ----------

class ShelfRegion(_HasBBox):
    shelf_id: str


class Item(_HasBBox):
    item_id: str
    det_confidence: float = Field(ge=0, le=1)
    sku: str | None  # None = detected but not recognised
    sku_confidence: float = Field(ge=0, le=1)
    shelf_id: str | None
    slot_id: str | None  # slot the item physically occupies


class Gap(_HasBBox):
    gap_id: str
    shelf_id: str | None
    slot_id: str | None
    confidence: float = Field(ge=0, le=1)
    estimated_missing_facings: int = Field(ge=0)


class Label(_HasBBox):
    label_id: str
    shelf_id: str | None
    slot_id: str | None
    sku: str | None  # SKU the label refers to
    ocr_text: str
    price: float | None  # None = OCR could not parse a price
    ocr_confidence: float = Field(ge=0, le=1)


class DetectionOutput(BaseModel):
    """One shelf image. The vision stage fills shelves/items/gaps and leaves
    labels empty; the pricing stage fills labels."""

    schema_version: str = SCHEMA_VERSION
    image_id: str
    image_path: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    planogram_id: str
    captured_at: str
    produced_by: dict[str, str] = {}
    shelves: list[ShelfRegion] = []
    items: list[Item] = []
    gaps: list[Gap] = []
    labels: list[Label] = []


# ---------- Loaders ----------

def load_planogram(path: str | Path) -> Planogram:
    return Planogram.model_validate(json.loads(Path(path).read_text()))


def load_catalogue(path: str | Path) -> dict[str, CatalogueItem]:
    with open(path, newline="") as f:
        rows = [CatalogueItem.model_validate(r) for r in csv.DictReader(f)]
    return {r.sku: r for r in rows}


def load_detection_output(path: str | Path) -> DetectionOutput:
    return DetectionOutput.model_validate(json.loads(Path(path).read_text()))


def save_detection_output(out: DetectionOutput, path: str | Path) -> None:
    Path(path).write_text(out.model_dump_json(indent=2))
