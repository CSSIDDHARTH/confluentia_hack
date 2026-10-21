"""Helpers for reading Grozi's test-shelf annotations (data/grozi/Testing)."""

import csv
from collections.abc import Iterator
from pathlib import Path

from shelfsense.vision.sku_match import sku_id

# Stores 1 and 4 are held out of every training step (detector and adapter).
TRAIN_STORES = ["store2", "store3", "store5"]
VAL_STORES = ["store1", "store4"]


def labelled_boxes(
    grozi_dir: str | Path, stores: list[str], known: set[str] | None = None
) -> Iterator[tuple[Path, str, list[float]]]:
    """Yield (image path, sku, [left, top, right, bottom] relative 0-1) for every
    labelled box. Boxes whose product is not in the reference set (label -1) are
    skipped, as are products not in `known` when it is given."""
    grozi_dir = Path(grozi_dir)
    for store in stores:
        for ann in sorted((grozi_dir / "Testing" / store / "annotation").glob("*.csv")):
            img = grozi_dir / "Testing" / store / "images" / f"{ann.stem}.jpg"
            with open(ann, newline="") as f:
                for r in csv.DictReader(f):
                    if r["label"] == "-1":
                        continue
                    sku = sku_id(int(r["class"]), int(r["label"]))
                    if known is not None and sku not in known:
                        continue
                    yield img, sku, [float(r[k]) for k in ("left_x", "top_y", "right_x", "bottom_y")]
