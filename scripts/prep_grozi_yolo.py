"""Convert Grozi test shelves into a single-class YOLO dataset.

    python scripts/prep_grozi_yolo.py

Writes data/grozi_yolo/{images,labels}/{train,val}/ (images are symlinks) and
data/grozi_yolo/data.yaml. Class 0 = "product": one box per product block (all
facings of a product), as annotated in Grozi. SKU identity is phase 2, so the
label/class columns are ignored here; boxes with label -1 (product not in the
reference set) are kept because they are still products.

Split is by store, not random: several photos show the same shelf from slightly
different angles, so a random split would leak. Stores 1 and 4 are validation.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "grozi" / "Testing"
OUT = ROOT / "data" / "grozi_yolo"
VAL_STORES = {"store1", "store4"}


def main() -> None:
    counts = {"train": [0, 0], "val": [0, 0]}  # images, boxes
    skipped = 0
    for store_dir in sorted(SRC.glob("store*")):
        split = "val" if store_dir.name in VAL_STORES else "train"
        for sub in ("images", "labels"):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)
        for img in sorted((store_dir / "images").glob("*.jpg")):
            ann = store_dir / "annotation" / f"{img.stem}.csv"
            if not ann.exists():  # one Grozi image has no annotation file
                skipped += 1
                continue
            with open(ann, newline="") as f:
                rows = list(csv.DictReader(f))
            lines = []
            for r in rows:
                x1, x2, y1, y2 = (min(max(float(r[k]), 0.0), 1.0) for k in ("left_x", "right_x", "top_y", "bottom_y"))
                if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
                    continue
                lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
            name = f"{store_dir.name}_{img.stem}"
            link = OUT / "images" / split / f"{name}.jpg"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(img.resolve())
            (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
            counts[split][0] += 1
            counts[split][1] += len(lines)

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: product\n"
    )
    for split, (n_img, n_box) in counts.items():
        print(f"{split}: {n_img} images, {n_box} boxes")
    print(f"skipped {skipped} images without annotations")


if __name__ == "__main__":
    main()
