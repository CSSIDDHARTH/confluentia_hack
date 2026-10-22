"""Convert data/price_tags/*.parquet (Open Food Facts price-tag detection,
CC BY-SA 4.0) into the folder layout Ultralytics YOLO trains from:

    data/price_tags/yolo/
        images/{train,val}/<image_id>.jpg
        labels/{train,val}/<image_id>.txt   one "0 cx cy w h" line per tag (relative)
        dataset.yaml

    python scripts/prep_price_tag_yolo.py

Reads the parquet in small batches, so it stays well under 1 GB of RAM.
"""

from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "price_tags"
OUT = SRC / "yolo"


def convert(split: str) -> int:
    images = OUT / "images" / split
    labels = OUT / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    n = 0
    parquet = pq.ParquetFile(SRC / f"{split}-00000-of-00001.parquet")
    for batch in parquet.iter_batches(batch_size=32, columns=["image_id", "image", "objects"]):
        for row in batch.to_pylist():
            image_id = row["image_id"]
            (images / f"{image_id}.jpg").write_bytes(row["image"]["bytes"])
            lines = []
            # bbox is [y_min, x_min, y_max, x_max], relative to the image
            for y1, x1, y2, x2 in row["objects"]["bbox"]:
                # a few source boxes stick out past the image; YOLO rejects those
                y1, x1, y2, x2 = (min(max(v, 0.0), 1.0) for v in (y1, x1, y2, x2))
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
            (labels / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
    return n


def main() -> None:
    for split in ("train", "val"):
        print(f"{split}: {convert(split)} images")
    (OUT / "dataset.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: price-tag\n"
    )
    print(f"wrote {(OUT / 'dataset.yaml').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
