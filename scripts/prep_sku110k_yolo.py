"""Convert the downloaded SKU-110K parquet shards into a single-class YOLO dataset (class 0 = pack).

    python scripts/prep_sku110k_yolo.py

Needs pyarrow (pip install pyarrow). Reads data/sku110k_subset/{train,valid}-*.parquet and writes
data/sku110k_yolo/{images,labels}/{train,val}/ plus data.yaml. Boxes in the parquet are COCO
[x, y, width, height] in pixels, one per pack.
"""

import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "sku110k_subset"
OUT = ROOT / "data" / "sku110k_yolo"
SPLITS = {"train": "train-*.parquet", "val": "valid-*.parquet"}


def main() -> None:
    for split, pattern in SPLITS.items():
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_img = n_box = 0
        for shard in sorted(SRC.glob(pattern)):
            pf = pq.ParquetFile(shard)
            for g in range(pf.metadata.num_row_groups):
                for r in pf.read_row_group(g, columns=["image", "image_id", "width", "height", "objects"]).to_pylist():
                    name = f"{split}_{r['image_id']}"
                    im = Image.open(io.BytesIO(r["image"]["bytes"]))
                    w, h = im.size
                    lines = []
                    for x, y, bw, bh in r["objects"]["bbox"]:
                        x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            continue
                        lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                    (OUT / "images" / split / f"{name}.jpg").write_bytes(r["image"]["bytes"]) if im.format == "JPEG" else im.convert("RGB").save(OUT / "images" / split / f"{name}.jpg", quality=95)
                    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
                    n_img += 1
                    n_box += len(lines)
            print(f"  {shard.name}: running total {n_img} images", flush=True)
        print(f"{split}: {n_img} images, {n_box} boxes")
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: pack\n")


if __name__ == "__main__":
    main()
