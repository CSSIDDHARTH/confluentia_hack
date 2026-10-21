"""Embed every Grozi studio reference photo once and save the index.

    python scripts/build_sku_index.py            # -> data/grozi/sku_index.npz
"""

from pathlib import Path

import numpy as np
from PIL import Image

from shelfsense.vision.sku_match import Embedder, ReferenceIndex, class_index, sku_id

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "grozi" / "Training"
OUT = ROOT / "data" / "grozi" / "sku_index.npz"


def main() -> None:
    paths, skus = [], []
    for idx, name in class_index(TRAIN.parent).items():
        if name == "Background":
            continue
        for f in sorted((TRAIN / name).iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg"} and f.stem.isdigit():
                paths.append(str(f.relative_to(TRAIN)))
                skus.append(sku_id(int(idx), int(f.stem)))
    print(f"{len(paths)} reference images, {len(set(skus))} products")

    emb = Embedder()
    vecs = []
    for i in range(0, len(paths), 256):
        vecs.append(emb.embed([Image.open(TRAIN / p) for p in paths[i : i + 256]]))
        print(f"  {min(i + 256, len(paths))}/{len(paths)}", flush=True)
    ReferenceIndex(np.concatenate(vecs), skus, paths).save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
