"""Regenerate tests/fixtures/grozi_labels/ from data/grozi/ (Grozi-3.2K,
academic use only -- see data/README.md). Crops a hand-picked set of
shelf-edge price labels so label_ocr.py has real supermarket labels to be
measured against, without checking the 2 GB dataset into git.

    python scripts/prep_grozi_label_fixtures.py

Grozi annotates product blocks, not labels, so each box below was marked
by hand on the shelf photo and each price transcribed by eye from the
crop. Labels span all five stores (different lighting, distance and
label printers); a few are cut off by the image edge on purpose, as long
as the whole price is still visible.
"""

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "grozi" / "Testing"
OUT = ROOT / "tests" / "fixtures" / "grozi_labels"

# (store, image, label bbox [x1, y1, x2, y2] in original pixels, price as printed)
SELECTION = [
    (1, "132", [76, 1102, 449, 1238], "2.90"),
    (1, "132", [1006, 1115, 1360, 1246], "4.50"),
    (1, "132", [2263, 1129, 2625, 1251], "2.40"),
    (1, "132", [2910, 1129, 3264, 1251], "2.40"),
    (1, "132", [413, 2244, 789, 2380], "3.10"),
    (1, "132", [1523, 2244, 1890, 2380], "3.70"),
    (1, "132", [2502, 2252, 2870, 2380], "3.10"),
    (1, "132", [857, 0, 1210, 109], "2.90"),
    (1, "132", [2217, 27, 2557, 122], "2.90"),
    (1, "133", [544, 1156, 870, 1265], "2.40"),
    (1, "133", [1510, 1047, 1958, 1183], "2.90"),
    (1, "133", [0, 2198, 231, 2317], "2.70"),
    (1, "133", [830, 2149, 1142, 2271], "2.60"),
    (1, "133", [2040, 2108, 2366, 2217], "3.10"),
    (1, "133", [0, 3142, 326, 3264], "1.60"),
    (1, "133", [911, 3142, 1129, 3264], "1.60"),
    (1, "133", [1871, 3128, 2217, 3250], "1.90"),
    (2, "10", [1126, 881, 1681, 1077], "2.70"),
    (2, "10", [2742, 881, 3264, 1077], "2.50"),
    (2, "10", [180, 2220, 702, 2415], "1.90"),
    (2, "10", [1812, 2220, 2366, 2415], "1.70"),
    (2, "50", [0, 1567, 483, 1795], "3.95"),
    (2, "50", [979, 1583, 1632, 1795], "2.80"),
    (3, "1", [0, 180, 424, 310], "2.40"),
    (3, "1", [2122, 196, 2481, 320], "3.95"),
    (3, "1", [0, 1175, 326, 1279], "1.95"),
    (3, "1", [2268, 1142, 2660, 1279], "1.95"),
    (4, "1", [0, 1061, 310, 1224], "8.60"),
    (4, "1", [1420, 1061, 1812, 1224], "8.10"),
    (4, "1", [2905, 1077, 3264, 1234], "3.90"),
    (5, "103", [979, 490, 2056, 865], "1.45"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    counts: dict[str, int] = {}
    for store, image, bbox, truth in SELECTION:
        stem = f"s{store}_{image}"
        idx = counts.get(stem, 0)
        counts[stem] = idx + 1
        im = Image.open(RAW / f"store{store}" / "images" / f"{image}.jpg").convert("RGB")
        out_name = f"{stem}_{idx}.jpg"
        im.crop(tuple(bbox)).save(OUT / out_name, quality=92)
        manifest[out_name] = truth

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest)} crops + manifest.json to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
