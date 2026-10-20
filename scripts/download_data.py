"""Download datasets into data/ (git-ignored).

    python scripts/download_data.py grozi        # Grocery Products / Grozi-3.2K, 1.7 GB (needs: pip install py7zr)
    python scripts/download_data.py price_tags   # Open Food Facts price-tag detection, 0.4 GB
    python scripts/download_data.py all

Archives are checked for absolute / '..' paths before extraction.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

HF_PRICE_TAGS = "https://huggingface.co/datasets/openfoodfacts/price-tag-detection/resolve/main"

DATASETS = {
    "grozi": {
        "dir": "grozi",
        "files": {
            "Grocery_products.7z": "https://drive.usercontent.google.com/download"
            "?id=1vvB1hvKhr4pE8zUpPImlogA6kof-kLVN&export=download&confirm=t",
        },
        "licence": "Academic research only (George & Floerkemeier, ECCV 2014)",
    },
    "price_tags": {
        "dir": "price_tags",
        "files": {
            "train-00000-of-00001.parquet": f"{HF_PRICE_TAGS}/data/train-00000-of-00001.parquet",
            "val-00000-of-00001.parquet": f"{HF_PRICE_TAGS}/data/val-00000-of-00001.parquet",
            "README.md": f"{HF_PRICE_TAGS}/README.md",
        },
        "licence": "CC BY-SA 4.0 (Open Food Facts / Open Prices)",
    },
}


def download(url: str, dest: Path) -> None:
    # curl is much faster than urllib here and resumes interrupted downloads (-C -).
    print(f"downloading {dest.name}")
    subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(dest), url], check=True)


def extract_7z(archive: Path, out: Path) -> None:
    import py7zr

    with py7zr.SevenZipFile(archive) as z:
        bad = [n for n in z.getnames() if n.startswith("/") or ".." in n.split("/")]
        if bad:
            sys.exit(f"refusing to extract {archive.name}: unsafe paths {bad[:3]}")
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(out)
    for junk in out.rglob(".DS_Store"):
        junk.unlink()


def main(name: str) -> None:
    spec = DATASETS[name]
    out = DATA / spec["dir"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] licence: {spec['licence']}")
    for filename, url in spec["files"].items():
        dest = out / filename
        download(url, dest)
        if dest.suffix == ".7z":
            print(f"extracting {dest.name}")
            extract_7z(dest, out)
            dest.unlink()  # keep disk usage down once extracted
    print(f"done: {out}")


if __name__ == "__main__":
    names = list(DATASETS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    if not names or any(n not in DATASETS for n in names):
        sys.exit(f"usage: download_data.py {{{'|'.join(DATASETS)}|all}}")
    for n in names:
        main(n)
