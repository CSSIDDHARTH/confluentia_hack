"""Download datasets into data/ (git-ignored).

    python scripts/download_data.py grozi        # Grocery Products / Grozi-3.2K, 1.7 GB (needs: pip install py7zr)
    python scripts/download_data.py price_tags   # Open Food Facts price-tag detection, 0.4 GB
    python scripts/download_data.py sku110k      # SKU-110K subset (800 train + 200 val images), 1 GB
    python scripts/download_data.py sku110k_more # 800 more train images (optional, retrains a stronger pack detector), 0.8 GB
    python scripts/download_data.py slm          # Qwen2.5-1.5B-Instruct, 4-bit GGUF, 1.1 GB, into models/slm/
    python scripts/download_data.py easyocr      # EasyOCR's English models, 0.1 GB, into ~/.EasyOCR/model/ (second price reader)
    python scripts/download_data.py all

Archives are checked for absolute / '..' paths before extraction.
"""

import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

HF_PRICE_TAGS = "https://huggingface.co/datasets/openfoodfacts/price-tag-detection/resolve/main"
HF_SKU110K = "https://huggingface.co/datasets/adnankhan-11/smart-retail-shelf-auditing-v1/resolve/main/data"

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
    "sku110k": {
        "dir": "sku110k_subset",
        "files": {
            **{f"train-{i:05d}-of-00035.parquet": f"{HF_SKU110K}/train-{i:05d}-of-00035.parquet" for i in range(4)},
            "valid-00000-of-00010.parquet": f"{HF_SKU110K}/valid-00000-of-00010.parquet",
        },
        "licence": "SKU-110K (Goldman et al., CVPR 2019): academic and non-commercial use only; "
        "single class 'object' (one box per pack), mirrored on Hugging Face as adnankhan-11/smart-retail-shelf-auditing-v1",
    },
    "sku110k_more": {
        "dir": "sku110k_subset",
        "files": {f"train-{i:05d}-of-00035.parquet": f"{HF_SKU110K}/train-{i:05d}-of-00035.parquet" for i in range(4, 8)},
        "licence": "SKU-110K: academic and non-commercial use only (same dataset, shards 4-7)",
    },
    "slm": {
        "root": "models",
        "dir": "slm",
        "files": {
            "qwen2.5-1.5b-instruct-q4_k_m.gguf": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF"
            "/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        },
        "licence": "Apache 2.0 (Qwen2.5-1.5B-Instruct)",
    },
    "easyocr": {  # label_ocr.py never lets EasyOCR download at run time, so fetch its models here
        "root": str(Path.home() / ".EasyOCR"),
        "dir": "model",
        "files": {
            "craft_mlt_25k.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
            "english_g2.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
        },
        "licence": "Apache 2.0 (EasyOCR)",
    },
}


def download(url: str, dest: Path, attempts: int = 60) -> None:
    """curl with resume (-C -), retried until the transfer completes.

    Big files from Hugging Face / Drive are often cut mid-transfer (curl exit 18), so a
    single curl call is not enough; each retry carries on from the bytes already saved.
    --http1.1: HF's HTTP/2 streams for big files drop more often.
    """
    print(f"downloading {dest.name}")
    cmd = ["curl", "-sS", "-L", "--fail", "--http1.1", "-C", "-", "-o", str(dest), url]
    for attempt in range(1, attempts + 1):
        if subprocess.run(cmd).returncode == 0:
            return
        print(f"  {dest.name}: transfer cut off, resuming ({attempt}/{attempts})", flush=True)
        time.sleep(2)
    raise RuntimeError(f"could not finish downloading {url}")


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


def extract_zip(archive: Path, out: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        bad = [n for n in z.namelist() if n.startswith("/") or ".." in n.split("/")]
        if bad:
            sys.exit(f"refusing to extract {archive.name}: unsafe paths {bad[:3]}")
        z.extractall(out)


def main(name: str) -> None:
    spec = DATASETS[name]
    out = ROOT / spec.get("root", "data") / spec["dir"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] licence: {spec['licence']}")
    with ThreadPoolExecutor(max_workers=4) as pool:  # a few files at once: much faster than one stream
        list(pool.map(lambda fu: download(fu[1], out / fu[0]), spec["files"].items()))
    for filename in spec["files"]:
        dest = out / filename
        if dest.suffix == ".7z":
            print(f"extracting {dest.name}")
            extract_7z(dest, out)
            dest.unlink()  # keep disk usage down once extracted
        elif dest.suffix == ".zip":
            print(f"extracting {dest.name}")
            extract_zip(dest, out)
            dest.unlink()
    print(f"done: {out}")


if __name__ == "__main__":
    names = list(DATASETS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    if not names or any(n not in DATASETS for n in names):
        sys.exit(f"usage: download_data.py {{{'|'.join(DATASETS)}|all}}")
    for n in names:
        main(n)
