"""Phase 2 vision stage: identify the product in a box by nearest-neighbour
search against studio reference photos (Grozi `Training/`).

Embeddings come from a frozen DINOv2 ViT-S/14 (no training), compared by
cosine similarity. Grozi SKU ids look like "G09-114": class index (1-based, from
TrainingClassesIndex.mat) and the product label within that class.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HUB_CACHE = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sku_id(class_idx: int, label: int) -> str:
    return f"G{class_idx:02d}-{label}"


def class_index(grozi_dir: str | Path) -> dict[int, str]:
    """Grozi class number (1-based) -> category path, e.g. 9 -> "Food/Biscuits".

    Read from Training/TrainingClassesIndex.mat once and cached as classes.json.
    The .mat is read in an isolated interpreter (-I) so a user-site scipy/numpy
    mismatch cannot break it.
    """
    grozi_dir = Path(grozi_dir)
    cache = grozi_dir / "classes.json"
    if not cache.exists():
        code = (
            "import json,sys,scipy.io as s;"
            "m=s.loadmat(sys.argv[1],simplify_cells=True);"
            "print(json.dumps({int(i):str(c) for c,i in zip(m['classes'],m['indices'])}))"
        )
        out = subprocess.run(
            [sys.executable, "-I", "-c", code, str(grozi_dir / "Training" / "TrainingClassesIndex.mat")],
            capture_output=True, text=True, check=True,
        ).stdout
        cache.write_text(out)
    return {int(k): v for k, v in json.loads(cache.read_text()).items()}


class Embedder:
    def __init__(self, device: str | None = None, size: int = 224):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        if HUB_CACHE.exists():  # avoid a network round-trip when already cached
            model = torch.hub.load(str(HUB_CACHE), "dinov2_vits14", source="local")
        else:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        self.model = model.to(self.device).eval()

    def _prep(self, im: Image.Image) -> np.ndarray:
        im = im.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        return ((np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)

    @torch.no_grad()
    def embed(self, images: list[Image.Image], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch):
            x = torch.from_numpy(np.stack([self._prep(im) for im in images[i : i + batch]])).to(self.device)
            e = self.model(x)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


class ReferenceIndex:
    """Reference embeddings plus the product each one belongs to."""

    def __init__(self, embeddings: np.ndarray, skus: list[str], paths: list[str]):
        self.embeddings = embeddings.astype(np.float32)
        self.skus = skus
        self.paths = paths

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, embeddings=self.embeddings, skus=np.array(self.skus), paths=np.array(self.paths))

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceIndex":
        d = np.load(path, allow_pickle=False)
        return cls(d["embeddings"], d["skus"].tolist(), d["paths"].tolist())

    def query(self, embeddings: np.ndarray, k: int = 5) -> list[list[tuple[str, float]]]:
        """Top-k distinct SKUs per query, best first, as (sku, cosine similarity)."""
        sims = embeddings @ self.embeddings.T
        results = []
        for row in sims:
            best: dict[str, float] = {}
            for j in np.argsort(-row)[: k * 6]:
                s = self.skus[j]
                if s not in best:
                    best[s] = float(row[j])
                if len(best) == k:
                    break
            results.append(list(best.items()))
        return results
