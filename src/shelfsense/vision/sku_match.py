"""Phase 2 vision stage: identify the product in a box by nearest-neighbour
search against studio reference photos (Grozi `Training/`).

Embeddings come from a frozen DINOv2 ViT-S/14 (no training), compared by
cosine similarity. Grozi SKU ids look like "G09-114": class index (1-based, from
TrainingClassesIndex.mat) and the product label within that class.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
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
        # group references by product so a product with several photos scores by its best one
        self.unique_skus, inv = np.unique(np.array(skus), return_inverse=True)
        self._order = np.argsort(inv, kind="stable")
        self._starts = np.r_[0, np.flatnonzero(np.diff(inv[self._order])) + 1]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, embeddings=self.embeddings, skus=np.array(self.skus), paths=np.array(self.paths))

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceIndex":
        d = np.load(path, allow_pickle=False)
        return cls(d["embeddings"], d["skus"].tolist(), d["paths"].tolist())

    def scores(self, embeddings: np.ndarray, candidates: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Cosine similarity of each query to each product (best reference photo per
        product). Returns (skus, scores) with scores shaped (n_queries, n_skus);
        restricted to `candidates` when given."""
        sims = embeddings @ self.embeddings.T
        best = np.maximum.reduceat(sims[:, self._order], self._starts, axis=1)
        if candidates is None:
            return self.unique_skus, best
        keep = np.isin(self.unique_skus, list(candidates))
        return self.unique_skus[keep], best[:, keep]

    def query(
        self, embeddings: np.ndarray, k: int = 5, candidates: set[str] | None = None
    ) -> list[list[tuple[str, float]]]:
        """Top-k products per query, best first, as (sku, cosine similarity)."""
        skus, best = self.scores(embeddings, candidates)
        k = min(k, len(skus))
        top = np.argsort(-best, axis=1)[:, :k]
        return [[(str(skus[j]), float(row[j])) for j in idx] for row, idx in zip(best, top)]


def pack_crop(im: Image.Image, box: list[float]) -> Image.Image:
    """Crop a product-block box ([x1, y1, x2, y2] pixels) down to roughly one pack.

    A block is a row of identical packs but the reference photo shows one, so the
    centre pack (about 0.75 x box height wide) matches better than the whole block.
    """
    w, h = im.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pack_w = min(bw, bh * 0.75)
    x1 = x1 + (bw - pack_w) / 2
    return im.crop((max(0, int(x1)), max(0, int(y1)), min(w, int(x1 + pack_w)), min(h, int(y2))))


@dataclass
class Match:
    sku: str
    confidence: float  # softmax probability of the top product among the candidates
    similarity: float  # raw cosine similarity of the top product
    alternatives: list[tuple[str, float]]  # runner-ups as (sku, similarity)


class SkuMatcher:
    """Embed crops (optionally through a trained adapter) and match them to products."""

    def __init__(
        self,
        index: ReferenceIndex,
        embedder=None,
        adapter: np.ndarray | None = None,
        temperature: float = 0.05,
    ):
        self.index = index
        self.embedder = embedder if embedder is not None else Embedder()
        self.adapter = adapter
        self.temperature = temperature

    @classmethod
    def load(cls, index_path: str | Path, adapter_path: str | Path | None = None, **kw) -> "SkuMatcher":
        adapter = None
        if adapter_path is not None and Path(adapter_path).exists():
            adapter = np.load(adapter_path, allow_pickle=False)
        return cls(ReferenceIndex.load(index_path), adapter=adapter, **kw)

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        e = self.embedder.embed(images)
        if self.adapter is not None:
            e = e @ self.adapter
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
        return e

    def match(self, images: list[Image.Image], candidates: set[str] | None = None, k: int = 3) -> list[Match]:
        if not images:
            return []
        skus, best = self.index.scores(self.embed(images), candidates)
        z = best / self.temperature
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        out = []
        for row, prob in zip(best, p):
            order = np.argsort(-row)[: max(k, 1)]
            top = order[0]
            out.append(
                Match(
                    sku=str(skus[top]),
                    confidence=float(prob[top]),
                    similarity=float(row[top]),
                    alternatives=[(str(skus[j]), float(row[j])) for j in order[1:]],
                )
            )
        return out
