"""Train the shelf-to-studio adapter for SKU matching and report held-out accuracy.

    python scripts/train_sku_adapter.py        # -> models/sku_adapter.npy

The adapter is a 384x384 linear map applied to crop embeddings (initialised to the
identity) so shelf crops land near the studio reference of the same product. It is
trained on stores 2, 3 and 5 only; stores 1 and 4 are held out for the numbers below.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from shelfsense.vision.grozi import TRAIN_STORES, VAL_STORES, labelled_boxes
from shelfsense.vision.sku_match import Embedder, ReferenceIndex, SkuMatcher, pack_crop

ROOT = Path(__file__).resolve().parents[1]
GROZI = ROOT / "data" / "grozi"
OUT = ROOT / "models" / "sku_adapter.npy"


def crops_for(stores: list[str], known: set[str]):
    by_img: dict[Path, list] = {}
    for img, sku, rel in labelled_boxes(GROZI, stores, known):
        by_img.setdefault(img, []).append((sku, rel))
    crops, skus, imgs = [], [], []
    for img, items in by_img.items():
        im = Image.open(img).convert("RGB")
        w, h = im.size
        for sku, (l, t, r, b) in items:
            crops.append(pack_crop(im, [l * w, t * h, r * w, b * h]))
            skus.append(sku)
            imgs.append(str(img))
    return crops, skus, imgs


def train_adapter(x: np.ndarray, targets: np.ndarray, refs: np.ndarray, steps: int = 300) -> np.ndarray:
    torch.manual_seed(0)
    dim = x.shape[1]
    eye = torch.eye(dim)
    w = torch.nn.Parameter(eye.clone())
    opt = torch.optim.AdamW([w], lr=1e-3, weight_decay=1e-2)
    xt, rt, tt = torch.from_numpy(x), torch.from_numpy(refs), torch.from_numpy(targets)
    for _ in range(steps):
        z = torch.nn.functional.normalize(xt @ w, dim=-1)
        loss = torch.nn.functional.cross_entropy(z @ rt.T / 0.05, tt) + 1e-2 * ((w - eye) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"adapter train loss {loss.item():.3f}")
    return w.detach().numpy()


def report(matcher: SkuMatcher, crops, truth, imgs, name: str) -> None:
    """Open-set and planogram-restricted (~25 candidates) accuracy, plus confidence calibration."""
    rng = random.Random(0)
    all_skus = list(matcher.index.unique_skus)
    by_img: dict[str, set[str]] = {}
    for im, t in zip(imgs, truth):
        by_img.setdefault(im, set()).add(t)

    open_set = matcher.match(crops)
    closed = [matcher.match([c], candidates=by_img[im] | set(rng.sample(all_skus, 20)))[0] for c, im in zip(crops, imgs)]
    acc_open = np.mean([m.sku == t for m, t in zip(open_set, truth)])
    acc_closed = np.mean([m.sku == t for m, t in zip(closed, truth)])
    print(f"{name:10s} open-set top-1 {acc_open:.2f} | planogram-restricted top-1 {acc_closed:.2f}")

    print("  planogram-restricted: confidence cutoff -> share kept, accuracy of kept")
    conf = np.array([m.confidence for m in closed])
    ok = np.array([m.sku == t for m, t in zip(closed, truth)])
    for cut in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        keep = conf >= cut
        if keep.any():
            print(f"    >= {cut:.1f}: kept {keep.mean():.2f}, accurate {ok[keep].mean():.2f}")


def main() -> None:
    index = ReferenceIndex.load(GROZI / "sku_index.npz")
    known = set(index.skus)
    embedder = Embedder()
    pos = {s: i for i, s in enumerate(index.unique_skus)}

    tr_crops, tr_skus, _ = crops_for(TRAIN_STORES, known)
    va_crops, va_skus, va_imgs = crops_for(VAL_STORES, known)
    print(f"train crops {len(tr_crops)} | held-out crops {len(va_crops)}")

    # per-product reference matrix aligned with index.unique_skus (best photo per product = first)
    first = {}
    for i, s in enumerate(index.skus):
        first.setdefault(s, i)
    refs = np.stack([index.embeddings[first[s]] for s in index.unique_skus])

    report(SkuMatcher(index, embedder), va_crops, va_skus, va_imgs, "frozen")

    x = embedder.embed(tr_crops)
    w = train_adapter(x, np.array([pos[s] for s in tr_skus]), refs)
    OUT.parent.mkdir(exist_ok=True)
    np.save(OUT, w)
    print("saved", OUT)

    report(SkuMatcher(index, embedder, adapter=w), va_crops, va_skus, va_imgs, "adapter")


if __name__ == "__main__":
    main()
