"""Time the pack detector the way a small shelf-edge device would run it, and size the models.

    PYTHONPATH= taskset -c 0-3 .venv/bin/python scripts/benchmark_edge.py --threads 4

taskset pins the process to 4 cores so the ONNX run is capped as well as PyTorch. Needs onnx and
onnxruntime (in the project .venv). Compares, per image, median latency of:
  gpu     PyTorch FP16 on the laptop GPU (the demo setting)
  cpu     PyTorch on CPU limited to --threads threads
  onnx    the exported ONNX model on CPU (cap it with taskset, as above)

This is a stand-in for an edge device (no Jetson or phone was available): a CPU capped at a few
threads is a conservative lower bound for what such a device can do. It times the pack detector, the
step the empty-space page needs. The product-block detector is the same network size.
"""

import argparse
import statistics
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def median_ms(fn, images, warmup=1, repeats=2) -> float:
    for img in images[:warmup]:
        fn(img)
    times = []
    for img in images:
        for _ in range(repeats):
            t = time.perf_counter()
            fn(img)
            times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "pack_detector.pt"))
    ap.add_argument("--images", default=str(ROOT / "data" / "test_images"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[1024, 640])
    a = ap.parse_args()

    images = sorted(str(p) for p in Path(a.images).glob("*.jpg"))
    weights = Path(a.weights)
    print(f"{len(images)} images, {a.threads} CPU threads, weights {weights.name} ({weights.stat().st_size / 1e6:.1f} MB)")

    for size in a.imgsz:
        print(f"\n-- input {size}px --")
        onnx_path = Path(YOLO(str(weights)).export(format="onnx", imgsz=size, simplify=False, verbose=False))
        print(f"onnx file: {onnx_path.stat().st_size / 1e6:.1f} MB")

        if torch.cuda.is_available():
            gpu = YOLO(str(weights))
            ms = median_ms(lambda p: gpu.predict(p, imgsz=size, conf=0.25, device=0, half=True, max_det=600, verbose=False), images)
            print(f"gpu  (PyTorch fp16):      {ms:7.0f} ms/image")

        torch.set_num_threads(a.threads)
        cpu = YOLO(str(weights))
        ms = median_ms(lambda p: cpu.predict(p, imgsz=size, conf=0.25, device="cpu", max_det=600, verbose=False), images)
        print(f"cpu  (PyTorch, {a.threads} threads):  {ms:7.0f} ms/image")

        # a fixed-size export, so ultralytics' own wrapper handles pre/post-processing
        onnx_model = YOLO(str(onnx_path), task="detect")
        ort_cpu = lambda p: onnx_model.predict(p, imgsz=size, conf=0.25, device="cpu", max_det=600, verbose=False)  # noqa: E731
        ms = median_ms(ort_cpu, images)
        print(f"onnx (onnxruntime CPU):   {ms:7.0f} ms/image")


if __name__ == "__main__":
    main()
