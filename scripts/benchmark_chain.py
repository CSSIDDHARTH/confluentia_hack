"""Time the whole planogram chain (blocks, naming, alignment, packs, empty spaces) on one photo.

    PYTHONPATH=src python3 scripts/benchmark_chain.py                                  # laptop GPU
    PYTHONPATH=src taskset -c 0-3 python3 scripts/benchmark_chain.py --device cpu --threads 4   # 4 CPU cores

Runs the five demo bays' original photos with their compliant plans: one warm-up pass (not counted), then
the median over the next passes. Same stand-in for an edge device as benchmark_edge.py (a CPU capped at a few
cores; no Jetson or phone was available).
"""

import argparse
import re
import statistics
from pathlib import Path

from shelfsense.contracts import load_planogram
from shelfsense.vision.live import LiveChain

ROOT = Path(__file__).resolve().parents[1]
BAYS = ROOT / "data" / "demo_bays"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cpu, or leave empty for the GPU")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    chain = LiveChain(a.device)
    jobs = []
    for bay in sorted(p for p in BAYS.iterdir() if (p / "detection_scenario.json").exists()):
        src = ROOT / re.search(r"(data/grozi/\S+\.jpg)", (bay / "README.txt").read_text()).group(1)
        jobs.append((src, load_planogram(bay / "planogram_compliant.json")))
    for src, pg in jobs:  # warm-up
        chain.run(src, pg)
    runs = [chain.run(src, pg) for _ in range(a.passes) for src, pg in jobs]
    print(f"device {chain.device}, {len(jobs)} photos x {a.passes} passes, input is the original 3264 px photo")
    for stage in runs[0].timings:
        print(f"  {stage:22s} median {1000 * statistics.median(r.timings[stage] for r in runs):7.0f} ms")
    print(f"  {'WHOLE CHAIN':22s} median {1000 * statistics.median(r.total for r in runs):7.0f} ms   (max {1000 * max(r.total for r in runs):.0f} ms)")


if __name__ == "__main__":
    main()
