"""Audit one shelf photo from the command line, as a shelf-edge device would: no dashboard, and no GPU needed.

    PYTHONPATH=src python3 scripts/audit.py data/demo_bays/store4_9 --plan scenario        # a demo bay
    PYTHONPATH=src python3 scripts/audit.py shelf.jpg --planogram plan.json --catalogue prices.csv --out report.json
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 0-3 python3 scripts/audit.py ... --cpu   # edge stand-in: 4 cores

Reads the photo end to end (product blocks, naming, plan alignment, packs, empty spaces, and the price labels
when tesseract is installed), prints the score, the value at stake and the steps for staff, and writes one JSON
with the report, steps, value at stake per violation, the reading itself and the seconds per stage: what a
device would send to the store's server.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="a demo bay folder (data/demo_bays/<bay>) or a shelf photo")
    ap.add_argument("--plan", choices=["compliant", "scenario"], default="compliant", help="demo bay only: which planogram")
    ap.add_argument("--planogram", type=Path, help="photo only: its planogram JSON")
    ap.add_argument("--catalogue", type=Path, help="photo only: the price list CSV")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU only")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for the models")
    ap.add_argument("--no-prices", action="store_true", help="skip the price labels")
    ap.add_argument("--out", type=Path, help="where to write the JSON (default: outputs/<image>_audit.json)")
    a = ap.parse_args()
    if a.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # EasyOCR picks its own device, so hide the GPU from it too

    import torch

    from shelfsense.advisor.plan import plan as plan_steps
    from shelfsense.contracts import load_catalogue, load_planogram
    from shelfsense.report.compliance import compute_report
    from shelfsense.report.impact import at_stake, currency
    from shelfsense.vision import live

    if a.source.is_dir():
        photo, _ = live.bay_photo(a.source)
        planogram = load_planogram(a.source / f"planogram_{a.plan}.json")
        cat_file = a.source / ("catalogue_scenario.csv" if a.plan == "scenario" and (a.source / "catalogue_scenario.csv").exists() else "catalogue.csv")
        catalogue = load_catalogue(cat_file)
    elif a.planogram and a.catalogue:
        photo, planogram, catalogue = a.source, load_planogram(a.planogram), load_catalogue(a.catalogue)
    else:
        sys.exit("a photo needs --planogram and --catalogue")
    if not live.available():
        sys.exit("needs the Grozi product index (scripts/build_sku_index.py) and the three detectors in models/")
    read_prices = not a.no_prices and live.prices_available()

    torch.set_num_threads(a.threads)
    chain = live.LiveChain("cpu" if a.cpu else None)
    result = chain.run(photo, planogram, read_prices=read_prices)
    det = result.det
    report = compute_report(det, planogram, catalogue, group_items=True)
    steps = plan_steps(report, catalogue, planogram)
    stake = at_stake(report, det, planogram, catalogue)
    money = currency(catalogue)

    print(f"{photo.name}: score {report.score:.0f}/100, value at stake {money} {sum(stake.values()):.2f} (estimate), "
          f"{len(steps)} steps, read in {result.total:.2f} s on {result.device}"
          + ("" if read_prices else " (price labels not read)"))
    for k, step in enumerate(steps, 1):
        value = sum(stake.get(v, 0.0) for v in step.violation_ids)
        print(f"  {k}. {step.text}" + (f"  [{money} {value:.2f} at stake]" if value else ""))

    out = a.out or ROOT / "outputs" / f"{Path(det.image_id).stem}_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "report": report.model_dump(mode="json"),
        "steps": [{"text": s.text, "violation_ids": s.violation_ids} for s in steps],
        "at_stake": stake,
        "currency": money,
        "timings_s": {**result.timings, "total": result.total},
        "device": result.device,
        "prices_read": read_prices,
        "detection": det.model_dump(mode="json"),
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
