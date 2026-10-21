"""Fine-tune the product-block detector on data/grozi_yolo (run prep_grozi_yolo.py first).

    python scripts/train_detector.py [--model models/yolo11s.pt] [--epochs 60] [--imgsz 960] [--name grozi_v1]

Best weights land in outputs/runs/<name>/weights/best.pt.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "yolo11s.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "grozi_yolo" / "data.yaml"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="grozi_v1")
    ap.add_argument("--optimizer", default="auto", help="e.g. AdamW; auto lets Ultralytics choose (SGD for long runs)")
    ap.add_argument("--lr0", type=float, default=0.01, help="initial learning rate (ignored when optimizer is auto)")
    a = ap.parse_args()

    YOLO(a.model).train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        patience=15,
        optimizer=a.optimizer,
        lr0=a.lr0,
        workers=8,
        flipud=0.0,  # shelves have a fixed "up"
        project=str(ROOT / "outputs" / "runs"),
        name=a.name,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
