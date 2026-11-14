"""Step 5: train the YOLOv8-OBB kiln detector.

Reads:  data/processed/yolo/kilns.yaml (step 3)
Writes: runs/<name>/weights/best.pt, runs/<name>/results.csv

Usage:  python src/05_train.py --model yolov8n-obb.pt --imgsz 256 --name yolov8n_256_hardneg

Kilns are about 3 x 8 pixels in a 128 x 128 patch, so we upscale to 256.
Satellite images have no "up", so vertical flips are added to the default horizontal flips.
workers=2 because more dataloader workers can hang on Windows.
"""

import argparse

from ultralytics import YOLO

from config import PROC, ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-obb.pt")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    YOLO(args.model).train(
        data=str(PROC / "yolo" / "kilns.yaml"),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=-1,
        workers=2,
        cache="ram",
        flipud=0.5,
        seed=0,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
