"""Step 6: evaluate detectors next to the baselines.

Reads:  runs/<name>/weights/best.pt, data/processed/split.csv, results/baselines.csv
Writes: results/yolo_<name>_patch.csv     patch-level kiln vs no kiln (same task as the baselines)
        results/yolo_<name>_detect.csv    detection-level mAP50, precision, recall, per class

Usage:
  python src/06_evaluate.py --name yolov8n_256_hardneg --splits val             (model choice)
  python src/06_evaluate.py --name yolov8n_256_hardneg --splits test_punjab test_kp kp_other sindh_baloch --conf <from val>

A patch counts as "kiln" when the model finds at least one box at or above --conf. When --conf is
not given, it is chosen on the validation split to maximise patch-level F1, never on a test split.
"""

import argparse

import numpy as np
import pandas as pd
from ultralytics import YOLO

from config import CLASS_NAMES, IMG_DIR, PROC, RESULTS, ROOT

YOLO_DIR = PROC / "yolo"


def max_conf_per_patch(model, names: list[str], imgsz: int) -> np.ndarray:
    out = np.zeros(len(names))
    paths = [str(IMG_DIR / n) for n in names]
    for i in range(0, len(paths), 256):
        for j, r in enumerate(model.predict(paths[i:i + 256], imgsz=imgsz, conf=0.01, verbose=False)):
            conf = r.obb.conf.cpu().numpy() if r.obb is not None else np.array([])
            out[i + j] = conf.max() if len(conf) else 0.0
    return out


def patch_scores(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"patches": len(y), "kiln_patches": int(y.sum()), "accuracy": float((pred == y).mean()),
            "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "missed_kiln_patches": fn, "false_alarm_patches": fp}


def detection_scores(model, split_file: str, imgsz: int, name: str) -> list[dict]:
    yaml = YOLO_DIR / f"_eval_{split_file}.yaml"
    lines = [f"path: {YOLO_DIR.as_posix()}", "train: train.txt", f"val: {split_file}.txt", "names:"]
    yaml.write_text("\n".join(lines + [f"  {i}: {n}" for i, n in CLASS_NAMES.items()]) + "\n")
    m = model.val(data=str(yaml), imgsz=imgsz, batch=64, workers=2, plots=True, verbose=False,
                  project=str(ROOT / "runs" / "eval"), name=f"{name}_{split_file}", exist_ok=True)
    rows = [{"class": "all", "precision": m.box.mp, "recall": m.box.mr, "mAP50": m.box.map50, "mAP50_95": m.box.map}]
    for k, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(k)
        rows.append({"class": CLASS_NAMES[int(c)], "precision": p, "recall": r, "mAP50": ap50, "mAP50_95": ap})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=256)
    args = ap.parse_args()

    model = YOLO(str(ROOT / "runs" / args.name / "weights" / "best.pt"))
    df = pd.read_csv(PROC / "split.csv")
    df["y"] = (df["n_kilns"] > 0).astype(int)

    if args.conf is None:
        val = df[df["split"] == "val"]
        s = max_conf_per_patch(model, val["image_name"].tolist(), args.imgsz)
        grid = np.round(np.arange(0.05, 0.95, 0.05), 2)
        f1s = [patch_scores(val["y"].to_numpy(), (s >= t).astype(int))["f1"] for t in grid]
        args.conf = float(grid[int(np.argmax(f1s))])
        print(f"patch threshold chosen on val: conf >= {args.conf} (val F1 {max(f1s):.3f})")

    patch_rows, det_rows = [], []
    for split in args.splits:
        sub = df[df["split"] == split]
        s = max_conf_per_patch(model, sub["image_name"].tolist(), args.imgsz)
        pred = (s >= args.conf).astype(int)
        urban = sub["image_name"].str.startswith("negu_").to_numpy()
        if urban.any():  # urban negatives only: every alarm here is a false alarm on a town or city
            patch_rows.append({"model": args.name, "split": split, "source": "urban_negatives", "conf": args.conf,
                               **patch_scores(sub["y"].to_numpy()[urban], pred[urban])})
        for source in ["skdb", "pc", "all"]:
            m = np.ones(len(sub), bool) if source == "all" else (sub["source"] == source).to_numpy()
            if m.any():
                patch_rows.append({"model": args.name, "split": split, "source": source, "conf": args.conf,
                                   **patch_scores(sub["y"].to_numpy()[m], pred[m])})
            if (YOLO_DIR / f"{split}{'' if source == 'all' else '_' + source}.txt").stat().st_size and m.any():
                split_file = split if source == "all" else f"{split}_{source}"
                for r in detection_scores(model, split_file, args.imgsz, args.name):
                    det_rows.append({"model": args.name, "split": split, "source": source, **r})

    RESULTS.mkdir(exist_ok=True)
    tag = "val" if args.splits == ["val"] else "test"
    patch = pd.DataFrame(patch_rows)
    det = pd.DataFrame(det_rows)
    patch.to_csv(RESULTS / f"yolo_{args.name}_{tag}_patch.csv", index=False)
    det.to_csv(RESULTS / f"yolo_{args.name}_{tag}_detect.csv", index=False)
    print("\nPatch level (kiln vs no kiln):\n", patch.round(3).to_string(index=False))
    print("\nDetection level:\n", det.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
