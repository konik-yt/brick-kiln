"""Step 9: collect wrong predictions of the final model for the failure gallery.

Reads:  models/ or runs/ weights, data/processed/split.csv, patch images and labels
Writes: results/failures/<kind>_<n>.png      one picture per failure (green = label, red = prediction)
        results/failures/candidates.csv      every failure found, for picking and labelling reasons

Kinds
- false_alarm   empty patch (hard or urban negative) where the model draws a box   (test_kp, test_punjab)
- missed_kiln   kiln patch where the model draws no box                           (test_kp)
- wrong_type    labelled Zigzag kiln found but typed FCBK                         (test_punjab)

The reasons are written by hand after looking at the pictures (results/failure_gallery.md).
"""

import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO

from config import CLASS_NAMES, FINAL_CONF, FINAL_IMGSZ, FINAL_MODEL, IMG_DIR, LBL_DIR, PROC, RESULTS, weights_path

OUT = RESULTS / "failures"
SCALE = 3


def labels(name: str) -> list[tuple[int, np.ndarray]]:
    f = LBL_DIR / name.replace(".png", ".txt")
    if not f.exists():
        return []
    rows = []
    for line in f.read_text().split("\n"):
        if line.strip():
            v = line.split()
            rows.append((int(v[0]), np.array(v[1:], dtype=float).reshape(4, 2)))
    return rows


def draw(name: str, gt, pred, title: str, path) -> None:
    im = Image.open(IMG_DIR / name).convert("RGB")
    size = im.width * SCALE
    im = im.resize((size, size), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for _, poly in gt:
        d.polygon([tuple(p * size) for p in poly], outline=(0, 255, 0), width=2)
    for _, poly, _ in pred:
        d.polygon([tuple(p * size) for p in poly], outline=(255, 0, 0), width=2)
    d.rectangle([0, 0, size, 14], fill=(0, 0, 0))
    d.text((3, 2), title, fill=(255, 255, 0))
    im.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=FINAL_MODEL)
    ap.add_argument("--imgsz", type=int, default=FINAL_IMGSZ)
    ap.add_argument("--conf", type=float, default=FINAL_CONF)
    args = ap.parse_args()

    model = YOLO(str(weights_path(args.name)))
    df = pd.read_csv(PROC / "split.csv")
    df = df[df["split"].isin(["test_kp", "test_punjab"])].reset_index(drop=True)

    preds = {}
    names = df["image_name"].tolist()
    for i in range(0, len(names), 256):
        batch = names[i:i + 256]
        for n, r in zip(batch, model.predict([str(IMG_DIR / n) for n in batch], imgsz=args.imgsz, conf=args.conf, verbose=False)):
            ob = r.obb
            preds[n] = [(int(c), p, float(s)) for c, p, s in zip(ob.cls, ob.xyxyxyxyn.cpu().numpy(), ob.conf)] if ob is not None else []

    rows = []
    for r in df.itertuples():
        gt, pr = labels(r.image_name), preds[r.image_name]
        if r.n_kilns == 0 and pr:
            rows.append({"kind": "false_alarm", "image_name": r.image_name, "split": r.split, "source": r.source,
                         "district": r.district, "score": max(p[2] for p in pr), "boxes": len(pr)})
        elif r.split == "test_kp" and r.n_kilns > 0 and not pr:
            rows.append({"kind": "missed_kiln", "image_name": r.image_name, "split": r.split, "source": r.source,
                         "district": r.district, "score": float(r.n_kilns), "boxes": 0})
        elif r.split == "test_punjab" and r.n_Zigzag > 0 and pr:
            # a Zigzag label whose nearest prediction (centre within 4 px) says FCBK
            for c, poly in gt:
                if CLASS_NAMES[c] != "Zigzag":
                    continue
                centre = poly.mean(0)
                near = [p for p in pr if np.abs(p[1].mean(0) - centre).max() < 4 / 128]
                if near and all(CLASS_NAMES[p[0]] == "FCBK" for p in near):
                    rows.append({"kind": "wrong_type", "image_name": r.image_name, "split": r.split, "source": r.source,
                                 "district": r.district, "score": max(p[2] for p in near), "boxes": len(pr)})
                    break

    cand = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    print(cand.groupby(["kind", "split", "source"]).size().to_string())

    # Pick a spread: most confident false alarms (urban and rural), missed kilns from both sources, wrong types.
    picks = pd.concat([
        cand[(cand.kind == "false_alarm") & cand.image_name.str.startswith("negu_")].nlargest(3, "score"),
        cand[(cand.kind == "false_alarm") & cand.image_name.str.startswith("neg_")].nlargest(3, "score"),
        cand[(cand.kind == "missed_kiln") & (cand.source == "skdb")].nlargest(3, "score"),
        cand[(cand.kind == "missed_kiln") & (cand.source == "pc")].nlargest(3, "score"),
        cand[cand.kind == "wrong_type"].nlargest(3, "score"),
    ]).reset_index(drop=True)
    picks["file"] = ""
    for i, p in picks.iterrows():
        f = OUT / f"{p.kind}_{i:02d}.png"
        title = f"{p.kind} {p.district} {p.source} " + (f"conf {p.score:.2f}" if p.kind != "missed_kiln" else f"{int(p.score)} kilns")
        draw(p.image_name, labels(p.image_name), preds[p.image_name], title, f)
        picks.loc[i, "file"] = f.name
    cand.to_csv(OUT / "candidates.csv", index=False)
    picks.to_csv(OUT / "picked.csv", index=False)

    sheet = [np.asarray(Image.open(OUT / f)) for f in picks["file"]]
    pad = [np.pad(s, ((3, 3), (3, 3), (0, 0)), constant_values=255) for s in sheet]
    rows_ = [np.hstack(pad[k:k + 5]) for k in range(0, len(pad), 5)]
    width = max(r.shape[1] for r in rows_)
    rows_ = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows_]
    Image.fromarray(np.vstack(rows_)).save(RESULTS / "failure_gallery.png")
    print(f"{len(picks)} failures drawn to {OUT} and results/failure_gallery.png")


if __name__ == "__main__":
    main()
