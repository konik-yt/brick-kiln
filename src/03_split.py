"""Step 3: split patches by district, never at random.

Reads:  data/processed/patches.csv
Writes: data/processed/split.csv                 (image_name, split)
        data/processed/yolo/<split>.txt          (image paths, one per line, for Ultralytics)
        data/processed/yolo/kilns.yaml           (Ultralytics dataset config)

Splits
- train         Punjab, all districts not held out below
- val           4 whole Punjab districts, used for early stopping and model choice
- test_punjab   4 other whole Punjab districts, the in-region test
- test_kp       Peshawar valley (Peshawar, Charsadda, Mardan, Nowshera, Swabi), the main test
- kp_other      the rest of KP, an extra out-of-region check
- sindh_baloch  Sindh and Balochistan, an extra out-of-region check

Patches overlap by 30 pixels, so neighbouring patches on either side of a district border share pixels.
When two patches from different splits overlap, the one in the less strict split is dropped
(strictness: test_kp > test_punjab > kp_other > sindh_baloch > val > train).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
YOLO = PROC / "yolo"

VAL_DISTRICTS = ["Gujranwala1", "Multan", "Bahawalnagar", "Chakwal"]
TEST_PUNJAB_DISTRICTS = ["Sheikhupura", "Sahiwal", "RahimyarKhan", "Jhelum"]
TEST_KP_DISTRICTS = ["Peshawar", "Charsadda", "Mardan", "Nowshera", "Swabi"]

STRICTNESS = ["test_kp", "test_punjab", "kp_other", "sindh_baloch", "val", "train"]
PATCH_M = 128 * 10  # patch width in metres; two patches overlap when both centre offsets are below this
CLASS_NAMES = ["CFCBK", "FCBK", "Zigzag"]


def assign_split(row) -> str:
    if row.district in TEST_KP_DISTRICTS:
        return "test_kp"
    if row.province == "Khyber-Pakhtunkhwa":
        return "kp_other"
    if row.province in ("Sindh", "Balochistan"):
        return "sindh_baloch"
    if row.district in TEST_PUNJAB_DISTRICTS:
        return "test_punjab"
    if row.district in VAL_DISTRICTS:
        return "val"
    return "train"


def drop_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any patch that overlaps a patch from a stricter split."""
    pts = gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326").to_crs("EPSG:32643")
    xy = np.column_stack([pts.x, pts.y])
    rank = df["split"].map(STRICTNESS.index).to_numpy()
    tree = cKDTree(xy)
    drop = np.zeros(len(df), dtype=bool)
    # Chebyshev distance (p=inf): squares overlap when both |dx| and |dy| are below the patch width.
    for i, j in tree.query_pairs(r=PATCH_M - 1e-6, p=np.inf):
        if rank[i] != rank[j]:
            drop[i if rank[i] > rank[j] else j] = True
    out = df.copy()
    out.loc[drop, "split"] = "dropped_overlap"
    return out


def write_yolo_lists(df: pd.DataFrame) -> None:
    YOLO.mkdir(parents=True, exist_ok=True)
    img_dir = PROC / "patches" / "images"
    for split in STRICTNESS:
        names = df.loc[df["split"] == split, "image_name"]
        (YOLO / f"{split}.txt").write_text("".join(f"{(img_dir / n).as_posix()}\n" for n in names))
    yaml = [f"path: {YOLO.as_posix()}", "train: train.txt", "val: val.txt", "test: test_kp.txt", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    (YOLO / "kilns.yaml").write_text("\n".join(yaml) + "\n")


def main() -> None:
    df = pd.read_csv(PROC / "patches.csv")
    for d in VAL_DISTRICTS + TEST_PUNJAB_DISTRICTS + TEST_KP_DISTRICTS:
        assert d in set(df["district"]), f"unknown district {d}"
    df["split"] = df.apply(assign_split, axis=1)
    df = drop_overlaps(df)

    df[["image_name", "split"]].to_csv(PROC / "split.csv", index=False)
    write_yolo_lists(df)

    summary = df.groupby("split").agg(
        patches=("image_name", "size"),
        empty=("n_kilns", lambda x: int((x == 0).sum())),
        kilns=("n_kilns", "sum"),
        FCBK=("n_FCBK", "sum"),
        Zigzag=("n_Zigzag", "sum"),
        CFCBK=("n_CFCBK", "sum"),
    )
    summary = summary.reindex(STRICTNESS[::-1] + ["dropped_overlap"])
    summary["pct_kilns"] = (100 * summary["kilns"] / summary["kilns"].sum()).round(1)
    print(summary.to_string())


if __name__ == "__main__":
    main()
