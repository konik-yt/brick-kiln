"""Step 3: split patches by district, never at random.

Reads:  data/processed/patches.csv          (SentinelKilnDB patches, step 1)
        data/processed/extra/*.csv          (hard negatives and re-cut kiln patches, step 2)
Writes: data/processed/split.csv            (one row per patch: location, counts, source, split)
        data/processed/yolo/<split>.txt     (image paths for Ultralytics)
        data/processed/yolo/kilns.yaml      (Ultralytics dataset config)

Splits (district lists live in config.py)
- train         Punjab, all districts not held out below
- val           4 whole Punjab districts, used for early stopping and model choice
- test_punjab   4 other whole Punjab districts, the in-region test
- test_kp       Peshawar valley (Peshawar, Charsadda, Mardan, Nowshera, Swabi), the main test
- kp_other      the rest of KP, an extra out-of-region check
- sindh_baloch  Sindh and Balochistan, an extra out-of-region check

Sources
- skdb  original SentinelKilnDB patch (Earth Engine export, authors' dates)
- pc    our patch cut from Planetary Computer, Nov 2023 to Feb 2024 (hard negative or re-cut kiln patch)

Patches overlap by 30 pixels, so patches on either side of a district border can share pixels.
When two patches from different splits overlap, the one in the less strict split is dropped.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

from config import CLASS_NAMES, IMG_DIR, PROC, assign_split

YOLO = PROC / "yolo"
STRICTNESS = ["test_kp", "test_punjab", "kp_other", "sindh_baloch", "val", "train"]
PATCH_M = 128 * 10  # two patches overlap when both centre offsets are below one patch width


def load_all() -> pd.DataFrame:
    df = pd.read_csv(PROC / "patches.csv")
    df["source"] = "skdb"
    extra = sorted((PROC / "extra").glob("*.csv"))
    if extra:
        ex = pd.concat([pd.read_csv(p) for p in extra], ignore_index=True)
        ex["source"] = "pc"
        df = pd.concat([df, ex.drop(columns=["scene"], errors="ignore")], ignore_index=True)
    return df


def drop_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any patch that overlaps a patch from a stricter split."""
    pts = gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326").to_crs("EPSG:32643")
    xy = np.column_stack([pts.x, pts.y])
    rank = df["split"].map(STRICTNESS.index).to_numpy()
    drop = np.zeros(len(df), dtype=bool)
    # Chebyshev distance (p=inf): squares overlap when both |dx| and |dy| are below the patch width.
    for i, j in cKDTree(xy).query_pairs(r=PATCH_M - 1e-6, p=np.inf):
        if rank[i] != rank[j]:
            drop[i if rank[i] > rank[j] else j] = True
    out = df.copy()
    out.loc[drop, "split"] = "dropped_overlap"
    return out


def write_yolo_lists(df: pd.DataFrame) -> None:
    YOLO.mkdir(parents=True, exist_ok=True)
    for split in STRICTNESS:
        for source in ["all", "skdb", "pc"]:
            m = df["split"] == split
            if source != "all":
                m &= df["source"] == source
            suffix = "" if source == "all" else f"_{source}"
            (YOLO / f"{split}{suffix}.txt").write_text("".join(f"{(IMG_DIR / n).as_posix()}\n" for n in df.loc[m, "image_name"]))
    yaml = [f"path: {YOLO.as_posix()}", "train: train.txt", "val: val.txt", "test: test_kp.txt", "names:"]
    yaml += [f"  {i}: {n}" for i, n in CLASS_NAMES.items()]
    (YOLO / "kilns.yaml").write_text("\n".join(yaml) + "\n")


def main() -> None:
    df = load_all()
    df["split"] = [assign_split(p, d) for p, d in zip(df["province"], df["district"])]
    df = drop_overlaps(df)
    df.to_csv(PROC / "split.csv", index=False)
    write_yolo_lists(df)

    summary = df.groupby(["split", "source"]).agg(
        patches=("image_name", "size"),
        empty=("n_kilns", lambda x: int((x == 0).sum())),
        kilns=("n_kilns", "sum"),
        FCBK=("n_FCBK", "sum"),
        Zigzag=("n_Zigzag", "sum"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
