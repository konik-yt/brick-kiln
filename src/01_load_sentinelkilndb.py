"""Step 1+2: extract the Pakistan patches from SentinelKilnDB and tag each with its district.

Reads:  data/raw/sentinelkilndb/{train,val,test}.parquet   (Hugging Face: SustainabilityLabIITGN/SentinelKilnDB)
        data/raw/gadm/gadm41_PAK_3.json                     (GADM 4.1, Pakistan level 3 = districts)
Writes: data/processed/patches/images/<lat>_<lon>.png
        data/processed/patches/labels/<lat>_<lon>.txt       (YOLO-OBB, only for patches with kilns)
        data/processed/patches.csv                          (one row per patch: location, district, kiln counts)

Notes
- Patch names are the patch CENTER as "<lat>_<lon>", rounded to 4 decimals
  (see data_scripts/tile_processing.py in rishabh-mondal/SENTINELKILNDB_NeurIPS_2025).
- Class ids in the labels: 0 = CFCBK, 1 = FCBK, 2 = Zigzag (same source, label_format_coversion.py).
- The Hugging Face train/val/test split is random, so we pool all three and re-split by district in step 3.
- GADM level 2 for Pakistan is divisions, so we use level 3 for districts.
"""

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from config import CLASS_NAMES, IMG_DIR, LBL_DIR, RAW
from config import PROC as OUT

HF_SPLITS = ["train", "val", "test"]


def load_index() -> pd.DataFrame:
    """Patch names and labels from all splits, without the image bytes."""
    frames = []
    for split in HF_SPLITS:
        t = pq.read_table(RAW / "sentinelkilndb" / f"{split}.parquet", columns=["image_name", "yolo_obb_label"])
        df = t.to_pandas()
        df["hf_split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    latlon = df["image_name"].str.removesuffix(".png").str.split("_", expand=True).astype(float)
    df["lat"], df["lon"] = latlon[0], latlon[1]
    return df


def attach_district(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only patches whose center falls inside a Pakistan district."""
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    districts = gpd.read_file(RAW / "gadm" / "gadm41_PAK_3.json")[["NAME_1", "NAME_3", "geometry"]]
    joined = gpd.sjoin(points, districts, how="inner", predicate="within")
    joined = joined.rename(columns={"NAME_1": "province", "NAME_3": "district"})
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


def count_classes(df: pd.DataFrame) -> pd.DataFrame:
    df["n_kilns"] = df["yolo_obb_label"].map(len)
    for cid, name in CLASS_NAMES.items():
        df[f"n_{name}"] = df["yolo_obb_label"].map(lambda rows, c=str(cid): sum(r.split()[0] == c for r in rows))
    return df


def write_patches(keep: pd.DataFrame) -> None:
    """Stream each parquet in batches and write images + label files for the kept patches."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(keep["image_name"])
    written = 0
    for split in HF_SPLITS:
        pf = pq.ParquetFile(RAW / "sentinelkilndb" / f"{split}.parquet")
        for batch in pf.iter_batches(batch_size=2000, columns=["image_name", "image", "yolo_obb_label"]):
            for name, img, labels in zip(*(batch.column(c).to_pylist() for c in ["image_name", "image", "yolo_obb_label"])):
                if name not in wanted:
                    continue
                (IMG_DIR / name).write_bytes(img)
                if labels:
                    (LBL_DIR / name.replace(".png", ".txt")).write_text("\n".join(labels) + "\n")
                written += 1
    print(f"wrote {written} images to {IMG_DIR}")


def main() -> None:
    df = load_index()
    print(f"SentinelKilnDB patches (all countries): {len(df)}")
    pak = count_classes(attach_district(df))
    print(f"Pakistan patches: {len(pak)}")

    write_patches(pak)
    cols = ["image_name", "lat", "lon", "province", "district", "hf_split", "n_kilns"] + [f"n_{n}" for n in CLASS_NAMES.values()]
    pak[cols].sort_values(["province", "district", "image_name"]).to_csv(OUT / "patches.csv", index=False)

    kiln_cols = ["n_kilns"] + [f"n_{n}" for n in CLASS_NAMES.values()]
    by_prov = pak.groupby("province").agg(patches=("image_name", "size"), empty=("n_kilns", lambda x: int((x == 0).sum())))
    by_prov = by_prov.join(pak.groupby("province")[kiln_cols].sum())
    print("\nPer province:\n", by_prov.to_string())
    kp = pak[pak["province"] == "Khyber-Pakhtunkhwa"]
    by_kp = kp.groupby("district").agg(patches=("image_name", "size")).join(kp.groupby("district")[kiln_cols].sum())
    print("\nKP districts:\n", by_kp.sort_values("n_kilns", ascending=False).to_string())


if __name__ == "__main__":
    main()
