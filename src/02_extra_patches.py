"""Step 2: cut extra patches from Sentinel-2 (Planetary Computer) with the SentinelKilnDB recipe.

Why: in SentinelKilnDB the empty patches are mostly south-Punjab desert and KP has none at all,
so "kiln vs no kiln" can be solved by telling desert from farmland. We add:
- hard negatives  ("neg_<lat>_<lon>.png"): random points in the split's districts, at least
                  2 km from every known kiln, so the whole patch is more than 1 km from a kiln.
- re-cut kilns    ("pc_<lat>_<lon>.png"): existing kiln patches cut again from the same source
                  as the negatives, with the original labels. This way both classes exist in both
                  image sources, and the model cannot score by recognising the source.

Usage:  python src/02_extra_patches.py --split test_kp --neg 321 --recut 321
Reads:  data/processed/patches.csv, data/processed/patches/labels/, data/raw/gadm/gadm41_PAK_3.json
Writes: data/processed/patches/images/{neg_,pc_}*.png, labels for pc_*, data/processed/extra/<split>.csv
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from shapely.geometry import Point, mapping

import s2patch
from config import CLASS_NAMES, IMG_DIR, LBL_DIR, PROC, RAW, S2_END, S2_START, assign_split

MIN_KILN_DIST_M = 2000
UTM = "EPSG:32643"
EXTRA = PROC / "extra"


def known_kilns(patches: pd.DataFrame) -> np.ndarray:
    """UTM xy of every labelled kiln centre in Pakistan."""
    span = s2patch.PATCH * s2patch.DEG_PER_PX
    lons, lats = [], []
    for row in patches[patches["n_kilns"] > 0].itertuples():
        for line in (LBL_DIR / row.image_name.replace(".png", ".txt")).read_text().split("\n"):
            if not line.strip():
                continue
            v = np.array(line.split()[1:], dtype=float)
            x, y = v[0::2].mean(), v[1::2].mean()
            lons.append(row.lon + (x - 0.5) * span)
            lats.append(row.lat - (y - 0.5) * span)
    pts = gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs="EPSG:4326").to_crs(UTM)
    return np.column_stack([pts.x, pts.y])


def sample_negatives(district_geom, n: int, kiln_tree: cKDTree, rng: np.random.Generator) -> list[tuple[float, float]]:
    """Uniform random points inside the (inward-buffered) district, far from known kilns."""
    inner = gpd.GeoSeries([district_geom], crs="EPSG:4326").to_crs(UTM).buffer(-1000).iloc[0]
    if inner.is_empty:
        return []
    minx, miny, maxx, maxy = inner.bounds
    out, tries = [], 0
    while len(out) < n and tries < n * 500:
        tries += 1
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if not inner.contains(Point(x, y)):
            continue
        if kiln_tree.query([x, y])[0] < MIN_KILN_DIST_M:
            continue
        out.append((x, y))
    if not out:
        return []
    ll = gpd.GeoSeries(gpd.points_from_xy(*zip(*out)), crs=UTM).to_crs("EPSG:4326")
    return [(round(p.x, 4), round(p.y, 4)) for p in ll]  # (lon, lat), rounded like SentinelKilnDB names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--neg", type=int, default=0, help="hard negatives to cut")
    ap.add_argument("--recut", type=int, default=0, help="existing kiln patches to re-cut")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    patches = pd.read_csv(PROC / "patches.csv")
    patches["split"] = [assign_split(p, d) for p, d in zip(patches["province"], patches["district"])]
    kiln_tree = cKDTree(known_kilns(patches))

    gadm = gpd.read_file(RAW / "gadm" / "gadm41_PAK_3.json")
    gadm["split"] = [assign_split(p, d) for p, d in zip(gadm["NAME_1"], gadm["NAME_3"])]
    districts = gadm[gadm["split"] == args.split]
    pos = patches[(patches["split"] == args.split) & (patches["n_kilns"] > 0)]
    weight = pos.groupby("district").size()  # negatives follow the kiln distribution across districts

    jobs = []  # (name, lon, lat, province, district, kind, source_image)
    for d in districts.itertuples():
        if d.NAME_3 not in weight.index:
            continue
        n_neg = int(round(args.neg * weight[d.NAME_3] / weight.sum() * 1.3))  # 30% spare for cloudy/edge patches
        for lon, lat in sample_negatives(d.geometry, n_neg, kiln_tree, rng):
            jobs.append((f"neg_{lat:.4f}_{lon:.4f}.png", lon, lat, d.NAME_1, d.NAME_3, "neg", None))
    recut = pos.sample(min(args.recut, len(pos)), random_state=args.seed) if args.recut else pos.iloc[:0]
    for r in recut.itertuples():
        jobs.append((f"pc_{r.image_name}", r.lon, r.lat, r.province, r.district, "recut", r.image_name))

    # One scene search per district (bounding box), reused for every patch in it.
    cat = s2patch.catalog()
    scenes = {}
    for d in districts.itertuples():
        scenes[d.NAME_3] = s2patch.find_scenes(mapping(d.geometry.envelope), S2_START, S2_END, max_cloud=5, cat=cat)
        print(f"{d.NAME_3}: {len(scenes[d.NAME_3])} scenes")

    def run(job):
        name, lon, lat, prov, dist, kind, src = job
        rgb, item = s2patch.cut_best(scenes.get(dist, []), lon, lat)
        if rgb is None:
            return None
        Image.fromarray(rgb).save(IMG_DIR / name)
        if kind == "recut":
            shutil.copy(LBL_DIR / src.replace(".png", ".txt"), LBL_DIR / name.replace(".png", ".txt"))
        return {"image_name": name, "lat": lat, "lon": lon, "province": prov, "district": dist,
                "hf_split": f"pc_{kind}", "scene": item.id}

    rows, failed = [], 0
    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                r = f.result()
            except Exception as e:  # network hiccups: skip the patch, keep going
                r = None
                print("error:", repr(e)[:200])
            if r is None:
                failed += 1
            else:
                rows.append(r)
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} done, {failed} skipped")

    out = pd.DataFrame(rows)
    negs = out[out["hf_split"] == "pc_neg"]
    if len(negs) > args.neg:  # drop the spare negatives
        drop = negs.sample(len(negs) - args.neg, random_state=args.seed)
        for n in drop["image_name"]:
            (IMG_DIR / n).unlink()
        out = out.drop(drop.index)

    counts = recut.set_index("image_name")[["n_kilns"] + [f"n_{c}" for c in CLASS_NAMES.values()]]
    out["src"] = out["image_name"].str.removeprefix("pc_")
    out = out.join(counts, on="src").drop(columns="src").fillna(0)
    for c in ["n_kilns"] + [f"n_{c}" for c in CLASS_NAMES.values()]:
        out[c] = out[c].astype(int)
    EXTRA.mkdir(parents=True, exist_ok=True)
    out.to_csv(EXTRA / f"{args.split}.csv", index=False)
    print(f"\n{args.split}: {len(out)} patches written ({(out.hf_split == 'pc_neg').sum()} negatives, "
          f"{(out.hf_split == 'pc_recut').sum()} re-cut kiln patches), {failed} skipped")


if __name__ == "__main__":
    main()
