"""Step 7: run the detector on fresh 2026 Sentinel-2 imagery of the Peshawar valley.

Reads:  runs/<name>/weights/best.pt, data/raw/gadm/gadm41_PAK_3.json
Writes: data/processed/s2_2026/images/<lat>_<lon>.png      (tiles, kept for the demo app)
        data/processed/s2_2026/tiles.csv                   (tile centre, district, scene used)
        results/kilns_2026.geojson                          (one point per kiln after merging)

Usage:  python src/07_detect_2026.py --name yolov8n_256_hardneg --conf 0.25 --districts Peshawar

Tiles are 128 x 128 pixels on the SentinelKilnDB grid (EPSG:4326, Earth Engine scale=10) with a
30-pixel overlap, cut with the same per-patch stretch as the training data (see s2patch.py).
Detections centred within EDGE_PX of a tile edge are dropped: tiles overlap by 30 pixels, so every
spot is at least 15 pixels from the edge in some tile, where the kiln is seen whole.
A kiln near a tile edge can be found in two tiles, so detections closer than MERGE_M are merged,
keeping the most confident one.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from shapely.geometry import box, mapping
from ultralytics import YOLO

import s2patch
from config import CLASS_NAMES, PROC, RAW, RESULTS, ROOT, weights_path

OUT = PROC / "s2_2026"
STRIDE = s2patch.PATCH - 30
MERGE_M = 150
EDGE_PX = 12


def tile_centres(geom) -> list[tuple[float, float]]:
    """Tile centres on the patch grid covering a district polygon."""
    step = STRIDE * s2patch.DEG_PER_PX
    half = s2patch.PATCH / 2 * s2patch.DEG_PER_PX
    minx, miny, maxx, maxy = geom.bounds
    out = []
    for lat in np.arange(maxy - half, miny - half, -step):
        for lon in np.arange(minx + half, maxx + half, step):
            if geom.intersects(box(lon - half, lat - half, lon + half, lat + half)):
                out.append((round(float(lon), 4), round(float(lat), 4)))
    return out


def cut_tiles(districts: gpd.GeoDataFrame, start: str, end: str, workers: int) -> pd.DataFrame:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    cat = s2patch.catalog()
    # Tiles cut in an earlier run are reused; their scene ids come from the earlier tiles.csv.
    old = OUT / "tiles.csv"
    known = dict(pd.read_csv(old)[["image_name", "scene"]].dropna().values) if old.exists() else {}
    jobs = []
    for d in districts.itertuples():
        scenes = s2patch.find_scenes(mapping(d.geometry.envelope), start, end, max_cloud=10, cat=cat)
        print(f"{d.NAME_3}: {len(scenes)} scenes with <10% cloud between {start} and {end}")
        jobs += [(lon, lat, d.NAME_3, scenes) for lon, lat in tile_centres(d.geometry)]

    def run(job):
        lon, lat, dist, scenes = job
        name = f"{lat:.4f}_{lon:.4f}.png"
        if (OUT / "images" / name).exists():
            return {"image_name": name, "lat": lat, "lon": lon, "district": dist, "scene": known.get(name)}
        rgb, item = s2patch.cut_best(scenes, lon, lat)
        if rgb is None:
            return {"image_name": None, "lat": lat, "lon": lon, "district": dist, "scene": None}
        Image.fromarray(rgb).save(OUT / "images" / name)
        return {"image_name": name, "lat": lat, "lon": lon, "district": dist, "scene": item.id}

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(run, j) for j in jobs]), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                print("error:", repr(e)[:200])
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} tiles")
    tiles = pd.DataFrame(rows)
    missing = tiles["image_name"].isna().sum()
    print(f"{len(tiles) - missing} tiles cut, {missing} without a clear scene")
    return tiles


def detect(model, tiles: pd.DataFrame, conf: float, imgsz: int) -> pd.DataFrame:
    span = s2patch.PATCH * s2patch.DEG_PER_PX
    ok = tiles.dropna(subset=["image_name"]).reset_index(drop=True)
    paths = [str(OUT / "images" / n) for n in ok["image_name"]]
    rows = []
    for i in range(0, len(paths), 256):
        for j, r in enumerate(model.predict(paths[i:i + 256], imgsz=imgsz, conf=conf, verbose=False)):
            t = ok.iloc[i + j]
            if r.obb is None or len(r.obb) == 0:
                continue
            corners = r.obb.xyxyxyxyn.cpu().numpy()  # (n, 4, 2), normalised to the tile
            for k in range(len(corners)):
                cx, cy = corners[k, :, 0].mean(), corners[k, :, 1].mean()
                margin = EDGE_PX / s2patch.PATCH
                if min(cx, cy, 1 - cx, 1 - cy) < margin:  # cut off by the tile edge; seen whole in a neighbour
                    continue
                rows.append({
                    "lon": t.lon + (cx - 0.5) * span,
                    "lat": t.lat - (cy - 0.5) * span,
                    "type": CLASS_NAMES[int(r.obb.cls[k])],
                    "confidence": float(r.obb.conf[k]),
                    "tile": t.image_name,
                    "district": t.district,
                    "box_px": ";".join(f"{x:.4f},{y:.4f}" for x, y in corners[k]),
                    "scene": t.scene,
                })
    return pd.DataFrame(rows)


def merge_duplicates(det: pd.DataFrame) -> pd.DataFrame:
    """Greedy merge: walk detections by confidence, drop any within MERGE_M of one already kept."""
    if det.empty:
        return det
    det = det.sort_values("confidence", ascending=False).reset_index(drop=True)
    pts = gpd.GeoSeries(gpd.points_from_xy(det["lon"], det["lat"]), crs="EPSG:4326").to_crs("EPSG:32642")
    tree = cKDTree(np.column_stack([pts.x, pts.y]))
    keep, removed = [], np.zeros(len(det), bool)
    for i in range(len(det)):
        if removed[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point([pts.x[i], pts.y[i]], MERGE_M):
            if j != i:
                removed[j] = True
    return det.iloc[keep].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--conf", type=float, required=True, help="box threshold, chosen on val in step 6")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--districts", nargs="+", default=["Peshawar"])
    ap.add_argument("--start", default="2026-10-01")
    ap.add_argument("--end", default="2026-11-14")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    gadm = gpd.read_file(RAW / "gadm" / "gadm41_PAK_3.json")
    districts = gadm[gadm["NAME_3"].isin(args.districts)]
    tiles = cut_tiles(districts, args.start, args.end, args.workers)
    tiles.to_csv(OUT / "tiles.csv", index=False)

    model = YOLO(str(weights_path(args.name)))
    det = detect(model, tiles, args.conf, args.imgsz)
    kilns = merge_duplicates(det)
    print(f"{len(det)} raw detections, {len(kilns)} kilns after merging duplicates within {MERGE_M} m")
    if not kilns.empty:
        print(kilns.groupby(["district", "type"]).size().to_string())

    RESULTS.mkdir(exist_ok=True)
    gdf = gpd.GeoDataFrame(kilns, geometry=gpd.points_from_xy(kilns["lon"], kilns["lat"]), crs="EPSG:4326")
    gdf.to_file(RESULTS / "kilns_2026.geojson", driver="GeoJSON")


if __name__ == "__main__":
    main()
