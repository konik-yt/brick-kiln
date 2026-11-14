"""Step 8: turn 2026 detections into an inspection priority list.

Reads:  results/kilns_2026.geojson                         (step 7)
        data/raw/nust/Brick_Kilns_IGP_PK.csv               (NUST/Oxford 2024 kiln points)
        data/raw/osm/schools_kp_osm.geojson                (OSM schools, colleges, universities for all of KP)
        data/raw/osm/hospitals_peshawar_valley.geojson     (OSM hospitals and clinics, step 7a)
        data/raw/worldpop/pak_ppp_2020_constrained.tif     (people per ~100 m pixel)
        data/raw/gadm/gadm41_PAK_3.json
Writes: results/priority_list.csv    detected kilns, ranked
        results/missing_2026.csv     2024 NUST kilns with no 2026 detection nearby

priority = w_type * (1 + schools_1km) + population_1km / 10,000
w_type = 2 for fixed-chimney kilns (FCBK, CFCBK), 1 for Zigzag.

Status from matching 2026 detections to 2024 NUST points within MATCH_M:
- confirmed     found in both
- new_2026      detected now, not in the 2024 list: possibly new or unregistered
- missing_2026  in the 2024 list, not detected now: closed, or a model miss (see missing_2026.csv)

Limitation: OpenStreetMap maps only a fraction of schools in KP, mostly in cities (for example 9 in
Swabi and 17 in Mardan), so schools_1km counts OSM-mapped schools and undercounts near rural kilns.
No official KP school list with coordinates is openly available. population_1km (WorldPop) does
cover rural areas, so it still carries the exposure signal where schools are missing.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

from config import RAW, RESULTS

MATCH_M = 150
RADIUS_M = 1000
UTM = "EPSG:32642"  # Peshawar valley lies in UTM zone 42N
W_TYPE = {"FCBK": 2, "CFCBK": 2, "Zigzag": 1}


def utm_xy(gdf: gpd.GeoDataFrame) -> np.ndarray:
    p = gdf.to_crs(UTM).geometry
    return np.column_stack([p.x, p.y])


def count_within(points: np.ndarray, targets: np.ndarray, r: float) -> np.ndarray:
    if len(targets) == 0:
        return np.zeros(len(points), int)
    return np.array([len(x) for x in cKDTree(targets).query_ball_point(points, r)])


def population_within(lons, lats, r: float = RADIUS_M) -> np.ndarray:
    """Sum WorldPop pixels whose centres are within r metres."""
    out = np.zeros(len(lons))
    with rasterio.open(RAW / "worldpop" / "pak_ppp_2020_constrained.tif") as src:
        for i, (lon, lat) in enumerate(zip(lons, lats)):
            dlat = r / 111_320
            dlon = r / (111_320 * np.cos(np.radians(lat)))
            win = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform).round_offsets().round_lengths()
            a = src.read(1, window=win, boundless=True, fill_value=0).astype(float)
            a[a < 0] = 0  # nodata
            t = src.window_transform(win)
            rows, cols = np.indices(a.shape)
            xs, ys = rasterio.transform.xy(t, rows.ravel(), cols.ravel())
            dx = (np.array(xs) - lon) * 111_320 * np.cos(np.radians(lat))
            dy = (np.array(ys) - lat) * 111_320
            out[i] = a.ravel()[np.hypot(dx, dy) <= r].sum()
    return out


def main() -> None:
    kilns = gpd.read_file(RESULTS / "kilns_2026.geojson")
    districts = sorted(kilns["district"].unique())
    gadm = gpd.read_file(RAW / "gadm" / "gadm41_PAK_3.json")
    area = gadm[gadm["NAME_3"].isin(districts)][["NAME_3", "geometry"]]

    nust = pd.read_csv(RAW / "nust" / "Brick_Kilns_IGP_PK.csv")
    nust = gpd.GeoDataFrame(nust, geometry=gpd.points_from_xy(nust["lon"], nust["lat"]), crs="EPSG:4326")
    nust = gpd.sjoin(nust, area, predicate="within").drop(columns="index_right")
    nust["type"] = nust["type"].replace({"ZZ": "Zigzag"})

    k_xy, n_xy = utm_xy(kilns), utm_xy(nust)
    dist, idx = cKDTree(n_xy).query(k_xy, distance_upper_bound=MATCH_M)
    matched = np.isfinite(dist)
    kilns["status"] = np.where(matched, "confirmed", "new_2026")
    kilns["nust_id"] = np.where(matched, nust["id"].to_numpy()[np.minimum(idx, len(nust) - 1)], np.nan)
    kilns["nust_type"] = np.where(matched, nust["type"].to_numpy()[np.minimum(idx, len(nust) - 1)], "")

    schools = gpd.read_file(RAW / "osm" / "schools_kp_osm.geojson")
    hospitals = gpd.read_file(RAW / "osm" / "hospitals_peshawar_valley.geojson")
    kilns["schools_1km"] = count_within(k_xy, utm_xy(schools), RADIUS_M)
    kilns["hospitals_1km"] = count_within(k_xy, utm_xy(hospitals), RADIUS_M)
    kilns["population_1km"] = population_within(kilns["lon"], kilns["lat"]).round()

    kilns["priority"] = kilns["type"].map(W_TYPE) * (1 + kilns["schools_1km"]) + kilns["population_1km"] / 10_000
    kilns = kilns.sort_values(["priority", "confidence"], ascending=False).reset_index(drop=True)
    kilns.insert(0, "rank", np.arange(1, len(kilns) + 1))
    cols = ["rank", "lat", "lon", "district", "type", "confidence", "schools_1km", "hospitals_1km",
            "population_1km", "priority", "status", "nust_id", "nust_type", "tile", "box_px", "scene"]
    kilns["lat"], kilns["lon"] = kilns["lat"].round(5), kilns["lon"].round(5)
    kilns["confidence"], kilns["priority"] = kilns["confidence"].round(3), kilns["priority"].round(3)
    kilns[cols].to_csv(RESULTS / "priority_list.csv", index=False)

    # 2024 kilns with no 2026 detection nearby
    back = cKDTree(k_xy).query(n_xy, distance_upper_bound=MATCH_M)[0]
    missing = nust[~np.isfinite(back)][["id", "lat", "lon", "type", "NAME_3"]].rename(columns={"NAME_3": "district"})
    missing.to_csv(RESULTS / "missing_2026.csv", index=False)

    print(f"districts: {districts}")
    print(f"2026 detections: {len(kilns)}   2024 NUST kilns in area: {len(nust)}")
    print(kilns["status"].value_counts().to_string())
    print(f"missing_2026 (in 2024 list, not detected): {len(missing)}")
    print("\nTop 10:\n", kilns[cols[:11]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
