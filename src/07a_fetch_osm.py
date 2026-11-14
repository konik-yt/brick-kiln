"""Step 7a: schools and hospitals from OpenStreetMap for the Peshawar valley.

Reads:  data/raw/gadm/gadm41_PAK_3.json
Writes: data/raw/osm/schools_peshawar_valley.geojson, data/raw/osm/hospitals_peshawar_valley.geojson

Uses the Overpass API directly (bounding box of the test districts plus ~5 km). The public server
rejects requests without a User-Agent, so we send a generic one.
"""

import time

import geopandas as gpd
import requests
from shapely.geometry import Point

from config import RAW, TEST_KP_DISTRICTS

URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "KilnWatch/0.1 (student hackathon project)"}
LAYERS = {
    "schools": ["school", "college", "kindergarten", "university"],
    "hospitals": ["hospital", "clinic"],
}


def query(bbox: tuple, amenities: list[str]) -> list[dict]:
    s, w, n, e = bbox
    body = "".join(f'nwr["amenity"="{a}"]({s},{w},{n},{e});' for a in amenities)
    q = f"[out:json][timeout:180];({body});out center;"
    for attempt in range(5):
        r = requests.post(URL, data={"data": q}, headers=HEADERS, timeout=240)
        if r.ok:
            return r.json()["elements"]
        print(f"overpass {r.status_code}, retrying in {10 * 2 ** attempt}s")
        time.sleep(10 * 2 ** attempt)
    r.raise_for_status()


def main() -> None:
    gadm = gpd.read_file(RAW / "gadm" / "gadm41_PAK_3.json")
    minx, miny, maxx, maxy = gadm[gadm["NAME_3"].isin(TEST_KP_DISTRICTS)].total_bounds
    bbox = (miny - 0.05, minx - 0.05, maxy + 0.05, maxx + 0.05)
    (RAW / "osm").mkdir(parents=True, exist_ok=True)
    for layer, amenities in LAYERS.items():
        rows = []
        for el in query(bbox, amenities):
            lat = el.get("lat", el.get("center", {}).get("lat"))
            lon = el.get("lon", el.get("center", {}).get("lon"))
            if lat is None:
                continue
            tags = el.get("tags", {})
            rows.append({"osm_id": f"{el['type']}/{el['id']}", "amenity": tags.get("amenity"),
                         "name": tags.get("name", ""), "geometry": Point(lon, lat)})
        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        gdf.to_file(RAW / "osm" / f"{layer}_peshawar_valley.geojson", driver="GeoJSON")
        print(layer, len(gdf), gdf["amenity"].value_counts().to_dict())


if __name__ == "__main__":
    main()
