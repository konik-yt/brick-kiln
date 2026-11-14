"""Cut Sentinel-2 patches the same way SentinelKilnDB made theirs.

SentinelKilnDB recipe (rishabh-mondal/SENTINELKILNDB_NeurIPS_2025, data_scripts/):
- Sentinel-2 L2A surface reflectance, bands B4, B3, B2 (red, green, blue)
- scenes with under 1% cloud, least cloudy first
- exported from Earth Engine at scale=10 in EPSG:4326, so pixels are 8.983e-5 degrees on both axes
- 128 x 128 pixel patches, named by centre "<lat>_<lon>" rounded to 4 decimals
- each band of each patch stretched on its own: (band - min) / (max - min) * 255

We read the same bands from Microsoft Planetary Computer (no account needed) and warp them
onto the same lat/lon pixel grid, then apply the same per-patch stretch.
"""

from __future__ import annotations

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEG_PER_PX = 10 / 111319.49079327357  # Earth Engine "scale=10" in EPSG:4326
PATCH = 128
BANDS = ["B04", "B03", "B02"]  # red, green, blue
CLOUD_CLASSES = [3, 8, 9, 10]  # SCL: cloud shadow, cloud medium, cloud high, cirrus


def catalog():
    # Unsigned search; asset URLs are signed at read time so long runs survive token expiry.
    return pystac_client.Client.open(STAC_URL)


def find_scenes(geometry: dict, start: str, end: str, max_cloud: float = 5.0, cat=None) -> list:
    """Scenes intersecting a GeoJSON geometry, least cloudy first."""
    cat = cat or catalog()
    items = cat.search(
        collections=["sentinel-2-l2a"],
        intersects=geometry,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()
    return sorted(items, key=lambda it: it.properties["eo:cloud_cover"])


def patch_transform(lon: float, lat: float, size: int = PATCH):
    half = size / 2 * DEG_PER_PX
    return from_origin(lon - half, lat + half, DEG_PER_PX, DEG_PER_PX)


def read_window(href: str, lon: float, lat: float, size: int = PATCH, resampling=Resampling.nearest) -> np.ndarray:
    """Read one band as a size x size array on the EPSG:4326 grid centred on (lon, lat).

    Nearest neighbour matches Earth Engine's default resampling, so our patches are not smoother
    than SentinelKilnDB's (a smoothness difference would be a shortcut the model could learn).
    """
    with rasterio.open(planetary_computer.sign(href)) as src:
        with WarpedVRT(src, crs="EPSG:4326", transform=patch_transform(lon, lat, size), width=size, height=size,
                       resampling=resampling) as vrt:
            return vrt.read(1)


def stretch(band: np.ndarray) -> np.ndarray:
    """SentinelKilnDB per-patch, per-band min-max stretch to uint8."""
    band = band.astype(np.float64)
    return ((band - band.min()) / (band.max() - band.min() + 1e-5) * 255).astype(np.uint8)


def to_rgb(raw: np.ndarray) -> np.ndarray:
    """raw: (3, H, W) reflectance in B4, B3, B2 order -> (H, W, 3) uint8."""
    return np.dstack([stretch(b) for b in raw])


def cut_patch(item, lon: float, lat: float, size: int = PATCH):
    """Return (rgb uint8 HxWx3, raw reflectance 3xHxW, cloud_fraction, nodata_fraction)."""
    scl = read_window(item.assets["SCL"].href, lon, lat, size)
    nodata = float((scl == 0).mean())
    cloud = float(np.isin(scl, CLOUD_CLASSES).mean())
    if nodata > 0 or cloud > 0.01:  # skip the expensive band reads
        return None, None, cloud, nodata
    raw = np.stack([read_window(item.assets[b].href, lon, lat, size) for b in BANDS])
    return to_rgb(raw), raw, cloud, nodata


def cut_best(items: list, lon: float, lat: float, size: int = PATCH):
    """Try scenes (least cloudy first) whose footprint covers the patch; return (rgb, item) or (None, None)."""
    from shapely.geometry import box, shape

    half = size / 2 * DEG_PER_PX
    fp = box(lon - half, lat - half, lon + half, lat + half)
    for item in items:
        if not shape(item.geometry).contains(fp):
            continue
        rgb, _, _, _ = cut_patch(item, lon, lat, size)
        if rgb is not None:
            return rgb, item
    return None, None
