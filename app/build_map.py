"""Build the map for the demo app: app/kiln_map.html.

Reads:  results/priority_list.csv, results/missing_2026.csv, data/processed/s2_2026/images/
Writes: app/kiln_map.html   (self-contained: each kiln popup embeds its 2026 satellite patch)

Run after steps 7 and 8:  python app/build_map.py
The 2026 tiles live in the git-ignored data/ folder, so the patches are baked into the HTML here.
That way the app runs from a fresh clone without the tiles.
"""

import base64
import io
import sys
from pathlib import Path

import folium
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PROC, RESULTS  # noqa: E402

APP = Path(__file__).resolve().parent
TILES = PROC / "s2_2026" / "images"
TIER_COLOUR = {1: "#d7301f", 2: "#fd8d3c", 3: "#969696"}
TIER_NAME = {1: "Tier 1: confirmed kilns, inspect", 2: "Tier 2: new in 2026, verify first",
             3: "Tier 3: likely city false alarms"}


def patch_png(row, size: int = 192) -> str:
    im = Image.open(TILES / row.tile).convert("RGB").resize((size, size), Image.NEAREST)
    d = ImageDraw.Draw(im)
    pts = [tuple(float(v) * size for v in xy.split(",")) for xy in row.box_px.split(";")]
    d.polygon(pts, outline=(255, 0, 0), width=2)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def popup_html(row) -> str:
    return (
        f"<div style='font-family:sans-serif;font-size:12px;width:200px'>"
        f"<b>Rank {row.rank}</b> &middot; tier {row.tier}<br>"
        f"<img src='data:image/jpeg;base64,{patch_png(row)}' width='192' height='192'><br>"
        f"Type: {row.type} (confidence {row.confidence:.2f})<br>"
        f"Status: {row.status}<br>"
        f"OSM-mapped schools within 1 km: {row.schools_1km}<br>"
        f"People within 1 km: {int(row.population_1km):,}<br>"
        f"Priority score: {row.priority:.2f}<br>"
        f"Action: {row.action}<br>"
        f"<a href='https://www.google.com/maps/@{row.lat},{row.lon},500m/data=!3m1!1e3' target='_blank'>"
        f"Open high-res imagery</a><br>"
        f"{row.lat:.5f}, {row.lon:.5f}</div>"
    )


def main() -> None:
    kilns = pd.read_csv(RESULTS / "priority_list.csv")
    missing = pd.read_csv(RESULTS / "missing_2026.csv")

    m = folium.Map(location=[kilns["lat"].mean(), kilns["lon"].mean()], zoom_start=11, tiles=None, control_scale=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets").add_to(m)

    max_p = kilns["priority"].max()
    for tier in [3, 2, 1]:  # tier 1 drawn last, on top
        layer = folium.FeatureGroup(name=TIER_NAME[tier], show=tier < 3)
        for row in kilns[kilns["tier"] == tier].itertuples():
            folium.CircleMarker(
                [row.lat, row.lon],
                radius=4 + 8 * row.priority / max_p,
                color="white", weight=1,
                fill=True, fill_color=TIER_COLOUR[tier], fill_opacity=0.9,
                tooltip=f"#{row.rank} {row.type}, tier {row.tier}",
                popup=folium.Popup(popup_html(row), max_width=230),
            ).add_to(layer)
        layer.add_to(m)

    gone = folium.FeatureGroup(name="In 2024 list, not detected in 2026", show=False)
    for row in missing.itertuples():
        folium.CircleMarker([row.lat, row.lon], radius=3, color="#3182bd", fill=True, fill_opacity=0.7,
                            tooltip=f"2024 kiln {int(row.id)} not detected in 2026").add_to(gone)
    gone.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    out = APP / "kiln_map.html"
    m.save(str(out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(kilns)} kilns, {len(missing)} missing)")


if __name__ == "__main__":
    main()
