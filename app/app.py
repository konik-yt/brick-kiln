"""KilnWatch demo app.

Run:  python app/app.py     then open http://127.0.0.1:7860

Tabs
- Map      Peshawar kilns found in Oct to Nov 2026 Sentinel-2 imagery, coloured by tier and sized by
           priority. Click a kiln for its satellite patch, type, confidence, schools and people nearby.
- Try it   Upload a 128 x 128 Sentinel-2 RGB patch (10 m per pixel) or pick an example; the detector
           draws the kilns it finds.
Sidebar    Top 10 of each tier and the full priority list as CSV.

Needs only files that are in the repo: models/, results/, app/kiln_map.html, app/examples/.
"""

import html
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import CLASS_NAMES, FINAL_CONF, FINAL_IMGSZ, FINAL_MODEL, RESULTS, weights_path  # noqa: E402

MODEL_NAME, IMGSZ, CONF = FINAL_MODEL, FINAL_IMGSZ, FINAL_CONF
APP = Path(__file__).resolve().parent
COLOURS = {"CFCBK": (255, 60, 60), "FCBK": (255, 60, 60), "Zigzag": (60, 220, 60)}

model = YOLO(str(weights_path(MODEL_NAME)))
kilns = pd.read_csv(RESULTS / "priority_list.csv")
TABLE_COLS = ["rank", "type", "confidence", "schools_1km", "population_1km", "priority", "status", "lat", "lon"]


def detect(image):
    if image is None:
        return None, pd.DataFrame(), "Upload a patch or pick an example."
    im = Image.fromarray(image).convert("RGB")
    note = ""
    if im.size != (128, 128):
        note = f"The image was {im.size[0]} x {im.size[1]} pixels; it was resized to 128 x 128. "
        im = im.resize((128, 128), Image.BILINEAR)
    # Ultralytics reads numpy arrays as BGR (OpenCV order); Gradio gives RGB.
    bgr = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
    r = model.predict(bgr, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    scale = 4
    big = im.resize((128 * scale, 128 * scale), Image.NEAREST)
    d = ImageDraw.Draw(big)
    rows = []
    if r.obb is not None:
        for cls, poly, conf in zip(r.obb.cls, r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf):
            name = CLASS_NAMES[int(cls)]
            pts = [tuple(p * scale) for p in poly]
            d.polygon(pts, outline=COLOURS[name], width=3)
            d.text((pts[0][0] + 3, pts[0][1] - 12), f"{name} {float(conf):.2f}", fill=COLOURS[name])
            rows.append({"type": name, "confidence": round(float(conf), 3)})
    table = pd.DataFrame(rows)
    msg = note + (f"Found {len(rows)} kiln(s)." if rows else f"No kilns found at confidence {CONF:.2f} or higher.")
    return np.asarray(big), table, msg


def tier_table(tier: int) -> pd.DataFrame:
    return kilns[kilns["tier"] == tier].head(10)[TABLE_COLS]


def map_iframe() -> str:
    page = (APP / "kiln_map.html").read_text(encoding="utf-8")
    return (f'<iframe srcdoc="{html.escape(page, quote=True)}" '
            f'style="width:100%;height:640px;border:0" loading="lazy"></iframe>')


counts = kilns["tier"].value_counts().to_dict()
with gr.Blocks(title="KilnWatch") as demo:
    with gr.Sidebar(open=True, width=440):
        gr.Markdown("### Inspection list, Peshawar district\nEach tier is ranked by priority = "
                    "type weight x (1 + schools within 1 km) + people within 1 km / 10,000.")
        tier = gr.Radio(choices=[("Tier 1: inspect", 1), ("Tier 2: verify first", 2), ("Tier 3: likely city", 3)],
                        value=1, label="Tier")
        top = gr.Dataframe(value=tier_table(1), label="Top 10", interactive=False, wrap=True)
        tier.change(tier_table, tier, top)
        gr.DownloadButton("Download full list (CSV)", value=str(RESULTS / "priority_list.csv"))
        gr.Markdown("Schools come from OpenStreetMap, which maps only part of KP's schools, mostly in cities.")

    gr.Markdown(
        "# KilnWatch\n"
        "Brick kilns detected in Sentinel-2 imagery of Peshawar (October to November 2026) and ranked so an "
        f"inspector knows where to go first. Tier 1: {counts.get(1, 0)} kilns also in the 2024 NUST/Oxford list. "
        f"Tier 2: {counts.get(2, 0)} new detections to verify. Tier 3: {counts.get(3, 0)} likely city false alarms."
    )
    with gr.Tab("Map"):
        gr.HTML(map_iframe())
        gr.Markdown("Red: tier 1, orange: tier 2, grey: tier 3 (hidden by default). Bigger circles have a higher "
                    "priority. Blue (hidden by default): kilns in the 2024 list that were not detected in 2026.")
    with gr.Tab("Try it"):
        with gr.Row():
            inp = gr.Image(label="Sentinel-2 RGB patch, 128 x 128 pixels at 10 m", type="numpy", height=384)
            out = gr.Image(label="Detections (red: fixed chimney, green: Zigzag)", height=384)
        msg = gr.Markdown()
        dets = gr.Dataframe(label="Kilns found", interactive=False)
        gr.Examples(examples=sorted(str(p) for p in (APP / "examples").glob("*.png")), inputs=inp)
        gr.Button("Detect kilns", variant="primary").click(detect, inp, [out, dets, msg])
        inp.upload(detect, inp, [out, dets, msg])

if __name__ == "__main__":
    demo.launch()
