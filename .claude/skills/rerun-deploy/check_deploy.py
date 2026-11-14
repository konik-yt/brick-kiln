"""Check that the refreshed KilnWatch deliverables agree with each other and with the model.

Run from the repo root while the app is running:
    .venv/Scripts/python .claude/skills/rerun-deploy/check_deploy.py

Exits 1 and prints FAIL lines if anything disagrees. Checks:
- every row of results/priority_list.csv has confidence >= config.FINAL_CONF
- app/kiln_map.html has one popup per kiln in the priority list
- the running app (http://127.0.0.1:7860) gives the same detections as a file-path prediction
  on every example in app/examples/ (catches RGB/BGR and threshold mismatches)
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from config import FINAL_CONF, FINAL_IMGSZ, FINAL_MODEL, RESULTS, weights_path  # noqa: E402

failures = []
kilns = pd.read_csv(RESULTS / "priority_list.csv")
print(f"model {FINAL_MODEL}, imgsz {FINAL_IMGSZ}, conf {FINAL_CONF}")
print("tiers:", kilns["tier"].value_counts().sort_index().to_dict(), "| kilns:", len(kilns))

if (kilns["confidence"] < FINAL_CONF - 1e-6).any():
    failures.append(f"priority_list.csv has confidences below {FINAL_CONF}: rerun step 7 with the current threshold")

popups = len(re.findall(r"Rank \d+</b>", (ROOT / "app" / "kiln_map.html").read_text(encoding="utf-8")))
if popups != len(kilns):
    failures.append(f"kiln_map.html has {popups} kilns, priority_list.csv has {len(kilns)}: rerun app/build_map.py")

try:
    from gradio_client import Client, handle_file
    from ultralytics import YOLO

    client = Client("http://127.0.0.1:7860/", verbose=False)
    model = YOLO(str(weights_path(FINAL_MODEL)))
    for f in sorted((ROOT / "app" / "examples").glob("*.png")):
        _, table, _ = client.predict(handle_file(str(f)), api_name="/detect")
        app_conf = sorted(round(r[1], 3) for r in table["data"])
        obb = model.predict(str(f), imgsz=FINAL_IMGSZ, conf=FINAL_CONF, verbose=False)[0].obb
        ref_conf = sorted(round(float(c), 3) for c in obb.conf)
        ok = app_conf == ref_conf
        print(f"{'ok  ' if ok else 'FAIL'} {f.name}: app {app_conf} | file path {ref_conf}")
        if not ok:
            failures.append(f"app and model disagree on {f.name}: restart the app, check RGB/BGR and CONF")
except Exception as e:  # app not running or not reachable
    failures.append(f"could not query the app at 127.0.0.1:7860 ({type(e).__name__}): start it and rerun this check")

for f in failures:
    print("FAIL", f)
print("ALL CHECKS PASSED" if not failures else f"{len(failures)} check(s) failed")
sys.exit(1 if failures else 0)
