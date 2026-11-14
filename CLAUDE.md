# KilnWatch

Brick kiln detector and inspection list for Peshawar, KP (Veritas AI hackathon). The plan is `docs/KilnWatch-project-plan.pdf`, the write-up is `README.md`.

## Environment

- Windows, Python 3.11 venv in `.venv`. Run scripts as `.venv/Scripts/python src/<script>.py` from the repo root.
- PyTorch is the CUDA 12.8 build (`cu128`), needed for the RTX 5080. Do not reinstall torch from plain PyPI.
- Training uses `workers=2`; more dataloader workers can hang on Windows.
- Background shell jobs stop after 2 hours. Run cutting and training as separate jobs, never chained in one command.

## Pipeline

Scripts in `src/` run in order. Each docstring lists what it reads and writes. Shared paths, district lists and `assign_split()` live in `src/config.py`.

1. `01_load_sentinelkilndb.py` Pakistan patches from SentinelKilnDB, tagged with GADM level 3 district (level 2 is divisions in Pakistan).
2. `02_extra_patches.py` + `s2patch.py` hard negatives (`neg_`), urban negatives (`negu_`, `--urban 40`) and re-cut kiln patches (`pc_`) from Planetary Computer.
3. `03_split.py` district split and overlap drop.
4. `04_baselines.py` majority and Random Forest.
5. `05_train.py` YOLOv8-OBB training.
6. `06_evaluate.py` patch-level and detection-level scores.
7. `07a_fetch_osm.py`, `07_detect_2026.py` 2026 detection on Peshawar.
8. `08_priority.py` three-tier inspection list.
9. `09_failure_gallery.py` failure examples.
App: `app/build_map.py` then `app/app.py` (http://127.0.0.1:7860).

To refresh the 2026 list, map and app after a model or threshold change, use the `/rerun-deploy` skill.

## Final model

- `yolov8s_384_urban`, `--imgsz 384`, box threshold `0.20`. The threshold was chosen on validation.
- These three values live only in `FINAL_MODEL`, `FINAL_IMGSZ`, `FINAL_CONF` in `src/config.py`; steps 7 and 9 and the app read them. Change them there, nowhere else.
- Weights: `models/yolov8s_384_urban.pt` (tracked). `config.weights_path(name)` prefers `runs/<name>/weights/best.pt` when it exists.

## Evaluation rules

- Split by district, never at random. Patches overlap by 30 pixels, so a random split leaks.
- Model choice and every threshold come from the `val` split only. Pass `--conf` explicitly when scoring test splits.
- Test splits (`test_kp`, `test_punjab`) are scored once per final model. Never tune anything on them.
- Read results by source: `pc` (same-source patches, the honest kiln vs no kiln test), `skdb` (original patches, easier because their empty patches are desert), `urban_negatives` (city false alarms).
- After any change to `02_extra_patches.py`, `03_split.py`, `04_baselines.py`, `06_evaluate.py`, `src/config.py` district lists or thresholds, run the `leakage-auditor` subagent before committing, and fix what it reports.

## Image recipe

Every patch must be cut like SentinelKilnDB, or the model sees a colour shift. Use `s2patch.py`, do not write a new cutter:

- Sentinel-2 L2A bands B4, B3, B2.
- EPSG:4326 grid at 8.983e-5 degrees per pixel (Earth Engine `scale=10`), 128 x 128 pixels, named `<lat>_<lon>` rounded to 4 decimals at the patch centre.
- Nearest-neighbour resampling.
- Each band of each patch stretched by its own min and max to 0 to 255.

## Gotchas

- Ultralytics reads numpy arrays as BGR. Flip RGB arrays (`arr[:, :, ::-1]`) before `model.predict`. File paths are fine.
- Label class ids: 0 = CFCBK, 1 = FCBK, 2 = Zigzag.
- The Overpass API rejects requests without a User-Agent header.
- `data/` is git-ignored except `data/raw/osm/schools_kp_osm.geojson`. `runs/` and `*.pt` are ignored except `models/*.pt`.
- OSM undercounts KP schools, so most rural kilns show 0 schools; this is a known limitation, not a bug.

## Commits

Commit to `main` and push. End commit messages with the Co-Authored-By line from the session instructions.
