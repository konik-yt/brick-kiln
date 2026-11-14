# KilnWatch

KilnWatch finds brick kilns in free Sentinel-2 satellite images of Peshawar district and ranks them, so a district Environmental Protection Agency inspector who can visit only a few kilns a week knows where to go first.

Veritas AI International High School Hackathon 2026, track AI + Climate. Builds on suggested project 4.3.3 (land-use mapping).

## The problem

Brick kilns are a major source of winter smog in Pakistan, and many are on no official list. An EPA inspector in Peshawar can visit only a handful a week and has no up-to-date map of where the kilns are, which are the older, dirtier fixed-chimney type, and which sit next to schools and homes.

Decision the tool supports: which kilns to inspect this week.

## What we built

1. A kiln detector (YOLOv8s with oriented boxes) trained on Punjab and tested on districts it never saw, including the Peshawar valley in Khyber Pakhtunkhwa (KP).
2. A run on fresh October to November 2026 Sentinel-2 images of Peshawar district: 1,729 image tiles, 217 kilns found.
3. An inspection list in three tiers, each ranked by a simple priority score:
   - Tier 1 (132 kilns): found by our model and also in the 2024 NUST/Oxford kiln list. Inspect.
   - Tier 2 (76 kilns): found in 2026 but not in the 2024 list. Possibly new or unregistered, so verify on high-resolution imagery first.
   - Tier 3 (9 kilns): new detections in dense city, where kilns cannot fit. Most likely false alarms.
4. A demo app with a map of the ranked kilns and a "try it" tab that runs the detector on any patch.

Priority score: `type weight x (1 + schools within 1 km) + people within 1 km / 10,000`, with type weight 2 for fixed-chimney kilns and 1 for Zigzag kilns.

## What we added beyond the datasets

- **Our own question 1: does a model trained on Punjab work in KP?** We split by district, never at random, and kept the Peshawar valley (Peshawar, Charsadda, Mardan, Nowshera, Swabi) out of training. Answer: mostly yes, but it misses more kilns in KP (see Results).
- **Our own question 2: what is new since 2024?** We matched 2026 detections to the 2024 kiln list.
- **Hard negatives.** In SentinelKilnDB, almost all empty patches are south Punjab desert and KP has none, so a colour-only Random Forest scored F1 0.97 by telling desert from farmland. We cut about 6,300 new empty patches from farmland, towns and cities at least 2 km from any known kiln, which makes "kiln or no kiln" a real test.
- **Same-source test images.** We also re-cut known kiln patches from the same image source and dates as our negatives. Both classes then come from the same source, so a model cannot score by recognising where an image came from.
- **A fix found by using the tool.** The first 2026 list ranked city blocks at the top. We added urban negatives to training, which cut false alarms on city patches from 18% to 7% on validation, and built the three tiers.

## How it works

| Step | Script | What it does |
|---|---|---|
| 1 | `src/01_load_sentinelkilndb.py` | Takes the 18,209 Pakistan patches from SentinelKilnDB and tags each with its district (GADM level 3). |
| 2 | `src/02_extra_patches.py`, `src/s2patch.py` | Cuts hard negatives, urban negatives and re-cut kiln patches from Sentinel-2 (Microsoft Planetary Computer) with the dataset's exact recipe: bands B4, B3, B2, the same lat/lon pixel grid, nearest-neighbour resampling, and each band of each patch stretched by its own min and max. |
| 3 | `src/03_split.py` | Splits by district. Train: Punjab. Validation: 4 Punjab districts. Tests: 4 other Punjab districts and the Peshawar valley. Patches that overlap a patch in a stricter split are dropped, since patches overlap by 30 pixels. |
| 4 | `src/04_baselines.py` | Majority-label and Random Forest (colour statistics) baselines. |
| 5 | `src/05_train.py` | Trains YOLOv8-OBB. Final model: `yolov8s-obb`, images upscaled from 128 to 384 pixels, 50 epochs, on an RTX 5080 laptop GPU. |
| 6 | `src/06_evaluate.py` | Scores patch-level kiln vs no kiln (same task as the baselines) and box-level detection. The threshold (0.20) is chosen on validation, never on a test set. |
| 7 | `src/07a_fetch_osm.py`, `src/07_detect_2026.py` | Fetches schools and hospitals from OpenStreetMap; tiles Peshawar district from clear Oct to Nov 2026 scenes, runs the detector, drops boxes cut off at tile edges and merges duplicates within 150 m. |
| 8 | `src/08_priority.py` | Matches detections to the 2024 list within 150 m, counts schools and people (WorldPop) within 1 km, and ranks the three tiers. |
| 9 | `src/09_failure_gallery.py` | Collects wrong predictions for the failure gallery. |
| App | `app/build_map.py`, `app/app.py` | Builds the map and runs the Gradio demo. |

We chose the model on validation only. Four models were compared there (YOLOv8n without and with hard negatives, YOLOv8s at 384 px with hard negatives, and the same with urban negatives).

## Results against the baselines

All numbers are on districts the model never saw in training. "Same-source patches" are kiln patches and empty patches cut from the same Sentinel-2 source and dates, including 200 town and city patches per test set. These are the fair test of "kiln or no kiln".

Patch level, kiln vs no kiln, same-source patches:

| Test set | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Peshawar valley (842 patches, 321 with kilns) | Majority label ("kiln") | 0.381 | 0.381 | 1.000 | 0.552 |
| | Random Forest on colour | 0.551 | 0.439 | 0.636 | 0.519 |
| | **KilnWatch (YOLOv8s-OBB)** | **0.899** | **0.918** | **0.807** | **0.859** |
| Held-out Punjab (1,000 patches, 400 with kilns) | Majority label | 0.400 | 0.400 | 1.000 | 0.571 |
| | Random Forest on colour | 0.602 | 0.503 | 0.420 | 0.458 |
| | **KilnWatch** | **0.945** | **0.914** | **0.952** | **0.933** |

On the original SentinelKilnDB patches the model scores F1 0.970 on the Peshawar valley and 0.985 on held-out Punjab. Those patches are easier, because their empty patches are desert.

**Does it transfer to KP?** Partly. Precision holds (0.918 in KP against 0.914 in Punjab), so when it raises an alarm in KP, it is usually right. Recall drops: it misses 19% of KP kiln patches against 5% in Punjab. Fixed-chimney kiln detection mAP50 falls from 0.79 to 0.66 on same-source patches.

Box-level detection, mAP50:

| Test set | Fixed chimney (FCBK) | Zigzag |
|---|---|---|
| Held-out Punjab, SentinelKilnDB images | 0.858 | 0.360 |
| Held-out Punjab, same-source images | 0.790 | 0.276 |
| Peshawar valley, SentinelKilnDB images | 0.808 | no Zigzag kilns in KP data |
| Peshawar valley, same-source images | 0.659 | no Zigzag kilns in KP data |

False alarms on town and city patches: 10 of 200 in the Peshawar valley (5%) and 21 of 200 in Punjab (11%). Before we added urban negatives to training, these were 14 and 34.

**Peshawar 2026.** 217 kilns found; 132 match the 2024 list. 154 kilns in the 2024 list were not detected. That number needs care: the two published 2024 kiln lists themselves agree within 150 m for only 59% of Peshawar kilns, and our detections match 62% of the NUST points, so many "missing" kilns are probably position differences between lists, not closures or misses.

Full tables: `results/baselines.csv`, `results/yolo_yolov8s_384_urban_test_patch.csv`, `results/yolo_yolov8s_384_urban_test_detect.csv`.

## Where it fails

See the [failure gallery](results/failure_gallery.md) for 15 labelled examples. In short:

- **False alarms** are buildings and plots with a kiln's size and shape: factory sheds, walled compounds, poultry farms, long narrow plots at village edges. This is why tier 2 says "verify first".
- **Haze hides kilns.** Our same-source images come from the November to February smog season and miss three times as many KP kiln patches as the dataset's own images of the same kilns (62 against 19). Smog season is exactly when inspections matter most.
- **Kiln type is not reliable.** In 280 of 321 Punjab test patches with a Zigzag kiln, a Zigzag kiln is called fixed-chimney. At 10 m both look like the same bright rectangle. Because fixed-chimney kilns get double weight, a misread Zigzag kiln is ranked too high, never too low.
- **Schools are undercounted.** OpenStreetMap maps only part of KP's schools, mostly in cities (for example 9 in Swabi), so most rural kilns show 0 schools nearby and population drives the ranking. We found no open official KP school list with coordinates.
- **Edge filtering has a cost.** Dropping boxes cut off at tile edges removed 59 noisy new detections but also 10 kilns that the 2024 list confirms.

## Next steps

- Use an official school list with coordinates (KP education department EMIS) instead of OpenStreetMap.
- Reduce haze misses: pick the clearest scene per tile with a haze measure, or use a cloud-free median of several dates.
- Check a sample of tier 2 and "missing" kilns by eye on high-resolution imagery to measure how many are real.
- Use the 20 m and infrared Sentinel-2 bands, or add a separate type classifier, to tell Zigzag kilns from fixed-chimney kilns.
- Run all five Peshawar valley districts: `python src/07_detect_2026.py --districts Peshawar Charsadda Mardan Nowshera Swabi ...`.

## Run it

Python 3.11 and an NVIDIA GPU for training (inference also runs on CPU).

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python app/app.py          # demo at http://127.0.0.1:7860, uses models/ and results/ from this repo
```

To rebuild everything, download SentinelKilnDB into `data/raw/sentinelkilndb/`, GADM 4.1 Pakistan level 3, the NUST kiln CSV, and WorldPop 2020 into `data/raw/`, then run the scripts in `src/` in order. Each script's docstring lists what it reads and writes.

## Sources and tools

Data
- SentinelKilnDB: Mondal et al., "SentinelKilnDB: A Large-Scale Dataset and Benchmark for OBB Brick Kiln Detection in South Asia Using Satellite Imagery", NeurIPS 2025 Datasets and Benchmarks. https://huggingface.co/datasets/SustainabilityLabIITGN/SentinelKilnDB (CC BY-NC 4.0). Their recipe for cutting patches is from https://github.com/rishabh-mondal/SENTINELKILNDB_NeurIPS_2025
- Brick Kiln Dataset for Pakistan's IGP Region Using AI (NUST and Oxford), Zenodo record 14036290, https://zenodo.org/records/14036290, and the paper in Scientific Data, 2025
- Sentinel-2 Level-2A imagery (Copernicus), accessed through Microsoft Planetary Computer, https://planetarycomputer.microsoft.com
- GADM 4.1 administrative boundaries, https://gadm.org
- WorldPop 2020 constrained population for Pakistan, https://www.worldpop.org
- OpenStreetMap schools and hospitals (ODbL), via the Overpass API

Software
- Ultralytics YOLOv8, https://github.com/ultralytics/ultralytics
- PyTorch, scikit-learn, GeoPandas, Rasterio, pystac-client, folium, Gradio

AI assistance
- Claude (Anthropic) was used as a coding assistant to write and debug the pipeline scripts, check data, and draft this README. All results come from running the code in this repository.

Organisers' material
- Veritas AI International High School Hackathon 2026, Project Briefs
- Project plan: `docs/KilnWatch-project-plan.pdf`
