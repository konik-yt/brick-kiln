---
name: rerun-deploy
description: Use when the KilnWatch final model, its threshold or image size changes, or when the 2026 tiles, priority list, map or demo app are out of date and must be rebuilt so they match each other.
disable-model-invocation: true
---

# Rerun deploy

Rebuilds the Peshawar 2026 detections, the tiered priority list, the map and the running demo app from the final model in `src/config.py`, then proves they agree.

## Before you start

- The model, image size and threshold come only from `FINAL_MODEL`, `FINAL_IMGSZ`, `FINAL_CONF` in `src/config.py`. To change them, edit those three lines; never pass a different `--conf` by hand.
- A new `FINAL_CONF` must come from the `val` split: `.venv/Scripts/python src/06_evaluate.py --name <model> --imgsz <size> --splits val`. Then run the `leakage-auditor` subagent before going on.

## Steps

Run from the repo root.

1. Detect: `.venv/Scripts/python src/07_detect_2026.py --districts Peshawar`
   Cached tiles in `data/processed/s2_2026/images/` are reused. Delete that folder only when the imagery itself should change.
2. Rank: `.venv/Scripts/python src/08_priority.py`
3. Map: `.venv/Scripts/python app/build_map.py`
4. Restart the app. It reads the CSV and map only at start-up, so stop the old one first:
   PowerShell: `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'app[\\/]app\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }`
   Then start it as a background job: `.venv/Scripts/python app/app.py > app_run.log 2>&1`
   Wait until `curl -s -o /dev/null -w "%{http_code}" --retry 30 --retry-connrefused --retry-delay 2 http://127.0.0.1:7860/` prints 200.
5. Check: `.venv/Scripts/python .claude/skills/rerun-deploy/check_deploy.py`
   It must print `ALL CHECKS PASSED`. Each FAIL line names the step to redo.
6. Update numbers that changed: the kiln and tier counts in `README.md` ("What we built" and "Peshawar 2026"). The app reads the CSV, so it needs no change. If `FINAL_CONF` or `FINAL_MODEL` changed, the test-set tables in `README.md` are for the old setting; say so to the user rather than rescoring the test splits again.
7. Commit `results/`, `app/kiln_map.html` and any README change, and push.

## Common mistakes

| Mistake | Result |
|---|---|
| Passing `--conf` or `--name` by hand | List, map and app disagree; `check_deploy.py` fails on confidence |
| Skipping the app restart | App still shows the old list and map |
| Calling numpy input to `model.predict` without flipping RGB to BGR | App misses kilns the model finds; `check_deploy.py` reports app vs file path mismatch |
| Re-scoring test splits after a threshold change | Uses the test set a second time; the README must say which run is final |
