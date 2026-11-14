"""Shared settings for the KilnWatch pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
IMG_DIR = PROC / "patches" / "images"
LBL_DIR = PROC / "patches" / "labels"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

CLASS_NAMES = {0: "CFCBK", 1: "FCBK", 2: "Zigzag"}

# District split (step 3). Punjab districts are held out whole.
VAL_DISTRICTS = ["Gujranwala1", "Multan", "Bahawalnagar", "Chakwal"]
TEST_PUNJAB_DISTRICTS = ["Sheikhupura", "Sahiwal", "RahimyarKhan", "Jhelum"]
TEST_KP_DISTRICTS = ["Peshawar", "Charsadda", "Mardan", "Nowshera", "Swabi"]

# Imagery window of SentinelKilnDB (dataset card)
S2_START, S2_END = "2023-11-01", "2024-02-29"


def assign_split(province: str, district: str) -> str:
    if district in TEST_KP_DISTRICTS:
        return "test_kp"
    if province == "Khyber-Pakhtunkhwa":
        return "kp_other"
    if province in ("Sindh", "Balochistan"):
        return "sindh_baloch"
    if district in TEST_PUNJAB_DISTRICTS:
        return "test_punjab"
    if district in VAL_DISTRICTS:
        return "val"
    return "train"


def weights_path(name: str) -> Path:
    """Trained weights: runs/<name>/weights/best.pt after training here, else the copy in models/<name>.pt."""
    local = ROOT / "runs" / name / "weights" / "best.pt"
    return local if local.exists() else MODELS / f"{name}.pt"
