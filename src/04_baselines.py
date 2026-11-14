"""Step 4: patch-level baselines (kiln vs no kiln).

Reads:  data/processed/patches.csv, data/processed/split.csv, data/processed/patches/images/
Writes: results/baselines.csv

Baselines
- majority       predicts the most common training label for every patch. SentinelKilnDB is
                 enriched with kilns (about 75% of Punjab training patches contain one), so this
                 predicts "kiln" everywhere, the opposite of a real district where kilns are rare.
- random_forest  Random Forest on simple per-patch colour statistics: mean, std, 10th and 90th
                 percentile of each RGB band (12 numbers per patch).

The test_kp split has no empty patches in SentinelKilnDB, so precision there is trivially 1.0
for any model that ever says "kiln". Read recall for test_kp; read precision on test_punjab.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
EVAL_SPLITS = ["val", "test_punjab", "test_kp", "kp_other", "sindh_baloch"]


def features(names: pd.Series) -> np.ndarray:
    feats = []
    for n in names:
        px = np.asarray(Image.open(PROC / "patches" / "images" / n).convert("RGB"), dtype=np.float32).reshape(-1, 3)
        feats.append(np.concatenate([px.mean(0), px.std(0), np.percentile(px, 10, axis=0), np.percentile(px, 90, axis=0)]))
    return np.stack(feats)


def scores(y_true, y_pred) -> dict:
    return {
        "patches": len(y_true),
        "kiln_patches": int(y_true.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main() -> None:
    df = pd.read_csv(PROC / "patches.csv").merge(pd.read_csv(PROC / "split.csv"), on="image_name")
    df["y"] = (df["n_kilns"] > 0).astype(int)
    df = df[df["split"].isin(["train"] + EVAL_SPLITS)].reset_index(drop=True)

    print(f"extracting colour features for {len(df)} patches ...")
    X = features(df["image_name"])
    train = df["split"] == "train"

    majority = int(df.loc[train, "y"].mode()[0])
    print(f"majority training label: {'kiln' if majority else 'no kiln'} ({df.loc[train, 'y'].mean():.1%} of training patches have a kiln)")

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=0)
    rf.fit(X[train], df.loc[train, "y"])

    rows = []
    for split in EVAL_SPLITS:
        m = (df["split"] == split).to_numpy()
        y = df.loc[m, "y"].to_numpy()
        rows.append({"model": "majority", "split": split, **scores(y, np.full_like(y, majority))})
        rows.append({"model": "random_forest", "split": split, **scores(y, rf.predict(X[m]))})

    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "baselines.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
