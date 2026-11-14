"""Step 4: patch-level baselines (kiln vs no kiln).

Reads:  data/processed/split.csv, data/processed/patches/images/
Writes: results/baselines.csv

Baselines
- majority       predicts the most common training label for every patch. SentinelKilnDB is
                 enriched with kilns, so this predicts "kiln" everywhere, the opposite of a real
                 district where kilns are rare.
- random_forest  Random Forest on simple per-patch colour statistics: mean, std, 10th and 90th
                 percentile of each RGB band (12 numbers per patch).

Each split is scored on three subsets:
- skdb  original SentinelKilnDB patches. Their empty patches are mostly desert (none in KP),
        so precision here is easy to inflate.
- pc    our Planetary Computer patches: hard negatives from farmland and towns plus re-cut kiln
        patches, both from the same source and dates. This is the honest patch-level test.
- all   both together.
"""

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import IMG_DIR, PROC, RESULTS

EVAL_SPLITS = ["val", "test_punjab", "test_kp", "kp_other", "sindh_baloch"]


def features(names: pd.Series) -> np.ndarray:
    feats = []
    for n in names:
        px = np.asarray(Image.open(IMG_DIR / n).convert("RGB"), dtype=np.float32).reshape(-1, 3)
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
    df = pd.read_csv(PROC / "split.csv")
    df["y"] = (df["n_kilns"] > 0).astype(int)
    df = df[df["split"].isin(["train"] + EVAL_SPLITS)].reset_index(drop=True)

    print(f"extracting colour features for {len(df)} patches ...")
    X = features(df["image_name"])
    train = (df["split"] == "train").to_numpy()

    majority = int(df.loc[train, "y"].mode()[0])
    print(f"majority training label: {'kiln' if majority else 'no kiln'} ({df.loc[train, 'y'].mean():.1%} of training patches have a kiln)")

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=0)
    rf.fit(X[train], df.loc[train, "y"])
    rf_pred = rf.predict(X)

    rows = []
    for split in EVAL_SPLITS:
        for source in ["skdb", "pc", "all"]:
            m = (df["split"] == split).to_numpy()
            if source != "all":
                m &= (df["source"] == source).to_numpy()
            if not m.any():
                continue
            y = df.loc[m, "y"].to_numpy()
            rows.append({"model": "majority", "split": split, "source": source, **scores(y, np.full_like(y, majority))})
            rows.append({"model": "random_forest", "split": split, "source": source, **scores(y, rf_pred[m])})

    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "baselines.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
