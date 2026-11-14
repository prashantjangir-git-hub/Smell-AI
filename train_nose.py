"""
Train the electronic nose: a random forest that names a smell from sensor readings.

Simulated nose (default):
    python train_nose.py
    Sniffs every smell in samples.json (plus clean air) 200 times with random
    humidity, strength and noise, trains a random forest, then tests it on
    brand-new sniffs it has never seen.

Real nose (after recording with record_bme688.py, see README Part 3):
    python train_nose.py --real data/real
    Trains on your BME688 recordings and tests on a whole session (day) the
    model never saw, which is the honest way to measure a real sensor.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import sensor

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SIM_PATH = MODEL_DIR / "nose.joblib"
REAL_PATH = MODEL_DIR / "nose_real.joblib"


# ---------------------------------------------------------------------------
# Simulated data
# ---------------------------------------------------------------------------


def simulated_dataset(n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    samples = sensor.load_samples()
    classes = list(samples) + [sensor.CLEAN_AIR]
    X, y = [], []
    for name in classes:
        comp = {} if name == sensor.CLEAN_AIR else sensor.components(samples[name])
        for _ in range(n_per_class):
            reading = sensor.sniff(
                comp,
                humidity=rng.uniform(0.2, 0.8),
                intensity=rng.uniform(0.5, 1.5),
                noise=1.0,
                rng=rng,
            )
            X.append(reading)
            y.append(name)
    return np.array(X), np.array(y), classes


def make_forest(seed: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=1, n_jobs=-1, random_state=seed
    )


def train_simulated(n_per_class: int = 200, seed: int = 0, verbose: bool = True) -> dict:
    t0 = time.time()
    X, y, classes = simulated_dataset(n_per_class, seed)
    forest = make_forest(seed)
    forest.fit(sensor.to_features(X), y)

    # Brand-new sniffs, with a different random seed
    Xt, yt, _ = simulated_dataset(max(n_per_class // 4, 20), seed + 1000)
    pred = forest.predict(sensor.to_features(Xt))
    acc = accuracy_score(yt, pred)

    bundle = {
        "kind": "simulated",
        "model": forest,
        "classes": [str(c) for c in forest.classes_],
        "channels": sensor.CHANNELS,
        "test_accuracy": float(acc),
        "n_per_class": n_per_class,
    }
    if verbose:
        print(f"Trained on {len(y)} sniffs of {len(classes)} smells in {time.time() - t0:.1f}s")
        print(f"Accuracy on new sniffs: {acc:.1%}")
        _print_confusion(yt, pred, [str(c) for c in forest.classes_])
    return bundle


# ---------------------------------------------------------------------------
# Real BME688 data
# ---------------------------------------------------------------------------


def load_real(folder: Path):
    import pandas as pd

    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV recordings found in {folder}. Record some with record_bme688.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    feat_cols = [c for c in df.columns if c.startswith("ratio_")]
    if not feat_cols or "label" not in df or "session" not in df:
        raise SystemExit("Recordings need 'label', 'session' and 'ratio_*' columns (made by record_bme688.py).")
    df = df.dropna(subset=feat_cols)
    return df, feat_cols


def train_real(folder: Path, seed: int = 0) -> dict:
    df, feat_cols = load_real(folder)
    X = sensor.to_features(df[feat_cols].to_numpy())
    y = df["label"].astype(str).to_numpy()
    sessions = df["session"].astype(str).to_numpy()
    print(f"{len(df)} readings, {len(set(y))} smells, {len(set(sessions))} sessions, {len(feat_cols)} heater steps")

    # Leave one whole session out: train on the other days, test on this one.
    scores = []
    for s in sorted(set(sessions)):
        test = sessions == s
        if test.all() or len(set(y[~test])) < 2:
            continue
        forest = make_forest(seed).fit(X[~test], y[~test])
        acc = accuracy_score(y[test], forest.predict(X[test]))
        scores.append(acc)
        print(f"  held-out session {s!r}: {acc:.1%}")
    if scores:
        print(f"Average accuracy on unseen sessions: {np.mean(scores):.1%}")
    else:
        print("Only one session recorded: record on at least 2-3 different days for an honest test.")

    forest = make_forest(seed).fit(X, y)
    return {
        "kind": "real",
        "model": forest,
        "classes": [str(c) for c in forest.classes_],
        "feature_columns": feat_cols,
        "test_accuracy": float(np.mean(scores)) if scores else None,
    }


# ---------------------------------------------------------------------------
# Helpers used by the app
# ---------------------------------------------------------------------------


def load_nose(path: Path = SIM_PATH) -> dict:
    return joblib.load(path)


def predict(bundle: dict, reading) -> dict[str, float]:
    """{smell: probability}, best guess first."""
    probs = bundle["model"].predict_proba(sensor.to_features(reading))[0]
    order = np.argsort(probs)[::-1]
    return {str(bundle["classes"][i]): float(probs[i]) for i in order}


def _print_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    short = [c[:7] for c in classes]
    print("\nConfusion matrix (rows = truth, columns = guess):")
    print(" " * 16 + " ".join(f"{s:>7s}" for s in short))
    for name, row in zip(classes, cm):
        print(f"{name:16s}" + " ".join(f"{v:7d}" for v in row))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, help="folder of BME688 recordings to train a real nose")
    ap.add_argument("--sniffs", type=int, default=200, help="simulated sniffs per smell (default 200)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    if args.real:
        bundle = train_real(args.real, args.seed)
        joblib.dump(bundle, REAL_PATH)
        print(f"Saved {REAL_PATH.relative_to(ROOT)}")
    else:
        bundle = train_simulated(args.sniffs, args.seed)
        joblib.dump(bundle, SIM_PATH, compress=3)
        print(f"\nSaved {SIM_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
