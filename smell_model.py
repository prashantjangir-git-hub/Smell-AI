"""
The smell model: reads a molecule and predicts what it smells like.

    python smell_model.py            # download data, tune, train, test, save (1-3 min)
    python smell_model.py --fast     # skip tuning (about 30 s)
    python smell_model.py --csv my_copy_of_the_dataset.csv
    python smell_model.py --predict "COc1cc(C=O)ccc1O"     # try one molecule

How it works
  1. Each molecule (written as SMILES text) becomes a Morgan fingerprint:
     a list of counts of the small chemical fragments it contains.
  2. For each of the 138 smell words (fruity, minty, smoky ...) a logistic
     regression learns which fragments go with that word.
  3. We test on 20 % of molecules the model never saw, using ROC-AUC
     (0.5 = guessing, 1.0 = perfect).
  4. We also build a 2-D "odor map" of all 4,983 molecules so the app can show
     which smells sit close to each other.

Data: the OpenPOM curated GoodScents + Leffingwell dataset (4,983 molecules
labelled by perfume experts). It is downloaded on first run and not stored
in git.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATASET_PATH = DATA_DIR / "openpom_gs_lf_4983.csv"
MODEL_PATH = MODEL_DIR / "smell_model.joblib"
MAP_PATH = DATA_DIR / "odor_map.csv"

DATASET_URLS = [
    "https://raw.githubusercontent.com/BioMachineLearning/openpom/main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv",
    "https://raw.githubusercontent.com/BioMachineLearning/openpom/master/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv",
]

FP_RADIUS = 2
FP_BITS = 2048
C_GRID = (0.03, 0.1, 0.3)
DEFAULT_C = 0.1

# Broad smell families, used to colour the odor map. Checked in this order.
FAMILIES: dict[str, set[str]] = {
    "sulfurous & savoury": {"sulfurous", "alliaceous", "garlic", "onion", "meaty", "beefy", "cabbage",
                            "fishy", "savory", "gassy", "radish", "potato", "cooked"},
    "animal & funky": {"animal", "cheesy", "sweaty", "musk", "fermented", "leathery", "musty", "sour"},
    "smoky & roasted": {"smoky", "burnt", "roasted", "coffee", "cocoa", "chocolate", "popcorn", "nutty",
                        "hazelnut", "malty", "tobacco", "phenolic", "medicinal"},
    "minty & cooling": {"mint", "cooling", "camphoreous"},
    "citrus": {"citrus", "lemon", "orange", "grapefruit", "bergamot"},
    "floral": {"floral", "rose", "jasmin", "lily", "muguet", "violet", "lavender", "hyacinth", "geranium",
               "orangeflower", "orris", "chamomile", "hawthorn", "honey"},
    "fruity": {"fruity", "apple", "apricot", "banana", "berry", "black currant", "cherry", "grape",
               "melon", "peach", "pear", "pineapple", "plum", "raspberry", "strawberry", "tropical",
               "ripe", "juicy", "fruit skin", "winey", "brandy", "cognac", "rummy", "alcoholic"},
    "spicy & warm": {"spicy", "cinnamon", "clove", "anisic", "warm", "aromatic"},
    "woody & resinous": {"woody", "cedar", "sandalwood", "vetiver", "pine", "amber", "balsamic",
                         "terpenic", "dry", "cortex"},
    "green & herbal": {"green", "grassy", "leafy", "herbal", "cucumber", "vegetable", "weedy", "hay",
                       "tea", "celery", "tomato", "earthy", "mushroom"},
    "sweet & creamy": {"sweet", "vanilla", "caramellic", "creamy", "buttery", "dairy", "milky",
                       "coconut", "lactonic", "coumarinic", "powdery"},
}
OTHER_FAMILY = "chemical & other"


def family_of(words) -> str:
    words = set(words)
    for fam, members in FAMILIES.items():
        if words & members:
            return fam
    return OTHER_FAMILY


# ---------------------------------------------------------------------------
# Molecules -> fingerprints
# ---------------------------------------------------------------------------

_GEN = None


def _chem():
    try:
        from rdkit import Chem, RDLogger
    except ImportError:  # pragma: no cover - only hit when rdkit is missing
        sys.exit("RDKit is missing. Run:  pip install -r requirements.txt")
    RDLogger.DisableLog("rdApp.*")
    return Chem


def mol_from_smiles(smiles: str):
    """RDKit molecule, or None if the SMILES text is not a valid molecule."""
    if not smiles or not isinstance(smiles, str):
        return None
    return _chem().MolFromSmiles(smiles.strip())


def fingerprint(smiles: str) -> np.ndarray | None:
    """log(1 + count) of each hashed Morgan fragment, or None if invalid."""
    global _GEN
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdFingerprintGenerator

        if _GEN is None:
            _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
        counts = _GEN.GetCountFingerprintAsNumPy(mol).astype(np.float32)
    except (ImportError, AttributeError):  # older RDKit
        from rdkit.Chem import AllChem

        fp = AllChem.GetHashedMorganFingerprint(mol, FP_RADIUS, nBits=FP_BITS)
        counts = np.zeros(FP_BITS, dtype=np.float32)
        for idx, v in fp.GetNonzeroElements().items():
            counts[idx] = v
    return np.log1p(counts)


def featurize(smiles_list) -> tuple[np.ndarray, np.ndarray]:
    """Fingerprints for many molecules. Returns (X, ok) where ok marks valid ones."""
    rows, ok = [], []
    for s in smiles_list:
        fp = fingerprint(s)
        ok.append(fp is not None)
        rows.append(fp if fp is not None else np.zeros(FP_BITS, dtype=np.float32))
    return np.vstack(rows) if rows else np.zeros((0, FP_BITS), np.float32), np.array(ok, dtype=bool)


def draw_svg(smiles: str, size: int = 220) -> str | None:
    """A picture of the molecule as SVG text (None if it can't be drawn)."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem.Draw import rdMolDraw2D

        d = rdMolDraw2D.MolDraw2DSVG(size, int(size * 0.75))
        d.DrawMolecule(mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def download_dataset(dest: Path = DATASET_PATH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in DATASET_URLS:
        try:
            print(f"Downloading dataset from {url.split('/')[2]} ...")
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if len(data) < 10_000:
                raise ValueError("file too small")
            dest.write_bytes(data)
            print(f"Saved {dest.relative_to(ROOT)} ({len(data) / 1e6:.1f} MB)")
            return dest
        except Exception as e:  # try the next mirror
            last_err = e
    sys.exit(
        f"Could not download the dataset ({last_err}).\n"
        "Download curated_GS_LF_merged_4983.csv from the OpenPOM GitHub repo by hand,\n"
        "then run:  python smell_model.py --csv path/to/curated_GS_LF_merged_4983.csv"
    )


def load_dataset(csv_path: Path | None = None):
    """Returns (smiles, Y, labels) where Y is a 0/1 matrix molecules x smell words."""
    path = Path(csv_path) if csv_path else DATASET_PATH
    if not path.exists():
        if csv_path:
            sys.exit(f"File not found: {path}")
        download_dataset(path)
    df = pd.read_csv(path)
    smiles_cols = [c for c in df.columns if "smiles" in c.lower()]
    if not smiles_cols:
        sys.exit("Could not find a SMILES column in the dataset.")
    smiles_col = smiles_cols[0]
    labels = []
    for c in df.columns:
        if c == smiles_col:
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().all() and set(col.unique()) <= {0, 1}:
            labels.append(c)
    if len(labels) < 10:
        sys.exit("Could not find the 0/1 smell-label columns in the dataset.")
    Y = df[labels].to_numpy(dtype=np.int8)
    return df[smiles_col].astype(str).tolist(), Y, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class SmellWordModel:
    """
    One logistic regression per smell word, trained one after another.

    The weights live in one matrix (row j = smell word j), so a word's scores can
    never get mixed up with another word's. Predicting is just
    sigmoid(X @ W.T + b): plain numpy, nothing hidden.
    """

    def __init__(self, C: float = DEFAULT_C, solver: str = "liblinear"):
        self.C = C
        self.solver = solver

    def fit(self, X, Y, verbose: bool = False):
        from sklearn.linear_model import LogisticRegression

        X = _as_float64(X)
        Y = np.asarray(Y)
        n_words = Y.shape[1]
        self.W = np.zeros((n_words, X.shape[1]), dtype=np.float64)
        self.b = np.full(n_words, -10.0)
        for j in range(n_words):
            y = Y[:, j].astype(int)
            if y.min() == y.max():  # word never (or always) appears here
                self.b[j] = 10.0 if y.max() == 1 else -10.0
                continue
            clf = LogisticRegression(C=self.C, solver=self.solver, class_weight="balanced", max_iter=2000)
            clf.fit(X, y)
            # make sure we take the weights for "yes, it smells like this word"
            sign = 1.0 if list(clf.classes_) == [0, 1] else -1.0
            self.W[j] = sign * clf.coef_.ravel()
            self.b[j] = sign * float(clf.intercept_[0])
            if verbose and (j + 1) % 20 == 0:
                print(f"\r    {j + 1}/{n_words} smell words", end="", flush=True)
        if verbose:
            print(f"\r    {n_words}/{n_words} smell words")
        return self

    def to_arrays(self) -> dict:
        """Plain numbers only, so the saved file never depends on class names."""
        return {"W": self.W.astype(np.float32), "b": self.b.astype(np.float64), "C": self.C, "solver": self.solver}

    @classmethod
    def from_arrays(cls, d: dict) -> "SmellWordModel":
        m = cls(d.get("C", DEFAULT_C), d.get("solver", "liblinear"))
        m.W = np.asarray(d["W"], dtype=np.float64)
        m.b = np.asarray(d["b"], dtype=np.float64)
        return m

    def predict_proba(self, X) -> np.ndarray:
        z = _as_float64(X) @ self.W.T + self.b
        z = np.asarray(z)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _as_float64(X):
    from scipy.sparse import issparse

    return X.astype(np.float64) if issparse(X) else np.asarray(X, dtype=np.float64)


def macro_auc(Y_true: np.ndarray, P: np.ndarray, labels=None):
    """Average ROC-AUC over smell words that appear in the test set."""
    from sklearn.metrics import roc_auc_score

    per = {}
    for j in range(Y_true.shape[1]):
        y = Y_true[:, j]
        if 0 < y.sum() < len(y):
            per[labels[j] if labels else j] = float(roc_auc_score(y, P[:, j]))
    return float(np.mean(list(per.values()))), per


SOLVER = "liblinear"


def _fit(X, Y, C, verbose=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SmellWordModel(C, SOLVER).fit(X, Y, verbose=verbose)


def train(csv_path: Path | None = None, fast: bool = False, seed: int = 42, verbose: bool = True) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    smiles, Y, labels = load_dataset(csv_path)
    from scipy.sparse import csr_matrix

    X, ok = featurize(smiles)
    smiles = [s for s, k in zip(smiles, ok) if k]
    X, Y = csr_matrix(X[ok]), Y[ok]  # fingerprints are mostly zeros: sparse is much faster
    log(f"{len(smiles)} molecules, {len(labels)} smell words, {X.shape[1]} fragment features")
    same = np.array_equal(fingerprint(smiles[0]), fingerprint(smiles[0]))
    log(f"  check: fingerprints repeatable: {same}")
    log(f"  check: avg {X.getnnz(axis=1).mean():.1f} fragments per molecule, "
        f"avg {Y.sum(axis=1).mean():.1f} smell words per molecule, "
        f"first molecule {smiles[0]!r} -> {[labels[j] for j in np.flatnonzero(Y[0])][:5]}")

    idx = np.arange(len(smiles))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)

    global SOLVER
    sub, val = train_test_split(tr, test_size=0.15, random_state=seed)
    best_C, best = DEFAULT_C, -1.0
    for solver in ("liblinear", "lbfgs"):
        SOLVER = solver
        for C in (C_GRID if not fast else (DEFAULT_C,)):
            auc, _ = macro_auc(Y[val], _fit(X[sub], Y[sub], C).predict_proba(X[val]))
            log(f"  tuning: solver={solver} C={C:<5} validation ROC-AUC {auc:.3f}")
            if auc > best:
                best, best_C = auc, C
        if best >= 0.6:
            break
        log("  check: that looks like guessing - trying the backup solver ...")

    model = _fit(X[tr], Y[tr], best_C, verbose=verbose)
    train_auc, _ = macro_auc(Y[tr], model.predict_proba(X[tr]))
    test_auc, per_label = macro_auc(Y[te], model.predict_proba(X[te]), labels)
    log(f"  check: ROC-AUC on the molecules it trained on: {train_auc:.3f} (should be above 0.85)")
    log(f"Test ROC-AUC on {len(te)} unseen molecules: {test_auc:.3f}   (0.5 = guessing, 1 = perfect)")
    if test_auc < 0.65:
        log("  WARNING: that's close to guessing. Something is off - please share the lines above.")

    if not fast:
        log("Refitting on all molecules ...")
        model = _fit(X, Y, best_C, verbose=verbose)

    # Odor map: squash each molecule's 138 smell scores down to 2-D.
    P = model.predict_proba(X)
    pca = PCA(n_components=2, random_state=seed).fit(_logit(P))
    xy = pca.transform(_logit(P))
    words = [[labels[j] for j in np.flatnonzero(row)] for row in Y]
    odor_map = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "smiles": smiles,
            "family": [family_of(w) for w in words],
            "smells": [", ".join(w[:6]) for w in words],
        }
    )

    log(f"Done in {time.time() - t0:.0f}s")
    return {
        "model": model.to_arrays(),
        "labels": labels,
        "C": best_C,
        "solver": SOLVER,
        "test_auc": test_auc,
        "per_label_auc": per_label,
        "pca": pca,
        "odor_map": odor_map,
        "label_rate": Y.mean(axis=0),
        "n_molecules": len(smiles),
    }


def _logit(P):
    P = np.clip(P, 1e-4, 1 - 1e-4)
    return np.log(P / (1 - P))


# ---------------------------------------------------------------------------
# Using a trained model
# ---------------------------------------------------------------------------


class SmellModel:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        m = bundle["model"]
        self.model = SmellWordModel.from_arrays(m) if isinstance(m, dict) else m
        self.labels: list[str] = list(bundle["labels"])
        self.test_auc: float = bundle["test_auc"]
        self.odor_map: pd.DataFrame = bundle["odor_map"]
        self.pca = bundle["pca"]

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "SmellModel":
        # Files saved by an earlier version stored the model under __main__;
        # point that name at the real class so they still load.
        import __main__

        if not hasattr(__main__, "SmellWordModel"):
            __main__.SmellWordModel = SmellWordModel
        return cls(joblib.load(path))

    def predict(self, smiles_list) -> tuple[np.ndarray, np.ndarray]:
        """(P, ok): P[i, j] = score that molecule i smells like word j."""
        X, ok = featurize(list(smiles_list))
        P = self.model.predict_proba(X) if len(X) else np.zeros((0, len(self.labels)))
        P[~ok] = np.nan
        return P, ok

    def scores(self, comp: dict[str, float]) -> pd.Series:
        """Smell-word scores for a mixture {smiles: weight}, highest first."""
        smiles = list(comp)
        if not smiles:
            return pd.Series(dtype=float)
        P, ok = self.predict(smiles)
        w = np.array([comp[s] for s in smiles], dtype=float)[ok]
        if w.sum() <= 0:
            return pd.Series(dtype=float)
        avg = (P[ok] * w[:, None]).sum(axis=0) / w.sum()
        return pd.Series(avg, index=self.labels).sort_values(ascending=False)

    def describe(self, comp: dict[str, float], top_k: int = 5) -> list[str]:
        words = [w for w in self.scores(comp).index if w != "odorless"]
        return words[:top_k]

    def map_position(self, comp: dict[str, float]) -> tuple[float, float] | None:
        s = self.scores(comp)
        if s.empty:
            return None
        vec = s.reindex(self.labels).to_numpy()[None, :]
        x, y = self.pca.transform(_logit(vec))[0]
        return float(x), float(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, help="use a local copy of the dataset")
    ap.add_argument("--fast", action="store_true", help="skip tuning and the final refit")
    ap.add_argument("--predict", metavar="SMILES", help="describe one molecule with a trained model")
    args = ap.parse_args()

    if args.predict:
        sm = SmellModel.load()
        s = sm.scores({args.predict: 1.0})
        if s.empty:
            sys.exit("That SMILES is not a valid molecule.")
        for word, v in s.head(10).items():
            print(f"{word:15s} {'#' * int(v * 30):30s} {v:.2f}")
        return

    bundle = train(args.csv, fast=args.fast)
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    bundle["odor_map"].to_csv(MAP_PATH, index=False)
    print(f"Saved {MODEL_PATH.relative_to(ROOT)} and {MAP_PATH.relative_to(ROOT)}")

    sm = SmellModel(bundle)
    import sensor

    print("\nWhat the model says about each sample:")
    for name, s in sensor.load_samples().items():
        print(f"  {name:15s} {', '.join(sm.describe(sensor.components(s)))}")


if __name__ == "__main__":
    main()
