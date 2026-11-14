"""
A simulated electronic nose: an array of 8 gas sensors.

Real metal-oxide gas sensors change their electrical resistance when gas
molecules land on their hot surface. Each sensor type (and each heater
temperature) reacts a bit differently, so an array of sensors gives a
"fingerprint" pattern for a smell even though no single sensor knows what it is.

This file fakes that. Each simulated channel reacts to a molecule's:
  * size            (bigger molecules evaporate less and stick differently)
  * oil-loving-ness (how carbon-heavy vs. oxygen/nitrogen-heavy it is)
  * oxygen content  (alcohols, aldehydes, esters ...)
  * sulfur content  (garlic, coffee thiols ...)
  * aromatic rings, double bonds and C=O groups

It also adds the messy stuff real sensors suffer from: humidity, changing
concentration, drift and noise.

No RDKit needed here: we read the SMILES text with a tiny parser, so this file
only needs numpy. To use a real sensor instead, see README Part 3.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SAMPLES_PATH = ROOT / "samples.json"

CLEAN_AIR = "clean air"

CHANNELS = [
    "S1 SnO2 cool",      # general purpose, likes small oxygenated molecules
    "S2 SnO2 hot",       # general purpose, likes bigger hydrocarbons
    "S3 WO3",            # oxygen-rich molecules
    "S4 Pt-doped",       # very sensitive to sulfur
    "S5 Pd-doped",       # small, volatile molecules
    "S6 ZnO",            # aromatic rings
    "S7 polymer",        # oily, carbon-heavy molecules
    "S8 humidity-ish",   # mostly water, a little of everything
]
N_CHANNELS = len(CHANNELS)

# ---------------------------------------------------------------------------
# 1. Read a SMILES string and count what's in it
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\[[^\]]+\]|Br|Cl|[BCNOPSFI]|[bcnops]")
_BRACKET_ELEMENT = re.compile(r"\[\d*([A-Z][a-z]?|[a-z]{1,2})")


def _atoms(smiles: str) -> list[str]:
    atoms = []
    for tok in _TOKEN.findall(smiles):
        if tok.startswith("["):
            m = _BRACKET_ELEMENT.match(tok)
            if not m:
                continue
            tok = m.group(1)
            if tok == "H":  # explicit hydrogen like [H]
                continue
        atoms.append(tok)
    return atoms


def descriptors(smiles: str) -> np.ndarray:
    """A small, chemistry-flavoured description of one molecule."""
    atoms = _atoms(smiles)
    heavy = max(len(atoms), 1)
    upper = [a.capitalize() for a in atoms]
    n_c = upper.count("C")
    n_o = upper.count("O")
    n_n = upper.count("N")
    n_s = upper.count("S")
    n_hal = sum(upper.count(x) for x in ("F", "Cl", "Br", "I"))
    n_arom = sum(1 for a in atoms if a.islower())
    n_double = smiles.count("=") + smiles.count("#")
    n_carbonyl = len(re.findall(r"=O|O=", smiles))  # C=O groups: aldehydes, ketones, esters

    size = heavy / 12.0
    oil_loving = (n_c + 0.5 * n_s + 0.5 * n_hal - 1.5 * n_o - 1.5 * n_n) / heavy
    return np.array(
        [
            size,
            oil_loving,
            n_o / heavy,
            n_n / heavy,
            n_s / heavy,
            n_arom / heavy,
            (n_double + n_arom / 2) / heavy,
            n_carbonyl / heavy,
            1.0,  # bias
        ],
        dtype=float,
    )


def volatility(smiles: str) -> float:
    """Small molecules evaporate more, so more of them reach the sensor."""
    heavy = max(len(_atoms(smiles)), 1)
    return math.exp(-heavy / 10.0) * 3.0


# How strongly each channel (row) reacts to each descriptor (column):
#              size  oil    O     N     S    arom  unsat  C=O   bias
_W = np.array(
    [
        [-0.6, -0.3,  2.2,  0.8,  0.6,  0.2,  0.4,  2.5,  0.3],   # S1 SnO2 cool
        [ 0.9,  1.4,  0.3,  0.4,  0.8,  0.4,  0.6,  0.0,  0.1],   # S2 SnO2 hot
        [ 0.1, -0.8,  4.5,  0.2,  0.1, -1.0, -0.2,  0.5,  0.0],   # S3 WO3
        [ 0.0,  0.1,  0.0,  1.0,  9.0,  0.2,  0.3,  0.0, -0.4],   # S4 Pt-doped
        [-1.5,  0.5,  0.6,  1.2,  1.5, -0.4,  3.0,  0.0,  0.6],   # S5 Pd-doped
        [ 0.3,  0.2,  0.5,  1.5,  0.4,  3.2,  1.2, -0.5, -0.3],   # S6 ZnO
        [ 1.2,  2.2, -0.8, -0.4,  0.6,  0.8,  0.2,  0.0, -0.2],   # S7 polymer
        [-0.2, -0.4,  1.2,  0.4,  0.2,  0.1,  0.1,  0.0,  0.2],   # S8 humidity-ish
    ]
)

# How much each channel is fooled by humidity (per unit change from 50 % RH)
_HUMIDITY = np.array([0.20, 0.10, 0.25, 0.05, 0.12, 0.08, 0.04, 0.90])

# How fast each channel responds (seconds) - used for the live trace
_TAU = np.array([3.0, 5.0, 4.0, 2.5, 2.0, 6.0, 8.0, 1.5])


def _softplus(x):
    return np.log1p(np.exp(x))


def molecule_response(smiles: str) -> np.ndarray:
    """Noise-free response of all 8 channels to one pure molecule."""
    return _softplus(_W @ descriptors(smiles)) * volatility(smiles)


# ---------------------------------------------------------------------------
# 2. Samples (real smells = lists of real molecules)
# ---------------------------------------------------------------------------


def load_samples(path: Path | str = SAMPLES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def components(sample: dict) -> dict[str, float]:
    """{smiles: weight} for one sample from samples.json."""
    out: dict[str, float] = {}
    for m in sample["molecules"]:
        out[m["smiles"]] = out.get(m["smiles"], 0.0) + float(m["weight"])
    return out


def mix(parts: list[tuple[dict[str, float], float]]) -> dict[str, float]:
    """Blend several component dicts: [(components, fraction), ...]."""
    out: dict[str, float] = {}
    total = sum(max(f, 0.0) for _, f in parts) or 1.0
    for comp, frac in parts:
        for smi, w in comp.items():
            out[smi] = out.get(smi, 0.0) + w * max(frac, 0.0) / total
    return out


# ---------------------------------------------------------------------------
# 3. Sniffing
# ---------------------------------------------------------------------------


def _ideal(comp: dict[str, float]) -> np.ndarray:
    total = np.zeros(N_CHANNELS)
    for smi, w in comp.items():
        total += w * molecule_response(smi)
    return total


def sniff(
    comp: dict[str, float],
    humidity: float = 0.5,
    intensity: float = 1.0,
    noise: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    One reading from the 8-channel array.

    comp       {smiles: weight}; empty dict = clean air
    humidity   0..1 relative humidity (0.5 = the clean-air baseline)
    intensity  how strong the smell is (1 = normal)
    noise      1 = realistic noise, 0 = perfect sensor
    Returns the response of each channel: 0 = no change, ~1 = saturated.
    """
    rng = rng or np.random.default_rng()
    signal = _ideal(comp) * intensity
    if noise > 0:
        gain = 1.0 + rng.normal(0, 0.04 * noise, N_CHANNELS)   # drift / ageing
        signal = signal * gain
    response = 1.0 - np.exp(-0.12 * signal)                   # sensors saturate
    response = response + _HUMIDITY * (humidity - 0.5)
    if noise > 0:
        response = response + rng.normal(0, 0.008 * noise, N_CHANNELS)
        response = response * (1.0 + rng.normal(0, 0.02 * noise, N_CHANNELS))
    return response


def trace(
    comp: dict[str, float],
    seconds: float = 40.0,
    exposure: tuple[float, float] = (5.0, 25.0),
    dt: float = 0.5,
    humidity: float = 0.5,
    intensity: float = 1.0,
    noise: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    A time series: clean air, then the smell, then clean air again.
    Returns (times, readings) with readings shaped (T, 8).
    """
    rng = rng or np.random.default_rng()
    target = sniff(comp, humidity, intensity, noise, rng)
    baseline = _HUMIDITY * (humidity - 0.5)
    times = np.arange(0, seconds + 1e-9, dt)
    level = baseline.copy()
    rows = []
    for t in times:
        goal = target if exposure[0] <= t < exposure[1] else baseline
        level = level + (goal - level) * (1 - np.exp(-dt / _TAU))
        jitter = rng.normal(0, 0.006 * noise, N_CHANNELS) if noise > 0 else 0
        rows.append(level + jitter)
    return times, np.array(rows)


# ---------------------------------------------------------------------------
# 4. Features for the classifier (shared by the simulated and the real nose)
# ---------------------------------------------------------------------------


def to_features(reading) -> np.ndarray:
    """
    Raw channel values + the *shape* of the pattern.
    The shape (reading divided by its size) stays the same when a smell is
    weaker or stronger, which helps the nose generalise.
    """
    r = np.asarray(reading, dtype=float)
    if r.ndim == 1:
        r = r[None, :]
    norm = np.linalg.norm(r, axis=1, keepdims=True) + 1e-6
    return np.hstack([r, r / norm])


if __name__ == "__main__":
    samples = load_samples()
    rng = np.random.default_rng(0)
    print(f"{'sample':16s} " + " ".join(f"{c.split()[0]:>5s}" for c in CHANNELS))
    for name, s in samples.items():
        r = sniff(components(s), rng=rng)
        print(f"{name:16s} " + " ".join(f"{v:5.2f}" for v in r))
    r = sniff({}, rng=rng)
    print(f"{CLEAN_AIR:16s} " + " ".join(f"{v:5.2f}" for v in r))
