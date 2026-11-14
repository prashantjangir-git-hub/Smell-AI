"""
Check that everything works, without opening the browser.

    python test_app.py        # or: pytest test_app.py

Run `python smell_model.py` and `python train_nose.py` first.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import sensor  # noqa: E402
import train_nose  # noqa: E402


def test_sensor():
    samples = sensor.load_samples()
    assert len(samples) >= 6
    for name, s in samples.items():
        r = sensor.sniff(sensor.components(s), noise=0)
        assert r.shape == (sensor.N_CHANNELS,), name
        assert r.max() > 0.1, f"{name} barely moves the sensors"
    air = sensor.sniff({}, noise=0)
    assert np.allclose(air, 0, atol=1e-9), "clean air at 50 % humidity should read zero"
    t, tr = sensor.trace(sensor.components(samples["coffee"]), seconds=10, exposure=(2, 8))
    assert tr.shape == (len(t), sensor.N_CHANNELS)


def test_nose():
    assert train_nose.SIM_PATH.exists(), "run: python train_nose.py"
    nose = train_nose.load_nose()
    rng = np.random.default_rng(123)
    samples = sensor.load_samples()
    correct = total = 0
    for name in list(samples) + [sensor.CLEAN_AIR]:
        comp = {} if name == sensor.CLEAN_AIR else sensor.components(samples[name])
        for _ in range(20):
            r = sensor.sniff(comp, humidity=rng.uniform(0.3, 0.7), intensity=rng.uniform(0.7, 1.3), rng=rng)
            correct += next(iter(train_nose.predict(nose, r))) == name
            total += 1
    acc = correct / total
    print(f"    nose accuracy on {total} fresh sniffs: {acc:.1%}")
    assert acc > 0.9


def test_smell_model():
    from smell_model import MODEL_PATH, SmellModel

    assert MODEL_PATH.exists(), "run: python smell_model.py"
    sm = SmellModel.load()
    print(f"    smell model test ROC-AUC: {sm.test_auc:.3f}")
    assert sm.test_auc > 0.7
    samples = sensor.load_samples()
    for name in ("coffee", "mint"):
        words = sm.describe(sensor.components(samples[name]))
        print(f"    {name}: {', '.join(words)}")
        assert len(words) == 5
    print(f"    vanillin: {', '.join(sm.describe({'COc1cc(C=O)ccc1O': 1.0}))}")
    assert sm.scores({"C1CC((": 1.0}).empty
    assert sm.map_position(sensor.components(samples["lemon"])) is not None


def test_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    at.checkbox(key="animate").uncheck().run()
    at.button(key="sniff").click().run()
    assert not at.exception, at.exception
    at.button(key="mix_go").click().run()
    assert not at.exception, at.exception
    at.button(key="game_new").click().run()
    at.button(key="game_lock").click().run()
    assert not at.exception, at.exception
    assert at.session_state["score"]["rounds"] == 1


if __name__ == "__main__":
    failed = 0
    for test in (test_sensor, test_nose, test_smell_model, test_app):
        print(f"{test.__name__} ...")
        try:
            test()
            print("    PASS")
        except Exception as e:  # keep going so you see every problem
            failed += 1
            print(f"    FAIL: {e}")
            traceback.print_exc(limit=2)
    print("\nAll good! 🎉" if not failed else f"\n{failed} test(s) failed.")
    sys.exit(1 if failed else 0)
