"""
Record smells from a real BME688 (running firmware/esp32_bme688) into CSV files
that `python train_nose.py --real data/real` can learn from.

    pip install -r requirements-hardware.txt
    python record_bme688.py --list-ports
    python record_bme688.py --port COM3 --label coffee --session day1
    python record_bme688.py --port /dev/ttyUSB0 --label "clean air" --session day1

Each recording:
  1. Baseline: the sensor sits in clean air for --baseline seconds (default 120).
     The median resistance at each heater step becomes R0.
  2. You put the sensor in the jar with the smell and press Enter.
  3. It records for --minutes (default 30). Every heater cycle becomes one row,
     with each resistance divided by R0 (columns ratio_200, ratio_250, ...).

Saved to data/real/<session>_<label>.csv. Record each smell on at least 3
different days (sessions) so you can test on a day the model never saw.

No hardware yet? Try the whole workflow with fake data:
    python record_bme688.py --simulate --label coffee --session day1 --minutes 1 --baseline 10
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "real"
DEFAULT_STEPS = [200, 250, 300, 350]


def parse_line(line: str, n_steps: int = len(DEFAULT_STEPS)):
    """'BME,ms,temp,hum,r1,r2,...' -> (temp, hum, [r1, r2, ...]) or None."""
    parts = line.strip().split(",")
    if len(parts) != 4 + n_steps or parts[0] != "BME":
        return None
    try:
        temp, hum = float(parts[2]), float(parts[3])
        res = [float(x) for x in parts[4:]]
    except ValueError:
        return None
    if any(r <= 0 for r in res):
        return None
    return temp, hum, res


# ---------------------------------------------------------------------------
# Line sources: a real serial port or a fake sensor
# ---------------------------------------------------------------------------


class SerialSource:
    def __init__(self, port: str, baud: int = 115200):
        try:
            import serial
        except ImportError:
            sys.exit("pyserial is missing. Run:  pip install -r requirements-hardware.txt")
        self.ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # the ESP32 resets when the port opens

    def lines(self):
        while True:
            raw = self.ser.readline().decode("utf-8", errors="ignore")
            if raw:
                if raw.startswith("#"):
                    print(raw.strip())
                yield raw

    def set_smell(self, label: str):
        pass


class SimulatedSource:
    """Turns sensor.py's simulated nose into fake BME688 lines (one per 'cycle')."""

    R0 = np.array([80_000.0, 45_000.0, 25_000.0, 12_000.0])
    CHANNELS = [0, 1, 2, 5]   # which simulated channels stand in for each heater step

    def __init__(self, session: str, fast: bool = True):
        import sensor

        self.sensor = sensor
        self.samples = sensor.load_samples()
        self.rng = np.random.default_rng(zlib.crc32(session.encode()))
        self.drift = self.rng.normal(1.0, 0.05, 4)   # each "day" the sensor is a bit different
        self.comp = {}
        self.fast = fast
        self.t0 = time.time()

    def set_smell(self, label: str):
        self.comp = {} if label == self.sensor.CLEAN_AIR else self.sensor.components(self.samples[label])

    def lines(self):
        while True:
            hum = float(np.clip(self.rng.normal(0.5, 0.08), 0.2, 0.8))
            resp = self.sensor.sniff(self.comp, humidity=hum, intensity=self.rng.uniform(0.7, 1.3), rng=self.rng)
            ratio = np.clip(1.0 - 0.8 * resp[self.CHANNELS], 0.05, 2.0)
            res = self.R0 * self.drift * ratio
            ms = int((time.time() - self.t0) * 1000)
            if not self.fast:
                time.sleep(1.0)
            yield "BME,%d,%.2f,%.2f,%s\n" % (ms, 24.0, hum * 100, ",".join(f"{r:.0f}" for r in res))


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def collect(source, seconds: float, steps: list[int], simulated: bool):
    """Collect parsed readings for `seconds` (or seconds-many cycles when simulated)."""
    rows = []
    start = time.time()
    for line in source.lines():
        parsed = parse_line(line, len(steps))
        if parsed:
            rows.append(parsed)
            if len(rows) % 10 == 0:
                print(f"\r  {len(rows)} readings", end="", flush=True)
        done = len(rows) >= seconds if simulated else time.time() - start >= seconds
        if done:
            break
    print()
    return rows


def record(source, label: str, session: str, minutes: float, baseline_s: float, steps: list[int],
           simulated: bool = False, wait_for_enter: bool = True) -> Path:
    print(f"1/2 Baseline: keep the sensor in CLEAN AIR for {baseline_s:.0f} s ...")
    source.set_smell("clean air")
    base = collect(source, baseline_s, steps, simulated)
    if not base:
        sys.exit("No readings came in. Is the ESP32 running the firmware and the port right?")
    r0 = np.median(np.array([r for _, _, r in base]), axis=0)
    print("  R0 =", ", ".join(f"{s}°C: {v / 1000:.1f} kΩ" for s, v in zip(steps, r0)))

    if wait_for_enter and not simulated:
        input(f"2/2 Put the sensor in the jar with '{label}' and press Enter ...")
    else:
        print(f"2/2 Recording '{label}' ...")
    source.set_smell(label)
    rows = collect(source, minutes * 60, steps, simulated)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in f"{session}_{label}")
    path = OUT_DIR / f"{safe}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "session", "temp_c", "humidity"]
                   + [f"ratio_{s}" for s in steps] + [f"raw_{s}" for s in steps])
        for temp, hum, res in rows:
            w.writerow([label, session, temp, hum]
                       + [f"{r / b:.5f}" for r, b in zip(res, r0)] + [f"{r:.0f}" for r in res])
    print(f"Saved {len(rows)} readings to {path.relative_to(ROOT)}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, e.g. COM3 or /dev/ttyUSB0")
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--label", help="what you're recording, e.g. coffee or 'clean air'")
    ap.add_argument("--session", default=time.strftime("day-%Y%m%d"), help="one name per day, e.g. day1")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--baseline", type=float, default=120, help="seconds of clean air first")
    ap.add_argument("--steps", default="200,250,300,350", help="heater steps, must match the firmware")
    ap.add_argument("--simulate", action="store_true", help="no hardware: fake readings from sensor.py")
    args = ap.parse_args()

    if args.list_ports:
        try:
            from serial.tools import list_ports
        except ImportError:
            sys.exit("pyserial is missing. Run:  pip install -r requirements-hardware.txt")
        for p in list_ports.comports():
            print(f"{p.device:15s} {p.description}")
        return
    if not args.label:
        ap.error("--label is required")
    steps = [int(s) for s in args.steps.split(",")]
    if args.simulate:
        source = SimulatedSource(args.session)
        if args.label != "clean air" and args.label not in source.samples:
            ap.error(f"--simulate knows: clean air, {', '.join(source.samples)}")
    else:
        if not args.port:
            ap.error("--port is required (or use --simulate)")
        source = SerialSource(args.port)
    record(source, args.label, args.session, args.minutes, args.baseline, steps, simulated=args.simulate)


if __name__ == "__main__":
    main()
