"""Shared protocol and append-only result helpers for the direct GPU suite."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS_CSV = Path(__file__).with_name("algorithms.csv")
GOLDEN_NE = REPO_ROOT / "data" / "numerical" / "golden_ne_lorenz_1024.csv"
GOLDEN_WP = REPO_ROOT / "data" / "numerical" / "golden_lorenz_32768.csv"

FIXED_DT = 2.0 ** -10
ADAPTIVE_TOL = 1.0e-8
NE_DTS = [2.0 ** -k for k in range(1, 14)]
NE_TOLS = [10.0 ** -k for k in range(2, 7)]
WP_DTS = [2.0 ** -k for k in range(4, 14)]
WP_TOLS = [10.0 ** -k for k in range(2, 9)]
N_NE = 1024
N_WP = 32768
DT0 = 0.01
DT_MIN = 1.0e-12
DT_MAX = 1.0e3

# "transfers": "both" includes h2d and d2h, "none" includes neither.
TIMING_FIELDS = ["framework", "algorithm", "phase", "mode", "tier", "transfers",
                 "n", "setting_kind", "setting", "sample", "time_ms"]
METRIC_FIELDS = ["framework", "algorithm", "phase", "mode", "tier", "n",
                 "setting_kind", "setting", "golden_rmse", "finite_trajectories",
                 "failed_trajectories", "finals_path"]
FAILURE_FIELDS = ["framework", "algorithm", "phase", "mode", "tier", "n",
                  "setting_kind", "setting", "error_type", "message"]


def algorithms():
    with ALGORITHMS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["order"] = int(row["order"])
    return rows


def performance_ns(nmax):
    values, n = [], 8
    while n <= nmax:
        values.append(n)
        n *= 4
    return values


def profile_protocol(profile, nmax, performance_repeats, work_repeats):
    if profile == "smoke":
        return {
            "performance_ns": performance_ns(min(nmax, 32)),
            "performance_repeats": min(performance_repeats, 2),
            "ne_n": 32,
            "ne_dts": [2.0 ** -4, 2.0 ** -8],
            "ne_tols": [1.0e-3],
            "wp_n": 256,
            "wp_dts": [2.0 ** -6],
            "wp_tols": [1.0e-4],
            "work_repeats": min(work_repeats, 2),
        }
    return {
        "performance_ns": performance_ns(nmax),
        "performance_repeats": performance_repeats,
        "ne_n": N_NE,
        "ne_dts": NE_DTS,
        "ne_tols": NE_TOLS,
        "wp_n": N_WP,
        "wp_dts": WP_DTS,
        "wp_tols": WP_TOLS,
        "work_repeats": work_repeats,
    }


def ensure_csv(path, fields, reset=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset or not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
    return path


def append_csv(path, fields, row):
    path = ensure_csv(path, fields)
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore").writerow(row)


def point_slug(value):
    return ("{:.10g}".format(float(value)).replace("-", "m")
            .replace("+", "p").replace(".", "p"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def finite_counts(finals):
    import numpy as np
    per_trajectory = np.all(np.isfinite(finals), axis=1)
    return int(np.sum(per_trajectory)), int(len(per_trajectory) - np.sum(per_trajectory))


def rmse(finals, golden):
    import numpy as np
    mask = np.all(np.isfinite(finals), axis=1)
    if not np.any(mask):
        return math.nan
    delta = np.asarray(finals[mask], dtype=np.float64) - np.asarray(golden[mask], dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def pi_controller(order):
    """Return the PI-controller configuration used by the comparison tier."""
    beta1 = 7.0 / (10.0 * order)
    beta2 = 2.0 / (5.0 * order)
    return {
        "step_controller": "pi",
        "kp": beta1 * (order + 1),
        "ki": -beta2 * (order + 1),
        "safety": 0.9,
        "min_gain": 0.2,
        "max_gain": 10.0,
    }
