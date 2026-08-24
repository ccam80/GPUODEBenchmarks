"""Shared protocol and append-only result helpers for the direct GPU suite."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS_CSV = Path(__file__).with_name("algorithms.csv")
# Numerical grids and adaptive pins are shared with the NE suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "numerical_equivalence"))
from ne_common import (  # noqa: E402 - path bootstrap above
    TOLS_NE as NE_TOLS, N_NE, DT0_FRACTION as DT0,
    DT_MIN_FRACTION as DT_MIN, DT_MAX_FRACTION as DT_MAX, controllers_equal,
    cubie_default_controller, read_ne_csv, read_ne_adaptive_csv,
)
from problems import NE_K  # noqa: E402 - path bootstrap above

# The ne dt grid as duration fractions; the workers scale by the duration.
NE_DTS = [2.0 ** -k for k in range(NE_K[0], NE_K[1] + 1)]

CUBIE_NE_DATA = REPO_ROOT / "data" / "numerical_equivalence" / "cubie"


def golden_ne(problem):
    """Path of the ne golden reference for a problem row or name."""
    name = problem["problem"] if isinstance(problem, dict) else problem
    return REPO_ROOT / "data" / "numerical" / "golden_ne_{0}_{1}.csv".format(
        name, N_NE)


def golden_wp(problem):
    """Path of the wp golden reference for a problem row or name."""
    name = problem["problem"] if isinstance(problem, dict) else problem
    return REPO_ROOT / "data" / "numerical" / "golden_{0}_{1}.csv".format(
        name, N_WP)

# CLI analysis names; the CSVs record the underscored form.
ANALYSES = ("performance", "numerical", "work-precision")
PHASES = ("performance", "numerical", "work_precision")


def phases_for(analysis):
    return PHASES if analysis == "all" else (analysis.replace("-", "_"),)

# Protocol constants; mirrored in julia_worker.jl. dt values are fractions of
# the problem duration.
FIXED_DT = 2.0 ** -10
ADAPTIVE_TOL = 1.0e-8
# Repeat ceilings; the count per leg follows its first timed run's duration.
PERFORMANCE_REPEATS = 20
WORK_REPEATS = 20
WP_DTS = [2.0 ** -k for k in range(4, 14)]
WP_TOLS = [10.0 ** -k for k in range(2, 9)]
# Mirrors runner_scripts/wp_common.py.
N_WP = 131072

# Overlap family labels -> ne_common family keys.
NE_FAMILY = {"ERK": "erk", "ESDIRK": "dirk", "Rosenbrock-W": "rosenbrock"}

# "transfers": "both" includes h2d and d2h, "none" includes neither.
# One row per timed point: the workers reduce their repeats before writing, so
# the headline statistic (min, as in the performance suite) is fixed at the
# point of measurement rather than recomputed downstream.
TIMING_STATS = ["samples", "min_ms", "p05_ms", "median_ms", "p95_ms", "max_ms"]
TIMING_FIELDS = ["framework", "algorithm", "phase", "mode", "tier", "transfers",
                 "n", "setting_kind", "setting"] + TIMING_STATS
METRIC_FIELDS = ["framework", "algorithm", "phase", "mode", "tier", "n",
                 "setting_kind", "setting", "golden_rmse", "finite_trajectories",
                 "failed_trajectories", "finals_path"]
FAILURE_FIELDS = ["framework", "algorithm", "phase", "mode", "tier", "n",
                  "setting_kind", "setting", "error_type", "message"]


def algorithms(name="all"):
    with ALGORITHMS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["order"] = int(row["order"])
    if name != "all":
        rows = [row for row in rows if row["cubie_alias"] == name]
        if not rows:
            raise SystemExit("unknown algorithm '{}'; see algorithms.csv".format(name))
    return rows


def algorithm_names():
    return ["all"] + [row["cubie_alias"] for row in algorithms()]


def performance_ns(nmax, from_n=0):
    values, n = [], 8
    while n <= nmax:
        if n >= from_n:
            values.append(n)
        n *= 4
    return values


def parse_ns(spec, from_n=0):
    """A single value is a sweep ceiling; a comma list is the exact counts."""
    text = str(spec)
    if "," not in text:
        return performance_ns(int(text), from_n)
    values = sorted({int(part) for part in text.split(",") if part})
    return [n for n in values if n >= max(from_n, 8)]


def protocol(nmax, from_n=0):
    return {
        "performance_ns": parse_ns(nmax, from_n),
        "performance_repeats": PERFORMANCE_REPEATS,
        "ne_n": N_NE,
        "ne_dts": NE_DTS,
        "ne_tols": NE_TOLS,
        "wp_n": N_WP,
        "wp_dts": WP_DTS,
        "wp_tols": WP_TOLS,
        "work_repeats": WORK_REPEATS,
    }


def timing_stats(values):
    """Reduce one point's repeats to the persisted timing statistics.

    Percentiles use linear interpolation so the Julia worker's
    ``Statistics.quantile`` defaults produce identical numbers.
    """
    import numpy as np
    a = np.asarray(list(values), dtype=float)
    return {"samples": len(a), "min_ms": float(np.min(a)),
            "p05_ms": float(np.percentile(a, 5)),
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "max_ms": float(np.max(a))}


def scaled_dts(problem):
    """Fixed step and the adaptive dt pins for a problem, in problem time."""
    duration = problem["duration"]
    return (duration * FIXED_DT, duration * DT0, duration * DT_MIN,
            duration * DT_MAX)


def ensure_csv(path, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
    return path


def regenerated(row, phases, from_n=0, algorithm="all", ns=None):
    """True when a run over `phases` will produce this row again."""
    if algorithm != "all" and row.get("algorithm") != algorithm:
        return False
    if row.get("phase") not in phases:
        return False
    if row.get("phase") != "performance" or (not from_n and ns is None):
        return True
    try:
        n = int(row["n"])
    except (KeyError, TypeError, ValueError):
        return True
    return n in ns if ns is not None else n >= from_n


def prune_csv(path, fields, phases, from_n=0, algorithm="all", ns=None):
    """Drop the rows a run regenerates; from_n, algorithm and ns narrow which."""
    path = ensure_csv(path, fields)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    kept = [row for row in rows if not regenerated(row, phases, from_n, algorithm, ns)]
    if len(kept) == len(rows):
        return 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    return len(rows) - len(kept)


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


def pi_controller(order, family):
    """Return the PI-controller configuration used by the comparison tier."""
    from cubie.integrators.algorithms.generic_dirk import (
        dirk_default_ki,
        dirk_default_kp,
    )
    return {
        "step_controller": "pi",
        "kp": dirk_default_kp,
        "ki": dirk_default_ki,
        "safety": 0.9,
        "min_gain": 0.2,
        "max_gain": 10.0,
    }
