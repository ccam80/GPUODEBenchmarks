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

# CLI analysis names; the CSVs record the underscored form.
ANALYSES = ("performance", "numerical", "work-precision")
PHASES = ("performance", "numerical", "work_precision")


def phases_for(analysis):
    return PHASES if analysis == "all" else (analysis.replace("-", "_"),)

# Protocol constants; mirrored in julia_worker.jl.
FIXED_DT = 2.0 ** -10
ADAPTIVE_TOL = 1.0e-8
PERFORMANCE_REPEATS = 20
WORK_REPEATS = 20
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


def _retire_stale_schema(path, fields):
    """Move a CSV written under a different schema aside, keeping the data.

    Timing rows changed from one-per-repeat to one-per-point. A file left
    over from the old layout cannot be appended to or pruned coherently, so
    it is renamed rather than silently mixed with new rows or deleted.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), None)
    if header is None or header == list(fields):
        return False
    target = path.with_name(path.stem + ".legacy" + path.suffix)
    index = 1
    while target.exists():
        target = path.with_name("{}.legacy{}{}".format(path.stem, index, path.suffix))
        index += 1
    path.rename(target)
    print("{} used the previous schema; moved to {} and starting fresh."
          .format(path.name, target.name))
    return True


def ensure_csv(path, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _retire_stale_schema(path, fields)
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


# OrdinaryDiffEq's qsteady deadband, resolved from its defaults and exported
# by the ne suite (data/numerical_equivalence/julia/controller_constants.csv):
# explicit RK holds dt over qsteady 1.0..1.0 (no deadband), every implicit
# family over 1.0..1.2. Julia's q divides dt where cubie's gain multiplies it,
# so the bounds inverate: deadband = (1/qsteady_max, 1/qsteady_min). The ne
# suite's "matched" tier derives the same numbers per algorithm; keep the two
# in sync so both suites' comparison tiers are the same experiment.
JULIA_QSTEADY_MAX = {"ERK": 1.0}
JULIA_QSTEADY_MAX_IMPLICIT = 1.2


def pi_controller(order, family):
    """Return the PI-controller configuration used by the comparison tier."""
    from cubie.integrators.algorithms.generic_dirk import (
        dirk_default_ki,
        dirk_default_kp,
    )
    qsteady_max = JULIA_QSTEADY_MAX.get(family, JULIA_QSTEADY_MAX_IMPLICIT)
    return {
        "step_controller": "pi",
        "kp": dirk_default_kp,
        "ki": dirk_default_ki,
        "safety": 0.9,
        "min_gain": 0.2,
        "max_gain": 10.0,
        "deadband_min": 1.0 / qsteady_max,
        "deadband_max": 1.0,
    }
