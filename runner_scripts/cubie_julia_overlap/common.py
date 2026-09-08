"""Shared protocol and append-only result helpers for the direct GPU suite."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Numerical grids and adaptive pins are shared with the NE suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "numerical_equivalence"))
from algorithms import overlap_algorithms  # noqa: E402 - path bootstrap above
from ne_common import (  # noqa: E402, F401 - path bootstrap above
    cubie_ne_adaptive_file, cubie_ne_file, load_golden_ne, ne_sweep,
    read_ne_csv, read_ne_adaptive_csv,
)
from protocol import (  # noqa: E402 - path bootstrap above
    N_NE, N_WP, NE_K, OVERLAP_TOL, REPEAT_CAP, TIMING_DT_K, TOLS, WP_K,
    fixed_dts, parse_ns, performance_ns,
)
from wp_common import golden_path as golden_wp  # noqa: E402, F401 - path bootstrap above

# dt grids as duration fractions; the workers scale by the duration.
NE_DTS = fixed_dts(1.0, NE_K)

def golden_ne_states(problem):
    """The (N_NE, states) Float64 golden of the ne ensemble."""
    return load_golden_ne(problem)[1]

# CLI analysis names; the CSVs record the underscored form.
ANALYSES = ("performance", "numerical", "work-precision")
PHASES = ("performance", "numerical", "work_precision")


def phases_for(analysis):
    return PHASES if analysis == "all" else (analysis.replace("-", "_"),)

FIXED_DT = 2.0 ** -TIMING_DT_K
ADAPTIVE_TOL = OVERLAP_TOL
PERFORMANCE_REPEATS = REPEAT_CAP
WORK_REPEATS = REPEAT_CAP
WP_DTS = fixed_dts(1.0, WP_K)
WP_TOLS = TOLS
NE_TOLS = TOLS

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
    """The overlap rows of runner_scripts/algorithms.csv, narrowed by name."""
    rows = overlap_algorithms(name)
    if not rows:
        raise SystemExit("'{}' is not in the overlap set; see "
                         "runner_scripts/algorithms.csv".format(name))
    return rows


def algorithm_names():
    return ["all"] + [row["algorithm"] for row in algorithms()]


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


def ensure_csv(path, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
    return path


def regenerated(row, phases, from_n=0, algorithm="all", ns=None):
    """True when a run over `phases` will produce this row again."""
    if algorithm != "all" and row.get("algorithm") not in algorithm.split(","):
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
