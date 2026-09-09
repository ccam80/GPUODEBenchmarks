"""The ensemble grid: v[i] in float64, linear or log, last point pinned to grid_max, cast to grid_dtype; grid.jl and GPU_ODE_MPGOS/grid.cuh match it bit for bit. `python grid.py write` refreshes tests/grids/<problem>_131072.npy."""

import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDS_DIR = os.path.join(HERE, "tests", "grids")
PROBLEMS_CSV = os.path.join(HERE, "problems.csv")

SCALES = ("linear", "log")
GRID_DTYPES = {"float32": np.float32}
PRECISIONS = {"float32": np.float32, "float64": np.float64}
REFERENCE_N = 131072


def _check(scale, grid_min, grid_max, n):
    if scale not in SCALES:
        raise ValueError("grid_scale '{0}' is not linear or log".format(scale))
    n = int(n)
    if n < 2:
        raise ValueError("a grid needs n >= 2, got {0}".format(n))
    lo, hi = float(grid_min), float(grid_max)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("grid_min and grid_max must be finite")
    if scale == "log" and (lo <= 0.0 or hi <= 0.0):
        raise ValueError("a log grid needs grid_min > 0 and grid_max > 0")
    return lo, hi, n


def grid_values(scale, grid_min, grid_max, n, grid_dtype="float32"):
    """v[0..n-1] as a grid_dtype array."""
    if grid_dtype not in GRID_DTYPES:
        raise ValueError("grid_dtype '{0}' is not float32".format(grid_dtype))
    lo, hi, n = _check(scale, grid_min, grid_max, n)
    if scale == "linear":
        step = (hi - lo) / (n - 1)
        # One multiply and one add per point, as every language does it.
        values = lo + np.arange(n, dtype=np.float64) * step
    else:
        a, b = math.log10(lo), math.log10(hi)
        step = (b - a) / (n - 1)
        # Scalar pow: the C runtime, as in grid.jl and grid.cuh.
        values = np.fromiter((10.0 ** (a + i * step) for i in range(n)),
                             dtype=np.float64, count=n)
    values[n - 1] = hi
    return values.astype(GRID_DTYPES[grid_dtype])


def grid_point(scale, grid_min, grid_max, n, index):
    """v[index] in float64 before the cast; the grid_max of a shorter grid that reproduces v[0..index]."""
    lo, hi, n = _check(scale, grid_min, grid_max, n)
    index = int(index)
    if not 0 <= index < n:
        raise ValueError("index {0} is outside 0..{1}".format(index, n - 1))
    if index == n - 1:
        return hi
    if scale == "linear":
        return lo + index * ((hi - lo) / (n - 1))
    a, b = math.log10(lo), math.log10(hi)
    return 10.0 ** (a + index * ((b - a) / (n - 1)))


def grid(spec):
    """The grid of a run spec in its precision: the grid_dtype values, widened back for float64 runs."""
    precision = spec.get("precision", "float32")
    if precision not in PRECISIONS:
        raise ValueError("precision '{0}' is not float32 or float64".format(precision))
    values = grid_values(spec["grid_scale"], spec["grid_min"], spec["grid_max"],
                         spec["n"], spec.get("grid_dtype", "float32"))
    return values.astype(PRECISIONS[precision])


def grid_equal(a, b):
    """The two grid specs produce the same float32 values over the same n."""
    if int(a["n"]) != int(b["n"]):
        return False
    return bool(np.array_equal(grid_values(a["grid_scale"], a["grid_min"], a["grid_max"], a["n"], a.get("grid_dtype", "float32")),
                               grid_values(b["grid_scale"], b["grid_min"], b["grid_max"], b["n"], b.get("grid_dtype", "float32"))))


def grid_contains(container, spec):
    """The container's grid equals the spec's, or shares scale, min and dtype with a larger n and v[n-1] float32-equal to the spec's grid_max."""
    if container.get("grid_dtype", "float32") != spec.get("grid_dtype", "float32"):
        return False
    if container["grid_scale"] != spec["grid_scale"]:
        return False
    if float(container["grid_min"]) != float(spec["grid_min"]):
        return False
    n_spec, n_container = int(spec["n"]), int(container["n"])
    if n_container < n_spec:
        return False
    if n_container == n_spec:
        return float(container["grid_max"]) == float(spec["grid_max"])
    values = grid_values(container["grid_scale"], container["grid_min"],
                         container["grid_max"], n_container,
                         container.get("grid_dtype", "float32"))
    return values[n_spec - 1] == np.float32(spec["grid_max"])


def problem_grids(path=PROBLEMS_CSV):
    """(problem, scale, min, max) of every problems.csv row."""
    with open(path, newline="", encoding="utf-8") as handle:
        return [(row["problem"], row["sweep_scale"], float(row["sweep_min"]),
                 float(row["sweep_max"])) for row in csv.DictReader(handle)]


def reference_path(problem, n=REFERENCE_N, grids_dir=GRIDS_DIR):
    return os.path.join(grids_dir, "{0}_{1}.npy".format(problem, n))


def write_reference_grids(grids_dir=GRIDS_DIR, n=REFERENCE_N):
    """Write tests/grids/<problem>_<n>.npy for every problem; returns the paths."""
    os.makedirs(grids_dir, exist_ok=True)
    paths = []
    for problem, scale, lo, hi in problem_grids():
        path = reference_path(problem, n, grids_dir)
        np.save(path, grid_values(scale, lo, hi, n))
        paths.append(path)
    return paths


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        for written in write_reference_grids():
            print(written)
    else:
        sys.exit("usage: grid.py write")
