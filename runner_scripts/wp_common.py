"""Shared constants and helpers for the work-precision (wp) sweeps.

wp mode runs the Lorenz ensemble at N = 32768, sweeping dt (fixed) and/or
rtol = atol (adaptive) per algorithm, timing each setting (untimed warmup,
``timeit.repeat``, min * 1000 ms) and scoring the ensemble l2 final-state
error against ``data/numerical/golden_lorenz_32768.csv``. Rows
``<setting> <time_ms> <error>`` are written (mode "w") to
``data/<FRAMEWORK>/<Prefix>_wp_<fixed|adaptive>_<algorithm>_<os>_<gpu>.txt``
with cubie-vocabulary algorithm names. The Julia and MPGOS writers mirror
these constants; keep them in sync.
"""

import os

import numpy as np

# Euler gets its own finer grid: a first-order method needs far smaller dt
# for errors in the same range as the order >= 4 methods.
DTS = [2.0 ** -k for k in range(4, 14)]        # 1/16 .. 1/8192, 10 points
DTS_EULER = [2.0 ** -k for k in range(8, 18)]  # 1/256 .. 1/131072, 10 points
TOLS = [10.0 ** -k for k in range(2, 9)]       # 1e-2 .. 1e-8, 7 points

N_WP = 32768

GOLDEN_PATH = os.path.join("data", "numerical", "golden_lorenz_32768.csv")


def dts_for(algorithm):
    """The fixed-step dt grid appropriate to the given algorithm."""
    return DTS_EULER if algorithm == "euler" else DTS


def load_golden():
    """Load the Float64 golden final states, shape (N_WP, 3)."""
    if not os.path.isfile(GOLDEN_PATH):
        raise FileNotFoundError(
            "{0} not found - generate it first with "
            "`julia -t auto --project=. runner_scripts/golden/generate_golden.jl`"
            .format(GOLDEN_PATH))
    golden = np.loadtxt(GOLDEN_PATH, delimiter=",")
    if golden.shape != (N_WP, 3):
        raise ValueError("golden reference has shape {0}, expected ({1}, 3)"
                         .format(golden.shape, N_WP))
    return golden


def ensemble_error(final_states, golden):
    """l2-at-final error over the ensemble, computed in float64."""
    diff = np.asarray(final_states, dtype=np.float64) - golden
    return float(np.sqrt(np.mean(diff ** 2)))


def wp_outfile(framework_dir, prefix, mode, algorithm, dataset_key):
    """Path of the keyed wp output file; ensures the directory exists."""
    d = os.path.join("data", framework_dir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "{0}_wp_{1}_{2}_{3}.txt".format(
        prefix, mode, algorithm, dataset_key))


def times_outfile(framework_dir, prefix, mode, algorithm, dataset_key):
    """Path of the keyed N-sweep timing file; ensures the directory exists."""
    d = os.path.join("data", framework_dir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "{0}_times_{1}_{2}_{3}.txt".format(
        prefix, mode, algorithm, dataset_key))


def parse_bench_args(argv, supported):
    """Parse ``<N> [wp] [algorithm|all]``; return (n, wp, requested subset of
    `supported`). Unknown names exit; a valid name this framework lacks
    yields an empty list."""
    if not argv:
        raise SystemExit("usage: <N> [wp] [algorithm|all]")
    n = int(argv[0])
    wp = False
    request = "all"
    for tok in argv[1:]:
        if tok == "wp":
            wp = True
        else:
            request = tok
    vocabulary = ("euler", "classical-rk4", "tsit5", "cash-karp-54")
    if request != "all" and request not in vocabulary:
        raise SystemExit(
            "unknown algorithm '{0}' (expected one of: all, {1})".format(
                request, ", ".join(vocabulary)))
    if request == "all":
        algorithms = list(supported)
    else:
        algorithms = [a for a in supported if a == request]
    return n, wp, algorithms
