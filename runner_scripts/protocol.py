"""The benchmark protocol, loaded from ``runner_scripts/protocol.csv``.

That file is the single source of truth for every constant the comparison
shares across frameworks and languages: the timing repeat count, the ensemble
sizes, the performance-sweep solver settings, and the work-precision and
numerical-equivalence sweep grids. The Julia consumers load the same file
through ``runner_scripts/protocol.jl`` and the MPGOS binary reads it at
runtime, so a protocol change edits one file.

Values that only one suite uses (the overlap suite's controller pins, the ne
suite's dt clamps) stay in that suite's module; this file carries only what
must agree everywhere.
"""

import csv
import os

_PROTOCOL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "protocol.csv")


def _load():
    out = {}
    with open(_PROTOCOL_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["name"]] = [float(v) for v in row["values"].split()]
    return out


_P = _load()

# Timed repeats per point (after one untimed warm-up); min is reported.
REPEATS = int(_P["repeats"][0])

# Ensemble sizes: the work-precision / numerical-output size and the
# numerical-equivalence size.
N_WP = int(_P["n_wp"][0])
N_NE = int(_P["n_ne"][0])

# Performance-sweep solver settings shared by every framework.
PERF_FIXED_DT = _P["perf_fixed_dt"][0]
PERF_ADAPTIVE_TOL = _P["perf_adaptive_tol"][0]

# Sweep grids. wp: dyadic dt fractions of the t=1 duration and atol=rtol
# tolerances. ne: the wp dt grid extended to coarser steps (the convergence
# region of order >= 5 methods) and a tolerance grid cut at the float32 floor.
WP_DTS = _P["wp_dts"]
WP_TOLS = _P["wp_tols"]
NE_DTS = _P["ne_dts"]
NE_TOLS = _P["ne_tols"]
