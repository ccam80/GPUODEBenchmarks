"""Shared benchmark protocol constants, loaded from protocol.csv."""

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

# Ensemble sizes for the work-precision and numerical-equivalence sweeps.
N_WP = int(_P["n_wp"][0])
N_NE = int(_P["n_ne"][0])

# Performance-sweep solver settings shared by every framework.
PERF_FIXED_DT = _P["perf_fixed_dt"][0]
PERF_ADAPTIVE_TOL = _P["perf_adaptive_tol"][0]

# Work-precision and numerical-equivalence sweep grids.
WP_DTS = _P["wp_dts"]
WP_TOLS = _P["wp_tols"]
NE_DTS = _P["ne_dts"]
NE_TOLS = _P["ne_tols"]
