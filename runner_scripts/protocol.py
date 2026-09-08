"""Benchmark protocol constants, loaded from protocol.toml; `python protocol.py get <table.key>` prints one value, `--cxx-header <path>` writes the C++ header."""

import os
import sys
import tomllib

PROTOCOL_TOML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "protocol.toml")

with open(PROTOCOL_TOML, "rb") as _handle:
    PROTOCOL = tomllib.load(_handle)

_ensemble = PROTOCOL["ensemble"]
_fixed = PROTOCOL["fixed"]
_adaptive = PROTOCOL["adaptive"]
_repeats = PROTOCOL["repeats"]
_watchdog = PROTOCOL["watchdog"]

N_WP = int(_ensemble["n_wp"])
N_NE = int(_ensemble["n_ne"])
STATES_N = int(_ensemble["n_states"])
N_MIN = int(_ensemble["n_min"])
N_STEP = int(_ensemble["n_step"])
NMAX_DEFAULT = int(_ensemble["nmax_default"])

# BENCH_STATES_GRID (comma list) overrides the states sweep sizes.
_STATES_ENV = os.environ.get("BENCH_STATES_GRID", "")
STATES_GRID = (tuple(sorted(int(tok) for tok in _STATES_ENV.split(",")))
               if _STATES_ENV else tuple(int(s) for s in _ensemble["states_grid"]))

TIMING_DT_K = int(_fixed["timing_k"])
WP_K = tuple(int(k) for k in _fixed["wp_k"])
EULER_K = tuple(int(k) for k in _fixed["euler_k"])
NE_K = tuple(int(k) for k in _fixed["ne_k"])

TIMING_TOL = float(_adaptive["timing_tol"])
OVERLAP_TOL = float(_adaptive["overlap_tol"])
TOL_K = tuple(int(k) for k in _adaptive["tol_k"])
TOLS = [10.0 ** -k for k in range(TOL_K[0], TOL_K[1] + 1)]
DT_MIN_FRACTION = float(_adaptive["dt_min_fraction"])

OPTIMIZE_N = int(PROTOCOL["optimize"]["n"])
OPTIMIZE_PER_POINT_FAMILIES = tuple(PROTOCOL["optimize"]["per_point_families"])

REPEAT_CAP = int(_repeats["cap"])
REPEAT_SCHEDULE = tuple((float(limit), int(floor), int(ceiling))
                        for limit, floor, ceiling in _repeats["schedule"])
REPEAT_SPREAD = float(_repeats["spread"])

# BENCH_WATCHDOG_SECONDS overrides the per-run wall-clock ceiling.
WATCHDOG_SECONDS = float(os.environ.get("BENCH_WATCHDOG_SECONDS",
                                        _watchdog["seconds"]))
WATCHDOG_EXIT_CODE = int(_watchdog["exit_code"])

MAX_ERRORED_PCT = float(PROTOCOL["plots"]["max_errored_pct"])


def fixed_dts(duration, k_range):
    """dt grid duration * 2^-k over the inclusive exponent range."""
    return [duration * 2.0 ** -k for k in range(k_range[0], k_range[1] + 1)]


def performance_ns(nmax=NMAX_DEFAULT, from_n=0):
    """N sweep n_min * n_step^k up to nmax, from from_n up."""
    values, n = [], N_MIN
    while n <= nmax:
        if n >= from_n:
            values.append(n)
        n *= N_STEP
    return values


def parse_ns(spec, from_n=0):
    """A single value is a sweep ceiling; a comma list is the exact counts."""
    text = str(spec)
    if "," not in text:
        return performance_ns(int(text), from_n)
    values = sorted({int(part) for part in text.split(",") if part})
    return [n for n in values if n >= max(from_n, N_MIN)]


def get(path):
    """One value by dotted table.key path."""
    node = PROTOCOL
    for part in path.split("."):
        node = node[part]
    return node


def _cxx_literal(value):
    if isinstance(value, float):
        return "INFINITY" if value == float("inf") else repr(value)
    return str(value)


def cxx_header():
    """The generated C++ header text."""
    lines = ["// Generated from runner_scripts/protocol.toml by protocol.py; do not edit.",
             "#pragma once", "#include <cmath>",
             "#define PROTOCOL_N_WP {0}".format(N_WP),
             "#define PROTOCOL_N_STATES {0}".format(STATES_N),
             "#define PROTOCOL_TIMING_K {0}".format(TIMING_DT_K),
             "#define PROTOCOL_WP_K_LO {0}".format(WP_K[0]),
             "#define PROTOCOL_WP_K_HI {0}".format(WP_K[1]),
             "#define PROTOCOL_TOL_K_LO {0}".format(TOL_K[0]),
             "#define PROTOCOL_TOL_K_HI {0}".format(TOL_K[1]),
             "#define PROTOCOL_TIMING_TOL {0}".format(_cxx_literal(TIMING_TOL)),
             "#define PROTOCOL_REPEAT_CAP {0}".format(REPEAT_CAP),
             "#define PROTOCOL_REPEAT_SPREAD {0}".format(_cxx_literal(REPEAT_SPREAD)),
             "#define PROTOCOL_WATCHDOG_SECONDS {0}".format(
                 _cxx_literal(float(_watchdog["seconds"]))),
             "#define PROTOCOL_WATCHDOG_EXIT_CODE {0}".format(WATCHDOG_EXIT_CODE),
             "#define PROTOCOL_REPEAT_SCHEDULE_ROWS {0}".format(len(REPEAT_SCHEDULE)),
             "static const double PROTOCOL_REPEAT_SCHEDULE[][3] = {"]
    lines += ["    {{{0}, {1}, {2}}},".format(_cxx_literal(limit), floor, ceiling)
              for limit, floor, ceiling in REPEAT_SCHEDULE]
    lines.append("};")
    return "\n".join(lines) + "\n"


def write_cxx_header(path):
    """Write the header only when its text changes, so build caches stay valid."""
    text = cxx_header()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            if handle.read() == text:
                return
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main(argv):
    if len(argv) == 2 and argv[0] == "get":
        value = get(argv[1])
        print(" ".join(str(v) for v in value) if isinstance(value, list)
              else value)
        return 0
    if len(argv) == 2 and argv[0] == "--cxx-header":
        write_cxx_header(argv[1])
        return 0
    print("usage: protocol.py get <table.key> | --cxx-header <path>")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
