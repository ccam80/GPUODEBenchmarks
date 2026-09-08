"""The algorithm axis: one row per integration algorithm in algorithms.csv, read the same way by algorithms.jl."""

import csv
import os

ALGORITHMS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "algorithms.csv")

_MODES = ("fixed", "adaptive")
_BOOLS = ("ne", "ne_adaptive")
FAMILIES = ("erk", "dirk", "firk", "rosenbrock", "implicit")


class Algorithm(dict):
    """One row of algorithms.csv: `fixed` and `adaptive` are framework tuples, `ne` and `ne_adaptive` bools, `julia_cpu` and `julia_gpu` constructor expressions or empty."""

    @property
    def name(self):
        return self["algorithm"]

    def supports(self, framework, mode=None):
        """True when the framework times this algorithm, in the mode if given."""
        modes = _MODES if mode is None else (mode,)
        return any(framework in self[m] for m in modes)

    @property
    def in_ne(self):
        return self["ne"]

    @property
    def in_overlap(self):
        return bool(self["julia_gpu"])

    @property
    def runs_fixed_ne(self):
        """The fixed-step ne sweep excludes the erk family."""
        return self["family"] != "erk"


def load_algorithms():
    """Every algorithm in declaration order."""
    with open(ALGORITHMS_CSV, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    algorithms = []
    for row in rows:
        for mode in _MODES:
            row[mode] = tuple(f for f in row[mode].split("|") if f)
        for flag in _BOOLS:
            row[flag] = row[flag].strip().lower() == "true"
        row["order"] = int(row["order"])
        algorithms.append(Algorithm(row))
    return algorithms


def algorithm_names():
    return [row["algorithm"] for row in load_algorithms()]


def get_algorithm(name):
    """One algorithm by name; exits on an unknown name."""
    for row in load_algorithms():
        if row["algorithm"] == name:
            return row
    raise SystemExit("unknown algorithm '{0}' (expected one of: all, {1})"
                     .format(name, ", ".join(algorithm_names())))


def supported_for(framework, mode=None):
    """Algorithm names a framework times, in declaration order."""
    return tuple(row["algorithm"] for row in load_algorithms()
                 if row.supports(framework, mode))


def resolve_algorithms(request, framework):
    """Resolve "all" or a comma list to the algorithms a framework times."""
    supported = supported_for(framework)
    if request in (None, "", "all"):
        return list(supported)
    names = [name for name in request.split(",") if name]
    for name in names:
        get_algorithm(name)
    return [name for name in names if name in supported]


def _select(rows, request):
    """Rows named by "all" or a comma list; an unknown name exits."""
    if request in (None, "", "all"):
        return rows
    names = [name for name in request.split(",") if name]
    for name in names:
        get_algorithm(name)
    return [row for row in rows if row["algorithm"] in names]


def ne_algorithms(request="all"):
    """The numerical-equivalence rows, narrowed by name."""
    return _select([row for row in load_algorithms() if row.in_ne], request)


def overlap_algorithms(request="all"):
    """The cubie-DiffEqGPU overlap rows, narrowed by name."""
    return _select([row for row in load_algorithms() if row.in_overlap],
                   request)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # <framework> [request]: the resolved algorithm names, one per line.
        request = sys.argv[2] if len(sys.argv) > 2 else "all"
        for name in resolve_algorithms(request, sys.argv[1]):
            print(name)
    else:
        for row in load_algorithms():
            print(row["algorithm"])
