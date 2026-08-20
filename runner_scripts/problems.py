"""The problem axis: one row per benchmark ODE/DAE in problems.csv, read the same way by problems.jl."""

import csv
import os

PROBLEMS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "problems.csv")

DEFAULT_PROBLEM = "lorenz"

# The states sweep resizes this problem; see STATES_GRID in wp_common.py.
STATES_PROBLEM = "lorenz96"


def states_row(n):
    """The lorenz96 row resized to n states, with exclusions cleared."""
    row = get_problem(STATES_PROBLEM)
    return Problem({**row, "states": n, "exclusions": frozenset()})

_INT_FIELDS = ("states",)
_FLOAT_FIELDS = ("duration", "sweep_min", "sweep_max", "golden_tol")

# Dyadic dt-grid exponents as duration fractions; mirrored in problems.jl.
WP_K = (4, 13)
# Euler runs a finer grid than the higher-order methods.
EULER_K = (8, 17)
NE_K = (1, 13)
# Timed N-sweep fixed step: duration * 2^-10.
TIMING_DT_K = 10


class Problem(dict):
    """One row of problems.csv with its derived grids."""

    @property
    def name(self):
        return self["problem"]

    @property
    def duration(self):
        return self["duration"]

    @property
    def timing_dt(self):
        """Fixed step used by the N-sweep: duration * 2^-10."""
        return self["duration"] * 2.0 ** -TIMING_DT_K

    def dts(self, algorithm=None):
        """Fixed-step dt grid for the work-precision sweep."""
        lo, hi = EULER_K if algorithm == "euler" else WP_K
        return [self["duration"] * 2.0 ** -k for k in range(lo, hi + 1)]

    def ne_dts(self):
        """Fixed-step dt grid for the numerical-equivalence sweep."""
        return [self["duration"] * 2.0 ** -k
                for k in range(NE_K[0], NE_K[1] + 1)]

    def sweep(self, n, dtype=None):
        """The ensemble parameter grid: n values over the sweep range."""
        import numpy as np
        lo, hi = self["sweep_min"], self["sweep_max"]
        if self["sweep_scale"] == "log":
            if lo <= 0.0:
                raise SystemExit(
                    "problem '{0}': a log sweep needs sweep_min > 0"
                    .format(self.name))
            return np.logspace(np.log10(lo), np.log10(hi), n, dtype=dtype)
        return np.linspace(lo, hi, n, dtype=dtype)

    def supports(self, framework):
        return framework in self["frameworks"]

    def runs(self, framework, algorithm):
        """True unless the (framework, algorithm) pair is excluded here."""
        return (self.supports(framework)
                and (framework, algorithm) not in self["exclusions"])


def load_problems():
    """Every problem in declaration order."""
    with open(PROBLEMS_CSV, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    problems = []
    for row in rows:
        for field in _INT_FIELDS:
            row[field] = int(row[field])
        for field in _FLOAT_FIELDS:
            row[field] = float(row[field])
        row["frameworks"] = tuple(row["frameworks"].split("|"))
        # framework:algorithm pairs this problem never attempts.
        row["exclusions"] = frozenset(
            tuple(token.split(":", 1))
            for token in (row.get("exclusions") or "").split("|") if token)
        problems.append(Problem(row))
    return problems


def problem_names():
    return [row["problem"] for row in load_problems()]


def get_problem(name):
    """One problem by name; exits on an unknown name."""
    for row in load_problems():
        if row["problem"] == name:
            return row
    raise SystemExit("unknown problem '{0}' (expected one of: {1})".format(
        name, ", ".join(problem_names())))


def resolve_problems(request, framework=None):
    """Resolve "all" or a comma list to the problems a framework runs."""
    if request in (None, "", "all"):
        selected = load_problems()
    else:
        selected = [get_problem(name) for name in request.split(",") if name]
    if framework is not None:
        selected = [row for row in selected if row.supports(framework)]
    return selected


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--states-grid":
        from wp_common import STATES_GRID
        print(" ".join(str(n) for n in STATES_GRID))
    elif len(sys.argv) > 1:
        # <framework> [request]: the resolved problem names, one per line.
        request = sys.argv[2] if len(sys.argv) > 2 else "all"
        for row in resolve_problems(request, sys.argv[1]):
            print(row["problem"])
    else:
        for row in load_problems():
            print(row["problem"])
