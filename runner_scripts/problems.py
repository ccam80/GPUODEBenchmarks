"""The problem axis: one row per benchmark ODE/DAE in problems.csv, read the same way by problems.jl."""

import csv
import os

from protocol import EULER_K, NE_K, TIMING_DT_K, WP_K, fixed_dts  # noqa: F401

PROBLEMS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "problems.csv")

DEFAULT_PROBLEM = "lorenz"

# The states sweep resizes this problem over protocol.STATES_GRID.
STATES_PROBLEM = "lorenz96"


def states_row(n):
    """The lorenz96 row resized to n states."""
    row = get_problem(STATES_PROBLEM)
    return Problem({**row, "states": n})

_INT_FIELDS = ("states",)
_FLOAT_FIELDS = ("duration", "sweep_min", "sweep_max", "golden_tol")


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
        """Fixed step used by the N-sweep: duration * 2^-timing_k."""
        return self["duration"] * 2.0 ** -TIMING_DT_K

    def dts(self, algorithm=None):
        """Fixed-step dt grid for the work-precision sweep."""
        return fixed_dts(self["duration"],
                         EULER_K if algorithm == "euler" else WP_K)

    def ne_dts(self):
        """Fixed-step dt grid for the numerical-equivalence sweep."""
        return fixed_dts(self["duration"], NE_K)

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
        from protocol import STATES_GRID
        print(" ".join(str(n) for n in STATES_GRID))
    elif len(sys.argv) > 1:
        # <framework> [request]: the resolved problem names, one per line.
        request = sys.argv[2] if len(sys.argv) > 2 else "all"
        for row in resolve_problems(request, sys.argv[1]):
            print(row["problem"])
    else:
        for row in load_problems():
            print(row["problem"])
