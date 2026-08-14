"""The problem axis: one row per benchmark ODE/DAE in problems.csv, read the same way by problems.jl."""

import csv
import os

PROBLEMS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "problems.csv")

DEFAULT_PROBLEM = "lorenz"

_INT_FIELDS = ("states", "dae_index", "wp_k_min", "wp_k_max",
               "euler_k_min", "euler_k_max", "ne_k_min", "ne_k_max")
_FLOAT_FIELDS = ("duration", "sweep_min", "sweep_max")


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
        """Fixed step used by the N-sweep: 1000 steps for every problem."""
        return self["duration"] / 1000.0

    def dts(self, algorithm=None):
        """Fixed-step dt grid for the work-precision sweep."""
        lo, hi = (self["euler_k_min"], self["euler_k_max"]) \
            if algorithm == "euler" else (self["wp_k_min"], self["wp_k_max"])
        return [self["duration"] * 2.0 ** -k for k in range(lo, hi + 1)]

    def ne_dts(self):
        """Fixed-step dt grid for the numerical-equivalence sweep."""
        return [self["duration"] * 2.0 ** -k
                for k in range(self["ne_k_min"], self["ne_k_max"] + 1)]

    def sweep(self, n, dtype=None):
        """The ensemble parameter grid: n values over the sweep range."""
        import numpy as np
        return np.linspace(self["sweep_min"], self["sweep_max"], n,
                           dtype=dtype)

    def supports(self, framework):
        return framework in self["frameworks"]

    @property
    def is_dae(self):
        return self["dae_index"] > 0


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
    for row in load_problems():
        print(row["problem"])
