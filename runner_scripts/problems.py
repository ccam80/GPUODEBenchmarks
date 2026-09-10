"""The problem catalogue: one row per benchmark ODE/DAE in problems.csv with its default state count, duration, sweep range, golden algorithm and the packages that implement it."""

import csv
import os

PROBLEMS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "problems.csv")

DEFAULT_PROBLEM = "lorenz"

# The one problem whose state count is a construction parameter.
RESIZABLE_PROBLEM = "lorenz96"

_INT_FIELDS = ("states",)
_FLOAT_FIELDS = ("duration", "sweep_min", "sweep_max", "golden_tol")


class Problem(dict):
    """One row of problems.csv."""

    @property
    def name(self):
        return self["problem"]

    @property
    def duration(self):
        return self["duration"]

    def supports(self, package):
        return package in self["frameworks"]

    def system_params(self, states=None):
        """The canonical construction parameters: {"states": n} for the resizable problem, {} otherwise."""
        if self.name == RESIZABLE_PROBLEM:
            return {"states": int(self["states"] if states is None else states)}
        if states is not None and int(states) != self["states"]:
            raise ValueError("{0} has {1} states; it cannot be resized".format(
                self.name, self["states"]))
        return {}

    def resized(self, states):
        """A copy of the row with another state count."""
        return Problem({**self, "states": int(states)})


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


def as_problem(problem):
    """The problem row for a row or a name."""
    return problem if isinstance(problem, dict) else get_problem(problem)


def resolve_problems(request, package=None):
    """Resolve "all" or a comma list to the problems a package implements."""
    if request in (None, "", "all"):
        selected = load_problems()
    else:
        selected = [get_problem(name) for name in request.split(",") if name]
    if package is not None:
        selected = [row for row in selected if row.supports(package)]
    return selected


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # <package> [request]: the resolved problem names, one per line.
        request = sys.argv[2] if len(sys.argv) > 2 else "all"
        for row in resolve_problems(request, sys.argv[1]):
            print(row["problem"])
    else:
        for row in load_problems():
            print(row["problem"])
