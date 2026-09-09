"""The algorithm catalogue: one row per integration algorithm in algorithms.csv with the packages that run it at a fixed step and adaptively."""

import csv
import os

ALGORITHMS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "algorithms.csv")

# The two stepping kinds, each a capability column.
KINDS = ("fixed", "adaptive")
FAMILIES = ("erk", "dirk", "firk", "rosenbrock", "implicit")


class Algorithm(dict):
    """One row of algorithms.csv: `fixed` and `adaptive` are package tuples, `order` an int."""

    @property
    def name(self):
        return self["algorithm"]

    @property
    def implicit(self):
        """True for every family that solves stage equations."""
        return self["family"] != "erk"

    def supports(self, package, kind=None):
        """True when the package runs this algorithm, in the stepping kind if given."""
        kinds = KINDS if kind is None else (kind,)
        return any(package in self[k] for k in kinds)


def load_algorithms():
    """Every algorithm in declaration order."""
    with open(ALGORITHMS_CSV, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    algorithms = []
    for row in rows:
        for kind in KINDS:
            row[kind] = tuple(f for f in row[kind].split("|") if f)
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


def supported_for(package, kind=None):
    """Algorithm names a package runs, in declaration order."""
    return tuple(row["algorithm"] for row in load_algorithms()
                 if row.supports(package, kind))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # <package> [fixed|adaptive]: the algorithms it runs, one per line.
        for name in supported_for(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None):
            print(name)
    else:
        for row in load_algorithms():
            print(row["algorithm"])
