"""The algorithm catalogue: one row per (package, algorithm) in algorithms.csv saying whether the package runs the algorithm at a fixed step, under an adaptive controller, and with a Newton tolerance."""

import csv
import os

from store import PACKAGES

ALGORITHMS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "algorithms.csv")

COLUMNS = ("algorithm", "package", "display", "family", "order", "fixed", "adaptive", "newton")
# The two stepping kinds, each a capability column.
KINDS = ("fixed", "adaptive")
CAPABILITIES = KINDS + ("newton",)
FAMILIES = ("erk", "dirk", "firk", "rosenbrock", "implicit")
# The columns every row of one algorithm must agree on.
FACTS = ("display", "family", "order")


class CatalogueError(ValueError):
    """algorithms.csv does not follow its schema."""


class Algorithm(dict):
    """One (package, algorithm) row: `fixed`, `adaptive` and `newton` are bools, `order` an int."""

    @property
    def name(self):
        return self["algorithm"]

    @property
    def package(self):
        return self["package"]

    @property
    def implicit(self):
        """True for every family that solves stage equations."""
        return self["family"] != "erk"

    def supports(self, package, kind=None):
        """True when this row is the package's and runs the stepping kind (either kind when None)."""
        if package != self["package"]:
            return False
        kinds = KINDS if kind is None else (kind,)
        return any(self[k] for k in kinds)


def _bool(value, where):
    if value == "true":
        return True
    if value == "false":
        return False
    raise CatalogueError("{0}: expected true or false, got '{1}'".format(where, value))


def load_algorithms():
    """Every (package, algorithm) row in declaration order; raises CatalogueError on a bad header, an unknown package or family, a repeated (package, algorithm), a row with no capability, or rows of one algorithm that disagree on display, family or order."""
    with open(ALGORITHMS_CSV, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        rows = list(reader)
    if header != COLUMNS:
        raise CatalogueError("algorithms.csv columns must be {0}, got {1}".format(
            ", ".join(COLUMNS), ", ".join(header)))
    algorithms = []
    facts = {}
    seen = set()
    for index, row in enumerate(rows, start=2):
        where = "algorithms.csv line {0}".format(index)
        if row["package"] not in PACKAGES:
            raise CatalogueError("{0}: unknown package '{1}'".format(where, row["package"]))
        if row["family"] not in FAMILIES:
            raise CatalogueError("{0}: unknown family '{1}'".format(where, row["family"]))
        try:
            row["order"] = int(row["order"])
        except ValueError:
            raise CatalogueError("{0}: order must be an integer".format(where))
        for column in CAPABILITIES:
            row[column] = _bool(row[column], where + " " + column)
        if not (row["fixed"] or row["adaptive"]):
            raise CatalogueError("{0}: {1} runs {2} neither fixed nor adaptive".format(
                where, row["package"], row["algorithm"]))
        pair = (row["package"], row["algorithm"])
        if pair in seen:
            raise CatalogueError("{0}: repeated row for {1} {2}".format(where, *pair))
        seen.add(pair)
        stated = tuple(row[f] for f in FACTS)
        if facts.setdefault(row["algorithm"], stated) != stated:
            raise CatalogueError("{0}: {1} states {2} = {3}; an earlier row states {4}".format(
                where, row["algorithm"], ", ".join(FACTS), stated, facts[row["algorithm"]]))
        algorithms.append(Algorithm(row))
    return algorithms


def algorithm_names():
    """Every algorithm name once, in declaration order."""
    return list(dict.fromkeys(row["algorithm"] for row in load_algorithms()))


def algorithm_facts(name):
    """{display, family, order} of an algorithm, shared by its rows; exits on an unknown name."""
    for row in load_algorithms():
        if row["algorithm"] == name:
            return {f: row[f] for f in FACTS}
    raise SystemExit("unknown algorithm '{0}' (expected one of: all, {1})"
                     .format(name, ", ".join(algorithm_names())))


def get_algorithm(name, package):
    """The (package, algorithm) row; None when the package has none."""
    for row in load_algorithms():
        if row["algorithm"] == name and row["package"] == package:
            return row
    return None


def supported_for(package, kind=None):
    """Algorithm names a package runs, in declaration order, in the stepping kind if given."""
    return tuple(row["algorithm"] for row in load_algorithms()
                 if row.supports(package, kind))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # <package> [fixed|adaptive]: the algorithms it runs, one per line.
        for name in supported_for(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None):
            print(name)
    else:
        for name in algorithm_names():
            print(name)
