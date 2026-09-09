"""The algorithm catalogue: typed rows, capability columns naming known packages, and the Julia constructor table covering every Julia capability and golden algorithm."""

import csv
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from algorithms import (  # noqa: E402
    ALGORITHMS_CSV, FAMILIES, KINDS, algorithm_names, get_algorithm, load_algorithms, supported_for,
)
from problems import load_problems  # noqa: E402
from store import PACKAGES  # noqa: E402

JULIA_CSV = os.path.join(os.path.dirname(HERE), "julia_algorithms.csv")


def julia_rows():
    with open(JULIA_CSV, newline="", encoding="utf-8") as handle:
        return {row["algorithm"]: row for row in csv.DictReader(handle)}


class CatalogueTests(unittest.TestCase):
    def test_columns_are_the_capability_schema(self):
        with open(ALGORITHMS_CSV, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(header, ["algorithm", "display", "family", "order", "fixed", "adaptive"])

    def test_rows_are_typed_and_name_known_packages(self):
        rows = load_algorithms()
        self.assertEqual(len(rows), 23)
        for row in rows:
            self.assertIn(row["family"], FAMILIES)
            self.assertIsInstance(row["order"], int)
            for kind in KINDS:
                self.assertIsInstance(row[kind], tuple)
                for package in row[kind]:
                    self.assertIn(package, PACKAGES, row.name)
            self.assertTrue(row["fixed"] or row["adaptive"], row.name)
            self.assertEqual(row.implicit, row["family"] != "erk")
        names = algorithm_names()
        self.assertEqual(len(names), len(set(names)))

    def test_lookups(self):
        self.assertEqual(get_algorithm("tsit5")["order"], 5)
        with self.assertRaises(SystemExit):
            get_algorithm("nosuchalgorithm")
        self.assertEqual(supported_for("pytorch"), ("euler", "classical-rk4", "tsit5"))
        self.assertEqual(supported_for("pytorch", "adaptive"), ())
        self.assertEqual(supported_for("myokit_cuda"), ("euler",))
        self.assertEqual(supported_for("cpp"), ("classical-rk4", "cash-karp-54"))
        for package in PACKAGES:
            union = set(supported_for(package, "fixed")) | set(supported_for(package, "adaptive"))
            self.assertEqual(union, set(supported_for(package)))
        self.assertTrue(get_algorithm("tsit5").supports("julia_cpu", "adaptive"))
        self.assertTrue(get_algorithm("cash-karp-54").supports("cubie", "fixed"))
        self.assertTrue(get_algorithm("backwards_euler").supports("julia_cpu", "adaptive"))
        self.assertFalse(get_algorithm("backwards_euler").supports("cubie", "adaptive"))
        self.assertFalse(get_algorithm("euler").supports("julia_cpu"))
        # No error estimate: no package runs these adaptively.
        for name in ("euler", "classical-rk4", "trapezoidal_dirk", "implicit_midpoint", "sdirk_2_2"):
            self.assertEqual(get_algorithm(name)["adaptive"], (), name)

    def test_julia_constructors_cover_every_julia_capability_and_golden_algorithm(self):
        table = julia_rows()
        self.assertEqual(list(next(iter(table.values()))), ["algorithm", "julia_cpu", "julia_gpu", "notes"])
        for row in load_algorithms():
            for package in ("julia_cpu", "julia_gpu"):
                if row.supports(package):
                    self.assertTrue(table.get(row.name, {}).get(package), "{0} {1}".format(row.name, package))
        for name in table:
            self.assertTrue(name in algorithm_names() or table[name]["notes"].startswith("golden"), name)
        for problem in load_problems():
            golden = problem["golden_algorithm"]
            self.assertTrue(table[golden]["julia_cpu"], golden)


if __name__ == "__main__":
    unittest.main()
