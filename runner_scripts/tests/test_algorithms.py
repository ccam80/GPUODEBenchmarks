"""The algorithm catalogue: one typed row per (package, algorithm), the loader's schema and consistency checks, and the Julia constructor table covering every Julia capability and golden algorithm."""

import csv
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import algorithms  # noqa: E402
from algorithms import (  # noqa: E402
    ALGORITHMS_CSV, CAPABILITIES, COLUMNS, FAMILIES, KINDS, CatalogueError, algorithm_facts,
    algorithm_names, get_algorithm, load_algorithms, supported_for,
)
from problems import load_problems  # noqa: E402
from store import PACKAGES  # noqa: E402

JULIA_CSV = os.path.join(os.path.dirname(HERE), "julia_algorithms.csv")


def julia_rows():
    with open(JULIA_CSV, newline="", encoding="utf-8") as handle:
        return {row["algorithm"]: row for row in csv.DictReader(handle)}


class CatalogueTests(unittest.TestCase):
    def test_columns_are_one_row_per_package_and_algorithm(self):
        with open(ALGORITHMS_CSV, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(tuple(header), COLUMNS)
        self.assertEqual(COLUMNS, ("algorithm", "package", "display", "family", "order",
                                   "fixed", "adaptive", "newton"))

    def test_rows_are_typed_and_name_known_packages(self):
        rows = load_algorithms()
        self.assertEqual(len(rows), 82)
        self.assertEqual(len(algorithm_names()), 23)
        pairs = [(row.package, row.name) for row in rows]
        self.assertEqual(len(pairs), len(set(pairs)))
        for row in rows:
            self.assertIn(row["package"], PACKAGES)
            self.assertIn(row["family"], FAMILIES)
            self.assertIsInstance(row["order"], int)
            for column in CAPABILITIES:
                self.assertIsInstance(row[column], bool, (row.name, column))
            self.assertTrue(row["fixed"] or row["adaptive"], (row.package, row.name))
            self.assertEqual(row.implicit, row["family"] != "erk")
            # A Newton tolerance belongs to a stage solve; explicit rows never take one.
            if not row.implicit:
                self.assertFalse(row["newton"], (row.package, row.name))
        # Every row of an algorithm states the same facts.
        for name in algorithm_names():
            facts = {(r["display"], r["family"], r["order"]) for r in rows if r.name == name}
            self.assertEqual(len(facts), 1, name)

    def test_newton_capability_follows_the_package(self):
        # The kernel package takes no Newton tolerance; cubie, jax and julia_cpu do on their implicit rows.
        for row in load_algorithms():
            if row.package == "julia_gpu":
                self.assertFalse(row["newton"], row.name)
            elif row.package in ("cubie", "cubie_mlir", "jax", "julia_cpu"):
                self.assertEqual(row["newton"], row.implicit, (row.package, row.name))
            else:
                self.assertFalse(row["newton"], (row.package, row.name))
        self.assertTrue(get_algorithm("kvaerno3", "jax")["newton"])
        self.assertFalse(get_algorithm("kvaerno3", "julia_gpu")["newton"])
        self.assertFalse(get_algorithm("tsit5", "cubie")["newton"])

    def test_lookups(self):
        self.assertEqual(algorithm_facts("tsit5"), {"display": "Tsit5", "family": "erk", "order": 5})
        with self.assertRaises(SystemExit):
            algorithm_facts("nosuchalgorithm")
        self.assertIsNone(get_algorithm("euler", "julia_cpu"))
        self.assertIsNone(get_algorithm("nosuchalgorithm", "cubie"))
        self.assertEqual(supported_for("pytorch"), ("euler", "classical-rk4", "tsit5"))
        self.assertEqual(supported_for("pytorch", "adaptive"), ())
        self.assertEqual(supported_for("myokit_cuda"), ("euler",))
        self.assertEqual(supported_for("cpp"), ("classical-rk4", "cash-karp-54"))
        self.assertEqual(supported_for("cpp", "fixed"), ("classical-rk4",))
        for package in PACKAGES:
            union = set(supported_for(package, "fixed")) | set(supported_for(package, "adaptive"))
            self.assertEqual(union, set(supported_for(package)))
        self.assertTrue(get_algorithm("tsit5", "julia_cpu").supports("julia_cpu", "adaptive"))
        self.assertFalse(get_algorithm("tsit5", "julia_cpu").supports("cubie", "adaptive"))
        self.assertTrue(get_algorithm("cash-karp-54", "cubie").supports("cubie", "fixed"))
        self.assertTrue(get_algorithm("backwards_euler", "julia_cpu")["adaptive"])
        self.assertFalse(get_algorithm("backwards_euler", "cubie")["adaptive"])
        # No error estimate: no package runs these adaptively.
        for name in ("euler", "classical-rk4", "trapezoidal_dirk", "implicit_midpoint", "sdirk_2_2"):
            for row in load_algorithms():
                if row.name == name:
                    self.assertFalse(row["adaptive"], (row.package, name))

    def test_julia_constructors_cover_every_julia_capability_and_golden_algorithm(self):
        table = julia_rows()
        self.assertEqual(list(next(iter(table.values()))), ["algorithm", "julia_cpu", "julia_gpu", "notes"])
        for row in load_algorithms():
            if row.package in ("julia_cpu", "julia_gpu"):
                self.assertTrue(table.get(row.name, {}).get(row.package), "{0} {1}".format(row.name, row.package))
        for name in table:
            self.assertTrue(name in algorithm_names() or table[name]["notes"].startswith("golden"), name)
        for problem in load_problems():
            golden = problem["golden_algorithm"]
            self.assertTrue(table[golden]["julia_cpu"], golden)


class LoaderChecksTests(unittest.TestCase):
    """The loader refuses a table off the schema; the checks run on a temporary copy."""

    HEADER = "algorithm,package,display,family,order,fixed,adaptive,newton\n"
    GOOD = HEADER + ("tsit5,cubie,Tsit5,erk,5,true,true,false\n"
                     "tsit5,jax,Tsit5,erk,5,true,true,false\n"
                     "kvaerno3,cubie,Kvaerno3,dirk,3,true,true,true\n")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="algorithms_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "algorithms.csv")
        original = algorithms.ALGORITHMS_CSV
        algorithms.ALGORITHMS_CSV = self.path
        self.addCleanup(setattr, algorithms, "ALGORITHMS_CSV", original)

    def write(self, text):
        with open(self.path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def test_a_consistent_table_loads(self):
        self.write(self.GOOD)
        rows = load_algorithms()
        self.assertEqual([(r.package, r.name) for r in rows], [("cubie", "tsit5"), ("jax", "tsit5"), ("cubie", "kvaerno3")])
        self.assertEqual(algorithm_names(), ["tsit5", "kvaerno3"])
        self.assertEqual(rows[2]["order"], 3)
        self.assertIs(rows[2]["newton"], True)

    def test_rows_of_one_algorithm_disagreeing_on_the_facts_are_refused(self):
        for label, bad in (("display", "tsit5,jax,Tsit-5,erk,5,true,true,false\n"),
                           ("family", "tsit5,jax,Tsit5,dirk,5,true,true,false\n"),
                           ("order", "tsit5,jax,Tsit5,erk,4,true,true,false\n")):
            self.write(self.GOOD.replace("tsit5,jax,Tsit5,erk,5,true,true,false\n", bad))
            with self.assertRaises(CatalogueError, msg=label) as raised:
                load_algorithms()
            self.assertIn("tsit5", str(raised.exception))

    def test_other_schema_breaks_are_refused(self):
        bad = {
            "header": self.GOOD.replace("newton", "implicit"),
            "unknown package": self.GOOD + "tsit5,fortran,Tsit5,erk,5,true,true,false\n",
            "unknown family": self.GOOD + "euler,cubie,Euler,explicit,1,true,false,false\n",
            "order not an integer": self.GOOD + "euler,cubie,Euler,erk,one,true,false,false\n",
            "capability not a bool": self.GOOD + "euler,cubie,Euler,erk,1,yes,false,false\n",
            "no capability": self.GOOD + "euler,cubie,Euler,erk,1,false,false,false\n",
            "repeated row": self.GOOD + "tsit5,cubie,Tsit5,erk,5,true,true,false\n",
        }
        for label, text in bad.items():
            self.write(text)
            with self.assertRaises(CatalogueError, msg=label):
                load_algorithms()


if __name__ == "__main__":
    unittest.main()
