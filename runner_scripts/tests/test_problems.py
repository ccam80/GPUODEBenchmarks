"""The problem catalogue: typed rows, package membership by package name, and the construction parameters."""

import csv
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from problems import (  # noqa: E402
    DEFAULT_PROBLEM, PROBLEMS_CSV, RESIZABLE_PROBLEM, as_problem, get_problem, load_problems,
    problem_names, resolve_problems,
)
from store import PACKAGES  # noqa: E402


class RegistryTests(unittest.TestCase):
    def test_columns(self):
        with open(PROBLEMS_CSV, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(header, ["problem", "display", "states", "duration", "sweep_parameter", "sweep_min",
                                  "sweep_max", "sweep_scale", "golden_algorithm", "golden_tol", "frameworks"])

    def test_default_problem_is_registered(self):
        self.assertIn(DEFAULT_PROBLEM, problem_names())
        self.assertEqual(len(problem_names()), 8)

    def test_rows_are_typed_and_name_known_packages(self):
        for row in load_problems():
            self.assertIsInstance(row["states"], int)
            self.assertIsInstance(row["duration"], float)
            self.assertIsInstance(row["golden_tol"], float)
            self.assertIsInstance(row["frameworks"], tuple)
            self.assertGreater(row["states"], 0)
            self.assertGreater(row["duration"], 0.0)
            self.assertGreater(row["golden_tol"], 0.0)
            self.assertIn(row["sweep_scale"], ("linear", "log"))
            self.assertTrue(row["golden_algorithm"])
            for package in row["frameworks"]:
                self.assertIn(package, PACKAGES, row.name)
            self.assertIn("julia_cpu", row["frameworks"], row.name)

    def test_unknown_problem_exits(self):
        with self.assertRaises(SystemExit):
            get_problem("nosuchproblem")

    def test_as_problem_takes_a_row_or_a_name(self):
        row = get_problem(DEFAULT_PROBLEM)
        self.assertIs(as_problem(row), row)
        self.assertEqual(as_problem(DEFAULT_PROBLEM), row)
        with self.assertRaises(SystemExit):
            as_problem("nosuchproblem")

    def test_resolve_filters_by_package(self):
        self.assertEqual([r.name for r in resolve_problems("all", "myokit_cuda")],
                         ["lorenz", "lorenz96", "pleiades"])
        self.assertEqual([], resolve_problems("all", "nosuchpackage"))
        self.assertEqual([DEFAULT_PROBLEM], [r.name for r in resolve_problems(DEFAULT_PROBLEM)])
        self.assertEqual(len(resolve_problems("all", "julia_gpu")), 8)

    def test_system_params(self):
        lorenz96 = get_problem(RESIZABLE_PROBLEM)
        self.assertEqual(lorenz96.system_params(), {"states": 32})
        self.assertEqual(lorenz96.system_params(8), {"states": 8})
        self.assertEqual(lorenz96.resized(8)["states"], 8)
        self.assertEqual(lorenz96["states"], 32)
        pollu = get_problem("pollu")
        self.assertEqual(pollu.system_params(), {})
        self.assertEqual(pollu.system_params(20), {})
        with self.assertRaises(ValueError):
            pollu.system_params(8)


if __name__ == "__main__":
    unittest.main()
