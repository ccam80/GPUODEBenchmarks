"""Registry tests: the algorithm table and the per-framework subsets."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from algorithms import (  # noqa: E402
    algorithm_names, get_algorithm, load_algorithms, resolve_algorithms,
    supported_for,
)
from wp_common import parse_bench_args  # noqa: E402


class RegistryTests(unittest.TestCase):
    def test_rows_are_typed(self):
        for row in load_algorithms():
            self.assertIsInstance(row["fixed"], tuple)
            self.assertIsInstance(row["adaptive"], tuple)
            self.assertIn(row["family"], ("explicit", "implicit"))
            self.assertTrue(row["fixed"] or row["adaptive"],
                            "{0} runs in no mode".format(row["algorithm"]))

    def test_unknown_algorithm_exits(self):
        with self.assertRaises(SystemExit):
            get_algorithm("nosuchalgorithm")

    def test_names_are_unique(self):
        names = algorithm_names()
        self.assertEqual(len(names), len(set(names)))

    def test_supported_is_the_union_of_the_modes(self):
        for framework in ("cubie", "julia", "jax", "pytorch", "cpp",
                          "myokit_cuda"):
            union = set(supported_for(framework, "fixed"))
            union |= set(supported_for(framework, "adaptive"))
            self.assertEqual(union, set(supported_for(framework)))

    def test_the_implicit_set_reaches_cubie_and_julia(self):
        implicit = [row["algorithm"] for row in load_algorithms()
                    if row["family"] == "implicit"]
        self.assertTrue(implicit)
        for name in implicit:
            self.assertIn(name, supported_for("cubie"))

    def test_resolve_drops_unsupported_names_but_rejects_unknown_ones(self):
        self.assertEqual([], resolve_algorithms("radau_iia_5", "pytorch"))
        with self.assertRaises(SystemExit):
            resolve_algorithms("nosuchalgorithm", "cubie")

    def test_resolve_all_is_the_framework_set(self):
        self.assertEqual(list(supported_for("jax")),
                         resolve_algorithms("all", "jax"))


class ParseTests(unittest.TestCase):
    def test_bench_args_resolve_both_axes(self):
        n, wp, algorithms, problems = parse_bench_args(
            ["1024", "wp", "kvaerno3", "--problem", "lorenz"], "cubie")
        self.assertEqual(1024, n)
        self.assertTrue(wp)
        self.assertEqual(["kvaerno3"], algorithms)
        self.assertEqual(["lorenz"], [p.name for p in problems])

    def test_an_algorithm_the_framework_lacks_yields_an_empty_list(self):
        _, _, algorithms, _ = parse_bench_args(
            ["1024", "radau_iia_5"], "pytorch")
        self.assertEqual([], algorithms)

    def test_an_unknown_algorithm_exits(self):
        with self.assertRaises(SystemExit):
            parse_bench_args(["1024", "nosuchalgorithm"], "cubie")


if __name__ == "__main__":
    unittest.main()
