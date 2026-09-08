"""Registry tests: the algorithm table and the per-framework subsets."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from algorithms import (  # noqa: E402
    FAMILIES, MODES, algorithm_names, get_algorithm, load_algorithms,
    ne_algorithms, overlap_algorithms, resolve_algorithms, resolve_modes,
    supported_for,
)
from wp_common import N_WP, parse_bench_args  # noqa: E402


class RegistryTests(unittest.TestCase):
    def test_rows_are_typed(self):
        for row in load_algorithms():
            self.assertIsInstance(row["fixed"], tuple)
            self.assertIsInstance(row["adaptive"], tuple)
            self.assertIn(row["family"], FAMILIES)
            self.assertIsInstance(row["order"], int)
            self.assertTrue(row["fixed"] or row["adaptive"] or row["ne"],
                            "{0} is in no suite".format(row["algorithm"]))

    def test_suite_memberships(self):
        ne = [row["algorithm"] for row in ne_algorithms()]
        self.assertEqual(len(ne), 21)
        for row in ne_algorithms():
            self.assertTrue(row["julia_cpu"], row["algorithm"])
        self.assertEqual([row["algorithm"] for row in overlap_algorithms()],
                         ["tsit5", "rosenbrock23_sciml", "kvaerno3", "vern7",
                          "kvaerno5"])
        self.assertEqual([row["algorithm"] for row in ne_algorithms("tsit5,vern7")],
                         ["tsit5", "vern7"])
        with self.assertRaises(SystemExit):
            ne_algorithms("nosuchalgorithm")
        for row in load_algorithms():
            if row["ne_adaptive"]:
                self.assertTrue(row["ne"], row["algorithm"])

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

    def test_the_timed_implicit_set_reaches_cubie(self):
        implicit = [row["algorithm"] for row in load_algorithms()
                    if row["family"] != "erk" and (row["fixed"] or row["adaptive"])]
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
    def test_bench_args_resolve_every_axis(self):
        ns, analysis, algorithms, problems, modes = parse_bench_args(
            ["wp", "kvaerno3", "--problem", "lorenz"], "cubie")
        self.assertEqual([N_WP], ns)
        self.assertEqual("wp", analysis)
        self.assertEqual(["kvaerno3"], algorithms)
        self.assertEqual(["lorenz"], [p.name for p in problems])
        self.assertEqual(MODES, modes)

    def test_mode_narrows_and_rejects_unknown_names(self):
        self.assertEqual(parse_bench_args(["wp", "--mode", "adaptive"], "cubie")[4], ("adaptive",))
        self.assertEqual(parse_bench_args(["wp", "--mode=fixed,adaptive"], "cubie")[4], MODES)
        self.assertEqual(resolve_modes("adaptive,fixed"), MODES)
        with self.assertRaises(SystemExit):
            parse_bench_args(["wp", "--mode", "sideways"], "cubie")
        with self.assertRaises(SystemExit):
            parse_bench_args(["wp", "--mode"], "cubie")

    def test_a_timing_count_parses_without_wp(self):
        ns, analysis, _, _, _ = parse_bench_args(["1024", "tsit5"], "cubie")
        self.assertEqual([1024], ns)
        self.assertEqual("times", analysis)

    def test_an_algorithm_the_framework_lacks_yields_an_empty_list(self):
        _, _, algorithms, _, _ = parse_bench_args(
            ["1024", "radau_iia_5"], "pytorch")
        self.assertEqual([], algorithms)

    def test_an_unknown_algorithm_exits(self):
        with self.assertRaises(SystemExit):
            parse_bench_args(["1024", "nosuchalgorithm"], "cubie")


if __name__ == "__main__":
    unittest.main()
