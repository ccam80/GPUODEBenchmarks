"""Registry tests: the problem table, its derived grids and the output paths."""

import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from bench_key import data_dir, group_dir  # noqa: E402
from problems import (  # noqa: E402
    DEFAULT_PROBLEM, get_problem, load_problems, problem_names,
    resolve_problems,
)
from wp_common import dts_for, golden_path, times_outfile, wp_outfile  # noqa: E402


class RegistryTests(unittest.TestCase):
    def test_default_problem_is_registered(self):
        self.assertIn(DEFAULT_PROBLEM, problem_names())

    def test_rows_are_typed(self):
        for row in load_problems():
            self.assertIsInstance(row["states"], int)
            self.assertIsInstance(row["duration"], float)
            self.assertIsInstance(row["golden_tol"], float)
            self.assertIsInstance(row["frameworks"], tuple)
            self.assertGreater(row["states"], 0)
            self.assertGreater(row["duration"], 0.0)
            self.assertGreater(row["golden_tol"], 0.0)
            self.assertIn(row["sweep_scale"], ("linear", "log"))

    def test_unknown_problem_exits(self):
        with self.assertRaises(SystemExit):
            get_problem("nosuchproblem")

    def test_resolve_filters_by_framework(self):
        row = get_problem(DEFAULT_PROBLEM)
        framework = row["frameworks"][0]
        self.assertIn(row.name,
                      [r.name for r in resolve_problems("all", framework)])
        self.assertEqual([], resolve_problems("all", "nosuchframework"))

    def test_resolve_accepts_a_comma_list(self):
        names = [r.name for r in resolve_problems(DEFAULT_PROBLEM)]
        self.assertEqual([DEFAULT_PROBLEM], names)

    def test_exclusions_gate_framework_algorithm_pairs(self):
        row = get_problem("lorenz96")
        self.assertTrue(row.runs("julia", "tsit5"))
        self.assertFalse(row.runs("julia", "rosenbrock23_sciml"))
        self.assertFalse(row.runs("julia", "kvaerno3"))
        self.assertTrue(row.runs("cubie", "rosenbrock23_sciml"))
        self.assertFalse(row.runs("nosuchframework", "tsit5"))
        # A row without exclusions runs everything its frameworks run.
        self.assertTrue(get_problem("lorenz96_20").runs("julia", "kvaerno3"))

    def test_bench_args_accept_an_n_list(self):
        from wp_common import N_WP, STATES_N, parse_bench_args
        ns, analysis, _, _ = parse_bench_args(["32,8,128"], "cubie")
        self.assertEqual([8, 32, 128], ns)
        self.assertEqual("times", analysis)
        ns, analysis, _, _ = parse_bench_args(["wp"], "cubie")
        self.assertEqual("wp", analysis)
        self.assertEqual([N_WP], ns)
        from wp_common import STATES_GRID
        ns, analysis, _, _ = parse_bench_args(["states"], "cubie")
        self.assertEqual("states", analysis)
        self.assertEqual(list(STATES_GRID), ns)
        ns, analysis, _, _ = parse_bench_args(["states:40"], "cubie")
        self.assertEqual("states", analysis)
        self.assertEqual([4, 8, 16, 32], ns)
        ns, analysis, _, _ = parse_bench_args(["states:32,36,40"], "cubie")
        self.assertEqual("states", analysis)
        self.assertEqual([32, 36, 40], ns)
        ns, analysis, _, _ = parse_bench_args(["warm:32,8"], "cubie")
        self.assertEqual("warm", analysis)
        self.assertEqual([8, 32], ns)
        ns, analysis, _, _ = parse_bench_args(["warm"], "cubie")
        self.assertEqual("warm", analysis)
        self.assertEqual([], ns)

    def test_states_rows_resize_lorenz96(self):
        from problems import states_row
        row = states_row(16)
        self.assertEqual("lorenz96", row.name)
        self.assertEqual(16, row["states"])
        self.assertTrue(row.runs("julia", "rosenbrock23_sciml"))


class GridTests(unittest.TestCase):
    def setUp(self):
        self.problem = get_problem(DEFAULT_PROBLEM)

    def test_dt_grids_are_dyadic_fractions_of_the_duration(self):
        duration = self.problem["duration"]
        for dt in self.problem.dts() + self.problem.ne_dts():
            ratio = duration / dt
            self.assertEqual(ratio, float(int(ratio)))
            self.assertEqual(0, int(ratio) & (int(ratio) - 1))

    def test_euler_grid_is_finer(self):
        self.assertLess(dts_for("euler", self.problem)[-1],
                        dts_for("tsit5", self.problem)[-1])

    def test_timing_dt_is_a_dyadic_fraction_of_the_duration(self):
        self.assertEqual(self.problem["duration"] * 2.0 ** -10,
                         self.problem.timing_dt)

    def test_sweep_spans_the_range(self):
        for row in load_problems():
            grid = row.sweep(4)
            self.assertAlmostEqual(row["sweep_min"], grid[0])
            self.assertAlmostEqual(row["sweep_max"], grid[-1])
            self.assertEqual(4, len(grid))

    def test_log_sweeps_are_geometric(self):
        for row in load_problems():
            if row["sweep_scale"] != "log":
                continue
            grid = row.sweep(5)
            ratios = [grid[i + 1] / grid[i] for i in range(len(grid) - 1)]
            for ratio in ratios[1:]:
                self.assertAlmostEqual(ratios[0], ratio, places=6)

    def test_log_sweeps_need_a_positive_minimum(self):
        row = get_problem(DEFAULT_PROBLEM)
        row["sweep_scale"], row["sweep_min"] = "log", 0.0
        with self.assertRaises(SystemExit):
            row.sweep(4)


class PathTests(unittest.TestCase):
    def setUp(self):
        self.problem = get_problem(DEFAULT_PROBLEM)
        # The path helpers create their directories, so run in a temp cwd.
        self._cwd = os.getcwd()
        self._tmp = tempfile.mkdtemp()
        os.chdir(self._tmp)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_output_paths_carry_the_problem(self):
        for path in (times_outfile("CUBIE", "Cubie", "fixed", "euler", "k",
                                   self.problem),
                     wp_outfile("CUBIE", "Cubie", "fixed", "euler", "k",
                                self.problem)):
            self.assertIn(os.path.join("k", self.problem.name), path)

    def test_golden_path_names_the_problem(self):
        self.assertIn(self.problem.name, golden_path(self.problem))

    def test_directories_accept_a_row_or_a_name(self):
        self.assertEqual(data_dir("CUBIE", "k", problem=self.problem),
                         data_dir("CUBIE", "k", problem=self.problem.name))
        self.assertEqual(group_dir("k", self.problem),
                         group_dir("k", self.problem.name))


if __name__ == "__main__":
    unittest.main()
