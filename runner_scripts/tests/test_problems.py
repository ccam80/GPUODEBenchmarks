"""Registry tests: the problem table, its derived grids and the output paths."""

import os
import sys
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
            self.assertIsInstance(row["frameworks"], tuple)
            self.assertGreater(row["states"], 0)
            self.assertGreater(row["duration"], 0.0)

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

    def test_timing_dt_takes_a_thousand_steps(self):
        self.assertAlmostEqual(self.problem["duration"] / 1000.0,
                               self.problem.timing_dt)

    def test_sweep_spans_the_range(self):
        grid = self.problem.sweep(4)
        self.assertAlmostEqual(self.problem["sweep_min"], grid[0])
        self.assertAlmostEqual(self.problem["sweep_max"], grid[-1])
        self.assertEqual(4, len(grid))


class PathTests(unittest.TestCase):
    def setUp(self):
        self.problem = get_problem(DEFAULT_PROBLEM)

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
