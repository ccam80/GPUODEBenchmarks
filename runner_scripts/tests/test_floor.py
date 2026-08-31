"""--floor merging: lower-time row merges, wp pair merges, and the no-prune rule."""

import io
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import resume  # noqa: E402
from resume import (  # noqa: E402
    floor_enabled, merge_min_row, merge_wp_row, prune_reruns,
    write_times_row, write_wp_row,
)


class FileCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("BENCH_FLOOR", None)
        os.environ.pop("BENCH_RESUME", None)
        handle, self.path = tempfile.mkstemp(suffix=".txt")
        os.close(handle)
        os.remove(self.path)
        self.addCleanup(lambda: os.path.exists(self.path)
                        and os.remove(self.path))

    def write(self, text):
        with open(self.path, "w") as f:
            f.write(text)

    def lines(self):
        with open(self.path) as f:
            return f.read().splitlines()


class TestFloorEnabled(FileCase):
    def test_off_by_default_and_for_zero(self):
        self.assertFalse(floor_enabled())
        os.environ["BENCH_FLOOR"] = "0"
        self.assertFalse(floor_enabled())

    def test_on_for_one(self):
        os.environ["BENCH_FLOOR"] = "1"
        self.assertTrue(floor_enabled())


class TestMergeMinRow(FileCase):
    def test_appends_to_a_missing_file(self):
        merge_min_row(self.path, 8, (1.5, 0.5))
        self.assertEqual(["8 1.5 0.5"], self.lines())

    def test_keeps_the_lower_value_per_column(self):
        self.write("8 2.0 0.25\n")
        merge_min_row(self.path, 8, (1.5, 0.5))
        self.assertEqual(["8 1.5 0.25"], self.lines())

    def test_nan_never_beats_a_recorded_time(self):
        self.write("8 2.0 0.25\n")
        merge_min_row(self.path, 8, (float("nan"), float("nan")))
        self.assertEqual(["8 2.0 0.25"], self.lines())

    def test_a_finite_time_replaces_a_recorded_nan(self):
        self.write("8 nan nan\n")
        merge_min_row(self.path, 8, (1.5, 0.5))
        self.assertEqual(["8 1.5 0.5"], self.lines())

    def test_other_rows_and_extra_columns_survive(self):
        self.write("8 2.0 0.25\n32 3.0 1.0 12.345\n")
        merge_min_row(self.path, 32, (2.5, 2.0))
        self.assertEqual(["8 2.0 0.25", "32 2.5 1.0 12.345"], self.lines())

    def test_a_new_key_is_appended(self):
        self.write("8 2.0 0.25\n")
        merge_min_row(self.path, 32, (3.0, 1.0))
        self.assertEqual(["8 2.0 0.25", "32 3.0 1.0"], self.lines())

    def test_tab_separator_for_the_mpgos_files(self):
        self.write("8\t2.0\t0.25\n")
        merge_min_row(self.path, 8, (1.5, 0.5), sep="\t")
        self.assertEqual(["8\t1.5\t0.25"], self.lines())

    def test_build_time_column_also_takes_the_lower_value(self):
        self.write("64 2.0 1.0 30.5\n")
        merge_min_row(self.path, 64, (3.0, 0.5, 12.0))
        self.assertEqual(["64 2.0 0.5 12.0"], self.lines())


class TestMergeWpRow(FileCase):
    def test_the_lower_time_brings_its_error_and_errored_percent_along(self):
        self.write("0.001 5.0 1.0000000000e-03 0.0000\n")
        merge_wp_row(self.path, 0.001, 4.0, 2e-3, 12.5)
        self.assertEqual(["0.001 4.0 2.0000000000e-03 12.5000"], self.lines())

    def test_a_higher_time_leaves_the_recorded_pair(self):
        self.write("0.001 5.0 1.0000000000e-03 0.0000\n")
        merge_wp_row(self.path, 0.001, 6.0, 2e-3, 0.0)
        self.assertEqual(["0.001 5.0 1.0000000000e-03 0.0000"], self.lines())

    def test_nan_replaces_nothing_but_fills_a_gap(self):
        self.write("0.001 5.0 1.0000000000e-03 0.0000\n")
        merge_wp_row(self.path, 0.001, float("nan"), float("nan"), 100.0)
        self.assertEqual(["0.001 5.0 1.0000000000e-03 0.0000"], self.lines())
        merge_wp_row(self.path, 0.0005, float("nan"), float("nan"), 100.0)
        self.assertEqual(2, len(self.lines()))
        self.assertEqual("0.0005 nan nan 100.0000", self.lines()[1])

    def test_a_recorded_nan_is_replaced(self):
        self.write("0.001 nan nan\n")
        merge_wp_row(self.path, 0.001, 4.0, 2e-3, 0.0)
        self.assertEqual(["0.001 4.0 2.0000000000e-03 0.0000"], self.lines())

    def test_settings_match_across_formattings(self):
        # Julia prints the raw float; the merge must still find the row.
        self.write("0.0009765625 5.0 1e-03\n")
        merge_wp_row(self.path, 9.765625e-4, 4.0, 2e-3, 0.0)
        self.assertEqual(1, len(self.lines()))
        self.assertIn("4.0", self.lines()[0])


class TestWriteRows(FileCase):
    def test_without_floor_the_row_goes_through_the_handle(self):
        handle = io.StringIO()
        write_times_row(handle, self.path, 8, (1.5, 0.5))
        self.assertEqual("8 1.5 0.5\n", handle.getvalue())
        self.assertFalse(os.path.exists(self.path))

    def test_with_floor_the_row_merges_into_the_file(self):
        os.environ["BENCH_FLOOR"] = "1"
        self.write("8 1.0 0.25\n")
        handle = io.StringIO()
        write_times_row(handle, self.path, 8, (1.5, 0.5))
        self.assertEqual("", handle.getvalue())
        self.assertEqual(["8 1.0 0.25"], self.lines())

    def test_wp_rows_follow_the_same_split(self):
        handle = io.StringIO()
        write_wp_row(handle, self.path, 1e-3, 5.0, 1e-3, 0.0)
        self.assertEqual("0.001 5.0 1.0000000000e-03 0.0000\n", handle.getvalue())
        os.environ["BENCH_FLOOR"] = "1"
        write_wp_row(handle, self.path, 1e-3, 4.0, 2e-3, 0.0)
        self.assertEqual(["0.001 4.0 2.0000000000e-03 0.0000"], self.lines())


class TestFloorNeverPrunes(FileCase):
    def test_prune_is_a_no_op_under_floor_even_with_resume(self):
        self.write("8 1.0 0.5\n32 2.0 1.0\n")
        os.environ["BENCH_RESUME"] = "1"
        os.environ["BENCH_FLOOR"] = "1"
        prune_reruns(self.path, [8, 32])
        self.assertEqual(["8 1.0 0.5", "32 2.0 1.0"], self.lines())


class TestCli(FileCase):
    def test_merge_command_uses_the_tab_separator(self):
        self.write("8\t2.0\t0.25\n")
        resume._cli(["merge", self.path, "tab", "8", "nan", "nan"])
        self.assertEqual(["8\t2.0\t0.25"], self.lines())
        resume._cli(["merge", self.path, "tab", "32", "nan", "nan"])
        self.assertEqual(["8\t2.0\t0.25", "32\tnan\tnan"], self.lines())

    def test_merge_command_takes_a_build_column(self):
        resume._cli(["merge", self.path, "tab", "64", "nan", "nan", "12.5"])
        self.assertEqual(["64\tnan\tnan\t12.5"], self.lines())


class TestNanArithmetic(unittest.TestCase):
    def test_lower_prefers_any_finite_value(self):
        nan = float("nan")
        self.assertEqual(1.0, resume._lower(nan, 1.0))
        self.assertEqual(1.0, resume._lower(1.0, nan))
        self.assertTrue(math.isnan(resume._lower(nan, nan)))
        self.assertEqual(1.0, resume._lower(1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
