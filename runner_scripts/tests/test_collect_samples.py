"""Master run-times table: what the path tags, how blocks are separated, and the
rows a half-written log contributes."""

import csv
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import collect_samples  # noqa: E402
from wp_common import (  # noqa: E402
    SAMPLE_FIELDS, append_samples, sample_point,
)


def log_path(data_root, package, key, problem, name):
    """Create a log's directory under a data root and return its file path."""
    directory = os.path.join(data_root, package, key, problem)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name)


def write_leg(path, n, samples, transfers="both", problem="lorenz",
              algorithm="tsit5", mode="adaptive"):
    """Append one timed leg's attempts, warm-up first."""
    append_samples(path, sample_point("times", problem, algorithm, mode, n, 3),
                   transfers, samples)


class FieldTests(unittest.TestCase):
    def test_the_log_columns_are_the_writers_own(self):
        self.assertEqual(SAMPLE_FIELDS, collect_samples.SAMPLE_FIELDS)

    def test_the_master_keeps_them_behind_the_tags_the_path_carries(self):
        self.assertEqual(("package", "key", "prefix", "series") + SAMPLE_FIELDS,
                         collect_samples.MASTER_FIELDS)

    def test_every_sort_key_is_a_column(self):
        for field in collect_samples.SORT_FIELDS:
            self.assertIn(field, collect_samples.MASTER_FIELDS)


class TagTests(unittest.TestCase):
    def test_the_package_and_machine_directories_and_the_writer_prefix(self):
        path = os.path.join("data", "CUBIE_MLIR", "linux_RTX-2060-SUPER",
                            "lorenz", "Cubie_mlir_samples_times_fixed_tsit5.csv")
        self.assertEqual(("CUBIE_MLIR", "linux_RTX-2060-SUPER", "Cubie_mlir"),
                         collect_samples.tags_for(path, "data"))

    def test_a_log_above_the_package_directories_is_tagged_with_what_it_has(self):
        path = os.path.join("data", "Cubie_samples_times_fixed_tsit5.csv")
        self.assertEqual(("", "", "Cubie"),
                         collect_samples.tags_for(path, "data"))

    def test_only_the_logs_are_picked_up(self):
        self.assertTrue(collect_samples.is_samples_file(
            "Cubie_samples_times_fixed_tsit5.csv"))
        self.assertFalse(collect_samples.is_samples_file(
            "Cubie_times_fixed_tsit5.txt"))
        self.assertFalse(collect_samples.is_samples_file("golden_lorenz.csv"))


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="collect_test_")
        self.path = log_path(self.root, "CUBIE", "key_gpu", "lorenz",
                             "Cubie_samples_times_adaptive_tsit5.csv")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_each_leg_is_its_own_block_and_a_repeated_leg_lands_after_it(self):
        write_leg(self.path, 8, [30.0, 2.0, 3.0])
        write_leg(self.path, 16, [31.0, 4.0, 5.0])
        write_leg(self.path, 8, [29.0, 1.5, 1.6])
        rows = collect_samples.collect(self.root)
        blocks = {}
        for row in rows:
            blocks.setdefault((row["n"], row["series"]), []).append(row["ms"])
        self.assertEqual([("16", 1), ("8", 0), ("8", 2)], sorted(blocks))
        self.assertEqual(["30.000000", "2.000000", "3.000000"],
                         blocks[("8", 0)])
        self.assertEqual(["29.000000", "1.500000", "1.600000"],
                         blocks[("8", 2)])

    def test_the_min_over_a_block_past_the_warm_up_is_the_reduced_time(self):
        write_leg(self.path, 8, [30.0, 2.0, 3.0])
        rows = collect_samples.collect(self.root)
        timed = [float(row["ms"]) for row in rows if row["repeat"] != "0"]
        self.assertAlmostEqual(2.0, min(timed), places=6)

    def test_a_torn_last_line_is_dropped_and_reported(self):
        write_leg(self.path, 8, [30.0, 2.0, 3.0])
        with open(self.path, "a") as handle:
            handle.write("times,lorenz,tsit5,adaptive,both,none,nan,16")
        skipped = []
        rows = collect_samples.collect(
            self.root, on_skip=lambda path, line, why: skipped.append(line))
        self.assertEqual(3, len(rows))
        self.assertEqual([5], skipped)

    def test_a_log_with_foreign_columns_is_left_alone(self):
        with open(self.path, "w") as handle:
            handle.write("n,ms\n8,1.0\n")
        skipped = []
        rows = collect_samples.collect(
            self.root, on_skip=lambda path, line, why: skipped.append(line))
        self.assertEqual([], rows)
        self.assertEqual([1], skipped)

    def test_rows_from_every_package_are_gathered_and_sorted_by_leg(self):
        write_leg(self.path, 16, [31.0, 4.0])
        write_leg(self.path, 8, [30.0, 2.0])
        other = log_path(self.root, "CPP", "key_gpu", "lorenz",
                         "MPGOS_samples_times_fixed_classical-rk4.csv")
        write_leg(other, 8, [10.0, 1.0], mode="fixed",
                  algorithm="classical-rk4")
        rows = collect_samples.collect(self.root)
        self.assertEqual(["CPP", "CPP", "CUBIE", "CUBIE", "CUBIE", "CUBIE"],
                         [row["package"] for row in rows])
        # n sorts as a number, so 8 precedes 16 whatever order they were written.
        self.assertEqual(["8", "8", "8", "8", "16", "16"],
                         [row["n"] for row in rows])


class MasterFileTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="collect_master_")
        self.out = os.path.join(self.root, "data", "master_run_times.csv")
        self.path = log_path(os.path.join(self.root, "data"), "CUBIE",
                             "key_gpu", "lorenz",
                             "Cubie_samples_times_adaptive_tsit5.csv")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def read_master(self):
        with open(self.out, newline="") as handle:
            return list(csv.DictReader(handle))

    def test_the_master_is_replaced_whole_rather_than_appended_to(self):
        data_root = os.path.join(self.root, "data")
        write_leg(self.path, 8, [30.0, 2.0])
        argv = ["--data-root", data_root, "--out", self.out, "--quiet"]
        self.assertEqual(0, collect_samples.main(argv))
        first = self.read_master()
        self.assertEqual(0, collect_samples.main(argv))
        self.assertEqual(first, self.read_master())
        self.assertEqual(2, len(first))
        self.assertEqual(collect_samples.MASTER_FIELDS,
                         tuple(first[0]))

    def test_a_dropped_log_leaves_the_master_without_its_rows(self):
        data_root = os.path.join(self.root, "data")
        argv = ["--data-root", data_root, "--out", self.out, "--quiet"]
        write_leg(self.path, 8, [30.0, 2.0])
        collect_samples.main(argv)
        os.remove(self.path)
        collect_samples.main(argv)
        self.assertEqual([], self.read_master())

    def test_a_missing_data_root_is_an_error_and_no_master_is_written(self):
        code = collect_samples.main(["--data-root",
                                     os.path.join(self.root, "absent"),
                                     "--out", self.out, "--quiet"])
        self.assertEqual(1, code)
        self.assertFalse(os.path.exists(self.out))


if __name__ == "__main__":
    unittest.main()
