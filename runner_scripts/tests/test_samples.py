"""Per-repeat timing log: file layout, row format and the min it reduces to."""

import csv
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from wp_common import (  # noqa: E402
    SAMPLE_FIELDS, SampleLog, samples_outfile, timed_min_ms,
)


class LayoutTests(unittest.TestCase):
    def test_the_name_sits_beside_its_reduced_sibling(self):
        path = samples_outfile("CUBIE", "Cubie", "times", "fixed", "tsit5",
                               "key_gpu", "lorenz")
        self.assertEqual(
            os.path.join("data", "CUBIE", "key_gpu", "lorenz",
                         "Cubie_samples_times_fixed_tsit5.csv"),
            path)
        os.removedirs(os.path.dirname(path))


class WriterTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".csv")
        os.close(handle)
        os.remove(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def rows(self):
        with open(self.path) as handle:
            return list(csv.DictReader(handle))

    def test_a_point_writes_one_row_per_attempt(self):
        with SampleLog(self.path) as log:
            sink = log.sink("times", "lorenz", "tsit5", "fixed", "both", 1024,
                            3)
            timed_min_ms(lambda: None, 4, sink)
        rows = self.rows()
        self.assertEqual(5, len(rows))
        self.assertEqual(list(SAMPLE_FIELDS), list(rows[0].keys()))
        self.assertEqual([0, 1, 2, 3, 4], [int(r["repeat"]) for r in rows])
        self.assertEqual(
            {"analysis": "times", "problem": "lorenz", "algorithm": "tsit5",
             "mode": "fixed", "transfers": "both", "setting_kind": "none",
             "setting": "nan", "n": "1024", "states": "3"},
            {k: rows[0][k] for k in SAMPLE_FIELDS[:-2]})

    def test_the_reported_minimum_is_the_minimum_of_the_timed_rows(self):
        with SampleLog(self.path) as log:
            best, _ = timed_min_ms(lambda: None, 8, log.sink(
                "times", "lorenz", "euler", "fixed", "none", 64, 3))
        timed = [float(r["ms"]) for r in self.rows() if int(r["repeat"])]
        self.assertEqual(8, len(timed))
        self.assertAlmostEqual(min(timed), best, places=6)

    def test_the_warm_up_is_logged_but_not_reduced(self):
        with SampleLog(self.path) as log:
            sink = log.sink("times", "lorenz", "euler", "fixed", "none", 64, 3)
            sink(0, 900.0)
            sink(1, 1.5)
        rows = self.rows()
        self.assertEqual(["900.000000", "1.500000"], [r["ms"] for r in rows])

    def test_a_wp_point_carries_its_setting(self):
        with SampleLog(self.path) as log:
            log.sink("wp", "lorenz", "tsit5", "adaptive", "d2h", 131072, 3,
                     "tol", 1e-8)(1, 2.5)
        row = self.rows()[0]
        self.assertEqual("tol", row["setting_kind"])
        self.assertEqual("1e-08", row["setting"])
        self.assertEqual(1e-8, float(row["setting"]))

    def test_appending_keeps_one_header_and_truncating_restarts(self):
        for _ in range(2):
            with SampleLog(self.path) as log:
                log.sink("times", "lorenz", "euler", "fixed", "none", 64,
                         3)(1, 1.0)
        with open(self.path) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(3, len(lines))
        self.assertEqual(",".join(SAMPLE_FIELDS), lines[0])

        with SampleLog(self.path, True) as log:
            log.sink("times", "lorenz", "euler", "fixed", "none", 64, 3)(1, 1.0)
        with open(self.path) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(2, len(lines))
        self.assertEqual(",".join(SAMPLE_FIELDS), lines[0])

    def test_two_appenders_on_one_path_write_one_header(self):
        # The states sweep runs one process per size, all appending here.
        first, second = SampleLog(self.path), SampleLog(self.path)
        try:
            first.sink("states", "lorenz96", "euler", "fixed", "none", 64,
                       4)(1, 1.0)
            second.sink("states", "lorenz96", "euler", "fixed", "none", 64,
                        8)(1, 2.0)
        finally:
            first.close()
            second.close()
        with open(self.path) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(3, len(lines))
        self.assertEqual([",".join(SAMPLE_FIELDS)],
                         [ln for ln in lines if ln.startswith("analysis,")])

    def test_rows_are_readable_before_the_log_is_closed(self):
        # A watchdog breach exits the process; unflushed rows would be lost.
        log = SampleLog(self.path)
        try:
            log.sink("times", "lorenz", "euler", "fixed", "none", 64, 3)(1, 1.0)
            self.assertEqual(1, len(self.rows()))
        finally:
            log.close()


if __name__ == "__main__":
    unittest.main()
