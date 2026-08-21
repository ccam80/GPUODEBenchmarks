"""Per-repeat timing log: file layout, row format and the min it reduces to."""

import csv
import os
import sys
import tempfile
import unittest
import unittest.mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from wp_common import (  # noqa: E402
    SAMPLE_FIELDS, append_samples, reset_samples, sample_point,
    samples_outfile, timed_min_ms,
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


def timed_on_a_fake_clock(durations, repeats):
    """Run timed_min_ms with each attempt taking the given seconds."""
    import timeit
    ticks = []
    now = 0.0
    for seconds in durations:
        ticks.extend([now, now + seconds])
        now += seconds
    with unittest.mock.patch.object(timeit, "default_timer",
                                    side_effect=ticks):
        return timed_min_ms(lambda: None, repeats)


class TimingTests(unittest.TestCase):
    def test_every_attempt_is_returned_and_the_min_skips_the_warm_up(self):
        best, _, samples = timed_on_a_fake_clock([0.030, 0.002, 0.003], 2)
        self.assertEqual(3, len(samples))
        for expected, actual in zip([30.0, 2.0, 3.0], samples):
            self.assertAlmostEqual(expected, actual, places=9)
        self.assertAlmostEqual(2.0, best, places=9)

    def test_a_breach_reports_no_time_but_keeps_the_attempts_it_made(self):
        from wp_common import WATCHDOG_SECONDS
        best, _, samples = timed_on_a_fake_clock(
            [0.030, 0.002, WATCHDOG_SECONDS + 1.0, 0.003], 3)
        self.assertIsNone(best)
        self.assertEqual(3, len(samples))
        self.assertAlmostEqual((WATCHDOG_SECONDS + 1.0) * 1000.0, samples[-1],
                               places=6)


class WriterTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".csv")
        os.close(handle)
        os.remove(self.path)
        self.point = sample_point("times", "lorenz", "tsit5", "fixed", 1024, 3)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def rows(self):
        with open(self.path) as handle:
            return list(csv.DictReader(handle))

    def test_a_leg_writes_one_row_per_attempt(self):
        append_samples(self.path, self.point, "both", [900.0, 1.5, 1.25])
        rows = self.rows()
        self.assertEqual(3, len(rows))
        self.assertEqual(list(SAMPLE_FIELDS), list(rows[0].keys()))
        self.assertEqual(["0", "1", "2"], [r["repeat"] for r in rows])
        self.assertEqual(["900.000000", "1.500000", "1.250000"],
                         [r["ms"] for r in rows])
        self.assertEqual(
            {"analysis": "times", "problem": "lorenz", "algorithm": "tsit5",
             "mode": "fixed", "transfers": "both", "setting_kind": "none",
             "setting": "nan", "n": "1024", "states": "3"},
            {k: rows[0][k] for k in SAMPLE_FIELDS[:-2]})

    def test_the_two_legs_of_a_point_share_its_identity(self):
        append_samples(self.path, self.point, "both", [2.0])
        append_samples(self.path, self.point, "none", [1.0])
        self.assertEqual(["both", "none"], [r["transfers"] for r in self.rows()])

    def test_a_wp_point_carries_its_setting(self):
        point = sample_point("wp", "lorenz", "tsit5", "adaptive", 131072, 3,
                             "tol", 1e-8)
        append_samples(self.path, point, "d2h", [2.5])
        row = self.rows()[0]
        self.assertEqual("tol", row["setting_kind"])
        self.assertEqual("1e-08", row["setting"])
        self.assertEqual(1e-8, float(row["setting"]))

    def test_appending_keeps_one_header_and_reset_restarts(self):
        append_samples(self.path, self.point, "both", [1.0])
        append_samples(self.path, self.point, "both", [2.0])
        with open(self.path) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(3, len(lines))
        self.assertEqual(",".join(SAMPLE_FIELDS), lines[0])

        reset_samples(self.path)
        self.assertFalse(os.path.exists(self.path))
        append_samples(self.path, self.point, "both", [1.0])
        with open(self.path) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(2, len(lines))
        self.assertEqual(",".join(SAMPLE_FIELDS), lines[0])

    def test_reset_on_a_missing_log_is_not_an_error(self):
        reset_samples(self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_the_logged_minimum_reproduces_the_reduced_time(self):
        best, _, samples = timed_on_a_fake_clock(
            [0.030, 0.004, 0.002, 0.003], 3)
        append_samples(self.path, self.point, "both", samples)
        timed = [float(r["ms"]) for r in self.rows() if int(r["repeat"])]
        self.assertAlmostEqual(min(timed), best, places=6)


if __name__ == "__main__":
    unittest.main()
