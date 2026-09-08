"""The result store: rows replace by identity, --floor keeps the lower time, status drives resume, and every attempt rides in samples_ms."""

import csv
import math
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import results  # noqa: E402
from protocol import N_WP, STATES_N, TIMING_TOL  # noqa: E402

NAN = float("nan")


class StoreCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("BENCH_FLOOR", None)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def leg(self, analysis="times", problem="lorenz", algorithm="tsit5",
            mode="fixed", package="cubie"):
        return results.Leg(package, "test_key", analysis, problem, algorithm,
                           mode, root=self.tmp)


class RecordTests(StoreCase):
    def test_a_times_point_is_two_rows_with_the_timing_setting(self):
        leg = self.leg()
        leg.record_times(8, 1.5, 0.5, 0.0, samples_both=[9.0, 2.0, 1.5],
                         samples_none=[3.0, 0.5, 0.6])
        rows = results.load(leg.path)
        self.assertEqual([r["transfers"] for r in rows], ["both", "none"])
        self.assertEqual(rows[0]["setting_kind"], "dt")
        self.assertAlmostEqual(float(rows[0]["setting"]), 2.0 ** -10)
        self.assertEqual(rows[0]["n"], "8")
        self.assertEqual(rows[0]["states"], "3")
        # Every attempt, warm-up first; the recorded time is the min after it.
        self.assertEqual(rows[0]["samples_ms"], "9;2;1.5")
        self.assertEqual(results.samples_of(rows[0]), [9.0, 2.0, 1.5])
        self.assertEqual(min(results.samples_of(rows[0])[1:]),
                         float(rows[0]["min_ms"]))
        self.assertEqual(rows[1]["min_ms"], "0.5")
        self.assertEqual(results.samples_of(rows[1]), [3.0, 0.5, 0.6])

    def test_a_row_without_attempts_has_an_empty_samples_column(self):
        leg = self.leg()
        leg.nan_times([8])
        row = results.load(leg.path)[0]
        self.assertEqual(row["samples_ms"], "")
        self.assertEqual(results.samples_of(row), [])
        with open(leg.path, newline="") as handle:
            self.assertEqual(next(csv.reader(handle)), list(results.FIELDS))

    def test_an_adaptive_leg_carries_the_timing_tolerance(self):
        leg = self.leg(mode="adaptive")
        leg.record_times(8, 1.0, 1.0, 0.0)
        row = results.load(leg.path)[0]
        self.assertEqual(row["setting_kind"], "tol")
        self.assertEqual(float(row["setting"]), TIMING_TOL)

    def test_recording_the_same_point_replaces_it(self):
        leg = self.leg()
        leg.record_times(8, 1.5, 0.5, 0.0)
        leg.record_times(8, 2.5, 0.7, 12.5)
        rows = results.load(leg.path)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["min_ms"] for r in rows], ["2.5", "0.7"])
        self.assertEqual(rows[0]["errored_pct"], "12.5")

    def test_floor_keeps_the_lower_time_and_nan_never_wins(self):
        leg = self.leg()
        leg.record_times(8, 1.5, 0.5, 0.0)
        os.environ["BENCH_FLOOR"] = "1"
        leg.record_times(8, 2.5, 0.4, 0.0)
        leg.record_times(8, NAN, NAN, 100.0)
        rows = results.load(leg.path)
        self.assertEqual([r["min_ms"] for r in rows], ["1.5", "0.4"])
        leg.record_times(32, NAN, NAN, 100.0)
        leg.record_times(32, 9.0, 8.0, 0.0)
        by_n = {(r["n"], r["transfers"]): r["min_ms"]
                for r in results.load(leg.path)}
        self.assertEqual(by_n[("32", "both")], "9")

    def test_wp_rows_and_states_rows_carry_their_own_columns(self):
        wp = self.leg("wp", mode="adaptive")
        wp.record_wp(1e-8, 3.0, 1e-4, 0.0, transfers="d2h")
        row = results.load(wp.path)[0]
        self.assertEqual((row["n"], row["setting_kind"], row["transfers"]),
                         (str(N_WP), "tol", "d2h"))
        self.assertEqual(float(row["setting"]), 1e-8)
        self.assertEqual(float(row["error"]), 1e-4)
        states = self.leg("states", problem="lorenz96")
        states.record_times(STATES_N, 4.0, 3.0, 0.0, build_s=2.5, states=16)
        row = [r for r in results.load(states.path)
               if r["analysis"] == "states"][0]
        self.assertEqual((row["n"], row["states"], row["build_s"]),
                         (str(STATES_N), "16", "2.5"))

    def test_status_and_clear(self):
        leg = self.leg()
        leg.record_times(8, 1.5, 0.5, 0.0)
        leg.nan_times([32])
        self.assertEqual(leg.status(8), "finite")
        self.assertEqual(leg.status(32), "nan")
        self.assertEqual(leg.status(128), "absent")
        wp = self.leg("wp")
        wp.record_wp(0.0625, 1.0, 0.1, 0.0)
        self.assertEqual(wp.status(N_WP, setting=0.0625 * (1 + 1e-10)),
                         "finite")
        # A list value matches any of its members, so a run can clear exactly its Ns.
        self.assertEqual(results.clear(leg.path, package="cubie",
                                       analysis="times", n=["32", "128"]), 2)
        self.assertEqual(results.clear(leg.path, package="cubie",
                                       analysis="times"), 2)
        self.assertEqual(len(results.load(leg.path)), 1)

    def test_cli_record_nan_and_status(self):
        with mock.patch.object(results, "data_root", lambda: self.tmp):
            self.assertEqual(results._cli(
                ["record", "cpp", "k", "times", "lorenz", "classical-rk4",
                 "fixed", "dt", "0.0009765625", "8", "3", "default", "both",
                 "min_ms=1.5", "errored_pct=0", "samples=9;2;1.5"]), 0)
            self.assertEqual(results._cli(
                ["nan", "cpp", "k", "states", "lorenz96", "classical-rk4",
                 "fixed", "16", "0.75"]), 0)
            rows = results.load(results.store_path("cpp", "k", self.tmp))
        self.assertEqual(rows[0]["samples_ms"], "9;2;1.5")
        states = [r for r in rows if r["analysis"] == "states"]
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0]["build_s"], "0.75")
        self.assertTrue(math.isnan(float(states[0]["min_ms"])))


if __name__ == "__main__":
    unittest.main()
