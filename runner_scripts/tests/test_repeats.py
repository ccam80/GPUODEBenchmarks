"""Duration-scheduled repeat counts: bounds table, spread test, timed_min_ms."""

import os
import sys
import timeit
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import wp_common  # noqa: E402
from wp_common import repeat_bounds, repeats_done, timed_min_ms  # noqa: E402


class FakeRun:
    """run() whose k-th call appears to take durations[k] seconds."""

    def __init__(self, durations_s):
        self.durations = list(durations_s)
        self.now = 0.0
        self.calls = 0

    def __call__(self):
        self.now += self.durations[self.calls]
        self.calls += 1
        return self.calls

    def timer(self):
        return self.now


class TimerCase(unittest.TestCase):
    """timeit.default_timer is patched to the fake run's clock."""

    def run_timed(self, durations_s, repeats=20):
        run = FakeRun(durations_s)
        original = timeit.default_timer
        timeit.default_timer = run.timer
        try:
            return timed_min_ms(run, repeats)
        finally:
            timeit.default_timer = original


class TestRepeatBounds(unittest.TestCase):
    def test_table(self):
        self.assertEqual(repeat_bounds(0.05, 20), (20, 20))
        self.assertEqual(repeat_bounds(0.5, 20), (10, 10))
        self.assertEqual(repeat_bounds(2.0, 20), (10, 10))
        self.assertEqual(repeat_bounds(4.0, 20), (5, 10))
        self.assertEqual(repeat_bounds(7.0, 20), (3, 10))
        self.assertEqual(repeat_bounds(60.0, 20), (3, 10))

    def test_boundaries_round_up(self):
        self.assertEqual(repeat_bounds(0.1, 20), (10, 10))
        self.assertEqual(repeat_bounds(3.0, 20), (5, 10))
        self.assertEqual(repeat_bounds(5.0, 20), (3, 10))

    def test_cap_bounds_both(self):
        self.assertEqual(repeat_bounds(0.05, 5), (5, 5))
        self.assertEqual(repeat_bounds(4.0, 7), (5, 7))


class TestRepeatsDone(unittest.TestCase):
    def test_floor_not_reached(self):
        self.assertFalse(repeats_done([4.0, 4.0], 5, 10))

    def test_tight_spread_stops_at_floor(self):
        self.assertTrue(repeats_done([4.0, 4.01, 4.0, 4.02, 4.0], 5, 10))

    def test_loose_spread_extends(self):
        self.assertFalse(repeats_done([4.0, 5.0, 5.0, 5.0, 5.0], 5, 10))

    def test_ceiling_always_stops(self):
        self.assertTrue(repeats_done([4.0, 5.0] * 5, 5, 10))


class TestTimedMinMs(TimerCase):
    def test_fast_leg_keeps_twenty_repeats(self):
        best, result, samples = self.run_timed([0.01] * 21)
        self.assertEqual(len(samples), 21)
        self.assertAlmostEqual(best, 10.0)
        self.assertEqual(result, 21)

    def test_settled_slow_leg_stops_at_floor(self):
        best, _, samples = self.run_timed([4.0] * 11)
        self.assertEqual(len(samples), 6)      # warm-up + floor of 5
        self.assertAlmostEqual(best, 4000.0)

    def test_unsettled_slow_leg_runs_to_ceiling(self):
        best, _, samples = self.run_timed([4.0, 4.0] + [5.0] * 9)
        self.assertEqual(len(samples), 11)     # warm-up + ceiling of 10
        self.assertAlmostEqual(best, 4000.0)

    def test_schedule_comes_from_first_timed_run_not_warmup(self):
        # A slow warm-up (the compile) must not shrink a fast leg's repeats.
        best, _, samples = self.run_timed([10.0] + [0.01] * 20)
        self.assertEqual(len(samples), 21)
        self.assertAlmostEqual(best, 10.0)

    def test_cap_limits_the_schedule(self):
        best, _, samples = self.run_timed([0.01] * 6, repeats=5)
        self.assertEqual(len(samples), 6)
        self.assertAlmostEqual(best, 10.0)

    def test_breach_returns_none_with_samples(self):
        cap = wp_common.WATCHDOG_SECONDS
        best, _, samples = self.run_timed([cap + 1.0])
        self.assertIsNone(best)
        self.assertEqual(len(samples), 1)


if __name__ == "__main__":
    unittest.main()
