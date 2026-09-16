"""clocks.py: the sampler line relayed with a UTC stamp, the window rule over a log, the drift verdict under a lock, the conf table read and written by the calibrator, and configure refusing without a target."""

import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "calibrate"))

import clocks  # noqa: E402

T0 = datetime(2026, 9, 16, 3, 0, 0, tzinfo=timezone.utc)


def log_lines(samples):
    """The relayed CSV of (seconds after T0, sm, reasons) samples."""
    lines = [clocks.HEADER]
    for offset, sm, reasons in samples:
        stamp = (T0 + timedelta(seconds=offset)).strftime(clocks.UTC_STAMP)
        lines.append("{0},{1},10251,60,170,100,0x{2:016x}".format(stamp, sm, reasons))
    return "\n".join(lines) + "\n"


class SamplerLine(unittest.TestCase):
    def test_a_local_stamp_becomes_utc_and_the_fields_parse(self):
        local = datetime(2026, 8, 21, 8, 30, 8, 488000)
        line = "2026/08/21 08:30:08.488, 2775, 10251, 57, 170.78, 100, 0x0000000000000004"
        utc, sm, mem, temp, power, util, reasons = clocks.parse_sample(line)
        self.assertEqual(utc, local.astimezone(timezone.utc))
        self.assertEqual(utc.tzinfo, timezone.utc)
        self.assertEqual((sm, mem, temp, power, util, reasons), (2775, 10251, 57.0, 170.78, 100, 4))

    def test_the_header_and_diagnostics_are_skipped(self):
        self.assertIsNone(clocks.parse_sample("timestamp, clocks.current.sm [MHz], x, y, z, w, v"))
        self.assertIsNone(clocks.parse_sample("Unable to determine the device handle"))
        self.assertIsNone(clocks.parse_sample(""))

    def test_the_relay_writes_utc_lines_and_flushes_each(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        path = os.path.join(tmp, "run.csv")
        stream = iter(["timestamp, clocks.current.sm [MHz], a, b, c, d, e\n",
                       "2026/08/21 08:30:08.488, 2775, 10251, 57, 170.78, 100, 0x0000000000000001\n",
                       "garbage\n",
                       "2026/08/21 08:30:08.588, 2790, 10251, 57, 171.00, 100, 0x0000000000000000\n"])
        clocks._relay(stream, path)
        with open(path) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[0], clocks.HEADER)
        self.assertEqual(len(lines), 3)
        expected = datetime(2026, 8, 21, 8, 30, 8, 488000).astimezone(timezone.utc)
        self.assertEqual(lines[1].split(",")[0], expected.strftime(clocks.UTC_STAMP))
        self.assertEqual(lines[1].split(",")[1:], ["2775", "10251", "57", "170.78", "100", "0x0000000000000001"])
        samples = clocks.load_samples(path)
        self.assertEqual(samples["sm"], [2775, 2790])
        self.assertAlmostEqual(samples["t"][1] - samples["t"][0], 0.1, places=3)


class Windows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def log(self, samples):
        path = os.path.join(self.tmp, "clocks.csv")
        with open(path, "w") as handle:
            handle.write(log_lines(samples))
        return clocks.load_samples(path)

    def at(self, seconds):
        return (T0 + timedelta(seconds=seconds)).timestamp()

    def test_the_window_holds_its_samples_and_the_sm_statistics_are_over_busy_ones(self):
        samples = self.log([(0.0, 210, 1), (0.1, 2445, 0), (0.2, 2430, 0), (0.3, 2445, 4),
                            (0.4, 2460, 0), (0.5, 210, 1)])
        stats = clocks.window_stats(samples, self.at(0.1), self.at(0.4))
        self.assertEqual(stats, {"clock_sm_mhz": 2445.0, "clock_sm_min_mhz": 2430.0,
                                 "clock_throttled": 1})

    def test_a_window_between_samples_extends_to_the_nearest_on_each_side(self):
        samples = self.log([(0.0, 2400, 0), (0.1, 2500, 0), (0.2, 2600, 0)])
        stats = clocks.window_stats(samples, self.at(0.12), self.at(0.13))
        self.assertEqual((stats["clock_sm_mhz"], stats["clock_sm_min_mhz"]), (2550.0, 2500.0))
        stats = clocks.window_stats(samples, self.at(0.05), self.at(0.06))
        self.assertEqual((stats["clock_sm_mhz"], stats["clock_sm_min_mhz"]), (2450.0, 2400.0))

    def test_a_window_the_sampler_never_observed_is_not_annotated(self):
        # The sampler stopped at 0.1 s.
        samples = self.log([(0.0, 2400, 0), (0.1, 2500, 0)])
        self.assertIsNone(clocks.window_stats(samples, self.at(3600.0), self.at(3601.0)))
        self.assertIsNone(clocks.window_stats(samples, self.at(5.0), self.at(6.0)))
        self.assertIsNone(clocks.window_stats(samples, self.at(-6.0), self.at(-5.0)))
        # A window ending within the coverage gap of the last sample is covered, on or past that sample.
        stats = clocks.window_stats(samples, self.at(0.1), self.at(0.5))
        self.assertEqual((stats["clock_sm_mhz"], stats["clock_sm_min_mhz"]), (2450.0, 2400.0))
        stats = clocks.window_stats(samples, self.at(0.11), self.at(0.5))
        self.assertEqual((stats["clock_sm_mhz"], stats["clock_sm_min_mhz"]), (2500.0, 2500.0))
        self.assertIsNone(clocks.window_stats(samples, self.at(0.11), self.at(1.2)))
        # No sample inside: a neighbour within the gap is needed on each side.
        samples = self.log([(0.0, 2400, 0), (10.0, 2500, 0)])
        self.assertIsNone(clocks.window_stats(samples, self.at(4.0), self.at(5.0)))
        self.assertEqual(clocks.window_stats(samples, self.at(0.5), self.at(0.6))["clock_sm_mhz"], 2400.0)
        self.assertIsNone(clocks.window_stats(samples, self.at(0.5), self.at(1.1)))

    def test_a_sampler_that_stops_during_the_batch_leaves_it_unannotated(self):
        # Two samples at the start, then the sampler died: the batch's end is 600 s past the last one.
        samples = self.log([(0.0, 2400, 0), (0.04, 2400, 0)])
        self.assertIsNone(clocks.window_stats(samples, self.at(0.0), self.at(600.0)))
        self.assertIsNone(clocks.window_stats(samples, self.at(0.0), self.at(1.05)))
        stats = clocks.window_stats(samples, self.at(0.0), self.at(1.0))
        self.assertEqual((stats["clock_sm_mhz"], stats["clock_sm_min_mhz"], stats["clock_throttled"]),
                         (2400.0, 2400.0, 0))
        # A sampler that started late leaves the batch's start uncovered the same way.
        samples = self.log([(599.96, 2400, 0), (600.0, 2400, 0)])
        self.assertIsNone(clocks.window_stats(samples, self.at(0.0), self.at(600.0)))

    def test_a_gap_in_the_log_wider_than_the_tolerance_leaves_the_batch_unannotated(self):
        # Samples bracket the batch within the gap on both sides, with nothing observed between them.
        samples = self.log([(0.0, 2400, 0), (600.0, 2400, 0)])
        self.assertIsNone(clocks.window_stats(samples, self.at(0.5), self.at(599.5)))
        self.assertIsNone(clocks.window_stats(samples, self.at(0.0), self.at(600.0)))
        # A hole in an otherwise steady log leaves a batch spanning it unannotated; one beside it is covered.
        steady = [(t / 25.0, 2400, 0) for t in range(0, 50)] + [(t / 25.0, 2400, 0) for t in range(100, 150)]
        samples = self.log(steady)
        self.assertIsNone(clocks.window_stats(samples, self.at(1.5), self.at(4.5)))
        self.assertIsNone(clocks.window_stats(samples, self.at(1.0), self.at(5.0)))
        self.assertEqual(clocks.window_stats(samples, self.at(0.5), self.at(1.5))["clock_sm_mhz"], 2400.0)
        self.assertEqual(clocks.window_stats(samples, self.at(4.5), self.at(5.5))["clock_sm_mhz"], 2400.0)
        # A hole no wider than the tolerance is covered.
        samples = self.log([(0.0, 2400, 0), (1.0, 2500, 0), (2.0, 2600, 0)])
        self.assertEqual(clocks.window_stats(samples, self.at(0.0), self.at(2.0))["clock_sm_mhz"], 2500.0)

    def test_idle_samples_alone_give_nan_clocks_and_an_empty_log_gives_none(self):
        samples = self.log([(0.0, 210, 1), (0.1, 210, 1)])
        stats = clocks.window_stats(samples, self.at(0.0), self.at(0.1))
        self.assertNotEqual(stats["clock_sm_mhz"], stats["clock_sm_mhz"])
        self.assertNotEqual(stats["clock_sm_min_mhz"], stats["clock_sm_min_mhz"])
        self.assertEqual(stats["clock_throttled"], 0)
        self.assertIsNone(clocks.window_stats(self.log([]), self.at(0.0), self.at(1.0)))
        self.assertEqual(clocks.load_samples(os.path.join(self.tmp, "absent.csv"))["t"], [])

    def test_drift_is_throttling_or_a_busy_sample_below_the_lock_less_tolerance(self):
        self.assertFalse(clocks.drifted({"clock_sm_mhz": 2445.0, "clock_sm_min_mhz": 2430.0,
                                         "clock_throttled": 0}, 2445))
        self.assertTrue(clocks.drifted({"clock_sm_mhz": 2445.0, "clock_sm_min_mhz": 2429.0,
                                        "clock_throttled": 0}, 2445))
        self.assertTrue(clocks.drifted({"clock_sm_mhz": 2445.0, "clock_sm_min_mhz": 2445.0,
                                        "clock_throttled": 1}, 2445))
        self.assertFalse(clocks.drifted({"clock_sm_mhz": 2000.0, "clock_sm_min_mhz": 1900.0,
                                         "clock_throttled": 3}, 0))
        self.assertFalse(clocks.drifted({"clock_sm_mhz": float("nan"), "clock_sm_min_mhz": float("nan"),
                                         "clock_throttled": 0}, 2445))
        self.assertFalse(clocks.drifted(None, 2445))


class ConfTable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.conf = os.path.join(self.tmp, "gpu_clocks.conf")
        with open(self.conf, "w") as handle:
            handle.write("# header comment\n\n# Measured 2026-07-30, driver 595.84: flat plateau.\n"
                         "# This card refuses -lmc.\nRTX-2060-SUPER 1470 6801\n")

    def test_the_slug_is_the_gpu_half_of_the_key(self):
        self.assertEqual(clocks.gpu_slug("windows_RTX-4070-SUPER"), "RTX-4070-SUPER")
        self.assertEqual(clocks.gpu_slug("linux_RTX-2060-SUPER"), "RTX-2060-SUPER")

    def test_a_row_reads_and_a_missing_one_is_none(self):
        self.assertEqual(clocks.conf_row("RTX-2060-SUPER", self.conf), ("1470", "6801"))
        self.assertEqual(clocks.conf_row("RTX-4070-SUPER", self.conf), (None, None))
        self.assertEqual(clocks.conf_row("RTX-4070-SUPER", os.path.join(self.tmp, "none")), (None, None))

    def test_the_calibrator_appends_a_new_row_with_its_note(self):
        row = clocks.write_conf_row("RTX-4070-SUPER", 2310, 10251, "Measured 2026-09-16: throttled", self.conf)
        self.assertEqual(row, "RTX-4070-SUPER 2310 10251")
        with open(self.conf) as handle:
            text = handle.read()
        self.assertTrue(text.endswith("RTX-2060-SUPER 1470 6801\n\n# Measured 2026-09-16: throttled\n"
                                      "RTX-4070-SUPER 2310 10251\n"))
        self.assertTrue(text.startswith("# header comment\n"))
        self.assertEqual(clocks.conf_row("RTX-4070-SUPER", self.conf), ("2310", "10251"))

    def test_the_calibrator_replaces_an_earlier_row_and_its_own_note(self):
        clocks.write_conf_row("RTX-2060-SUPER", 1440, None, "Measured 2026-09-16: varied", self.conf)
        with open(self.conf) as handle:
            text = handle.read()
        self.assertEqual(text.count("RTX-2060-SUPER"), 1)
        self.assertEqual(clocks.conf_row("RTX-2060-SUPER", self.conf), ("1440", None))
        # The comment block above the old row went with it; the header, set apart by a blank line, stays.
        self.assertNotIn("# Measured 2026-07-30", text)
        self.assertNotIn("# This card refuses -lmc.", text)
        self.assertEqual(text, "# header comment\n\n# Measured 2026-09-16: varied\nRTX-2060-SUPER 1440\n")
        clocks.write_conf_row("RTX-2060-SUPER", 1450, 6801, "Measured 2026-09-17: flat", self.conf)
        with open(self.conf) as handle:
            text = handle.read()
        self.assertNotIn("2026-09-16", text)
        self.assertEqual(text, "# header comment\n\n# Measured 2026-09-17: flat\nRTX-2060-SUPER 1450 6801\n")
        empty = os.path.join(self.tmp, "fresh.conf")
        clocks.write_conf_row("RTX-4070-SUPER", 1980, None, "", empty)
        with open(empty) as handle:
            self.assertEqual(handle.read(), "RTX-4070-SUPER 1980\n")

    def test_configure_refuses_without_a_target(self):
        with self.assertRaises(clocks.ClockError) as caught:
            clocks.configure("windows_RTX-4070-SUPER", conf=self.conf)
        self.assertIn("RTX-4070-SUPER", str(caught.exception))
        self.assertIn("--lock-clocks SM[,MEM]", str(caught.exception))
        # There is no unlocked run to point at.
        self.assertNotIn("--no-lock-clocks", str(caught.exception))
        with self.assertRaises(clocks.ClockError):
            clocks.configure("windows_RTX-4070-SUPER", conf=os.path.join(self.tmp, "none"))

    def test_configure_refuses_a_clock_the_gpu_does_not_offer(self):
        original = clocks.supported
        clocks.supported = lambda kind, mhz: (kind, str(mhz)) in (("gr", "1470"), ("mem", "6801"))
        self.addCleanup(setattr, clocks, "supported", original)
        self.assertEqual(clocks.configure("linux_RTX-2060-SUPER", conf=self.conf), ("1470", "6801"))
        self.assertEqual(clocks.configure("linux_RTX-2060-SUPER", "1470", conf=self.conf), ("1470", None))
        with self.assertRaises(clocks.ClockError):
            clocks.configure("linux_RTX-2060-SUPER", "1500", conf=self.conf)
        with self.assertRaises(clocks.ClockError):
            clocks.configure("linux_RTX-2060-SUPER", "1470,7000", conf=self.conf)

    def test_the_guard_refuses_to_lock_without_elevation(self):
        original = clocks.is_admin
        clocks.is_admin = lambda: False
        self.addCleanup(setattr, clocks, "is_admin", original)
        guard = clocks.ClockGuard("1470", None)
        with self.assertRaises(clocks.ClockError) as caught:
            guard.lock()
        self.assertIn("elevated", str(caught.exception))
        self.assertFalse(guard.locked)
        self.assertEqual(guard.status(), "unlocked")
        self.assertFalse(clocks.ClockGuard(None, None).lock())

    def test_a_sampler_that_dies_at_once_is_nonfatal(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        guard = clocks.ClockGuard("1470", None)
        original = clocks.subprocess.Popen
        clocks.subprocess.Popen = mock.Mock(side_effect=OSError("no nvidia-smi"))
        self.addCleanup(setattr, clocks.subprocess, "Popen", original)
        self.assertFalse(guard.start_monitor(os.path.join(tmp, "run.csv"), sample_ms=40))
        self.assertIsNone(guard.monitor)
        clocks.subprocess.Popen = original
        guard = clocks.ClockGuard("1470", None)
        with mock.patch.object(clocks.subprocess, "Popen",
                               return_value=mock.Mock(stdout=iter([]), poll=lambda: 1, kill=lambda: None,
                                                      wait=lambda *_: 1)):
            self.assertFalse(guard.start_monitor(os.path.join(tmp, "run.csv")))
        self.assertIsNone(guard.monitor)


class CalibratorTable(unittest.TestCase):
    def test_the_calibrator_burns_thirty_minutes_and_judges_the_second_half(self):
        import calibrate_clocks
        self.assertEqual(calibrate_clocks.MINUTES, 30)
        self.assertEqual(calibrate_clocks.WARMUP_S, 900)
        self.assertIs(calibrate_clocks.write_conf_row, clocks.write_conf_row)
        note = calibrate_clocks.conf_note(
            {"n": 900, "sm_min": 2445, "sm_max": 2490, "sm_mode": 2445, "mem": 10251, "temp": 60.0,
             "power": 172.9, "throttles": ["SwPowerCap"]}, "5% headroom applied")
        self.assertTrue(note.startswith("Measured 20"))
        self.assertIn("SM 2445-2490 MHz", note)
        self.assertIn("5% headroom applied", note)
        self.assertNotIn("\n", note)


if __name__ == "__main__":
    unittest.main()
