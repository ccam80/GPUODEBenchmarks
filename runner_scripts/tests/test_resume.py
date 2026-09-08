"""Continuation tests: cursor parsing, run-order skipping and store coverage."""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import resume  # noqa: E402
from algorithms import algorithm_names  # noqa: E402
from problems import problem_names  # noqa: E402
from protocol import N_WP, STATES_N, TOLS  # noqa: E402
from results import Leg  # noqa: E402

PROBLEMS = problem_names()
ALGORITHMS = algorithm_names()
NAN = float("nan")


class EnvCase(unittest.TestCase):
    """Every test starts with a clean environment and cursor cache."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("BENCH_RESUME", None)
        os.environ.pop("BENCH_NO_OVERWRITE", None)
        os.environ.pop("BENCH_RESUME_FROM", None)
        os.environ.pop("BENCH_FLOOR", None)
        resume._reset_cache()
        self.addCleanup(resume._reset_cache)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def leg(self, analysis="times", problem=None, algorithm=None,
            mode="fixed"):
        return Leg("cubie", "test_key", analysis, problem or PROBLEMS[0],
                   algorithm or ALGORITHMS[0], mode, root=self.tmp)


class ParseCursorTests(EnvCase):
    def test_problem_only(self):
        cursor = resume.parse_cursor(PROBLEMS[1])
        self.assertEqual(cursor["problem"], 1)
        self.assertIsNone(cursor["algorithm"])
        self.assertIsNone(cursor["mode"])
        self.assertIsNone(cursor["n"])

    def test_problem_and_n(self):
        cursor = resume.parse_cursor(PROBLEMS[0] + ":131072")
        self.assertEqual(cursor["problem"], 0)
        self.assertIsNone(cursor["algorithm"])
        self.assertEqual(cursor["n"], 131072)

    def test_full_form(self):
        spec = "{0}:{1}:adaptive:512".format(PROBLEMS[0], ALGORITHMS[1])
        cursor = resume.parse_cursor(spec)
        self.assertEqual(cursor["algorithm"], 1)
        self.assertEqual(cursor["mode"], 1)
        self.assertEqual(cursor["n"], 512)

    def test_algorithm_defaults_to_fixed(self):
        cursor = resume.parse_cursor(
            "{0}:{1}".format(PROBLEMS[0], ALGORITHMS[0]))
        self.assertEqual(cursor["mode"], 0)

    def test_unknown_problem_exits(self):
        with self.assertRaises(SystemExit):
            resume.parse_cursor("not-a-problem")

    def test_unknown_algorithm_exits(self):
        with self.assertRaises(SystemExit):
            resume.parse_cursor(PROBLEMS[0] + ":not-an-algorithm")

    def test_mode_before_algorithm_exits(self):
        with self.assertRaises(SystemExit):
            resume.parse_cursor(PROBLEMS[0] + ":fixed")

    def test_token_after_n_exits(self):
        with self.assertRaises(SystemExit):
            resume.parse_cursor(
                "{0}:8:{1}".format(PROBLEMS[0], ALGORITHMS[0]))


class CursorSkipTests(EnvCase):
    def test_no_cursor_skips_nothing(self):
        self.assertFalse(resume.cursor_skips(PROBLEMS[0], ALGORITHMS[0],
                                             "fixed", 8))

    def test_earlier_problem_skipped_later_runs(self):
        os.environ["BENCH_RESUME_FROM"] = PROBLEMS[1]
        self.assertTrue(resume.cursor_skips(PROBLEMS[0], ALGORITHMS[0],
                                            "fixed", 8))
        self.assertFalse(resume.cursor_skips(PROBLEMS[1], ALGORITHMS[0],
                                             "fixed", 8))
        self.assertFalse(resume.cursor_skips(PROBLEMS[2], ALGORITHMS[0],
                                             "fixed", 8))

    def test_problem_n_floors_every_leg(self):
        os.environ["BENCH_RESUME_FROM"] = PROBLEMS[1] + ":512"
        for algorithm in (ALGORITHMS[0], ALGORITHMS[-1]):
            self.assertTrue(resume.cursor_skips(PROBLEMS[1], algorithm,
                                                "adaptive", 128))
            self.assertFalse(resume.cursor_skips(PROBLEMS[1], algorithm,
                                                 "adaptive", 512))

    def test_leg_cursor(self):
        spec = "{0}:{1}:adaptive:512".format(PROBLEMS[1], ALGORITHMS[1])
        os.environ["BENCH_RESUME_FROM"] = spec
        # Earlier legs of the problem are skipped entirely.
        self.assertTrue(resume.cursor_skips(PROBLEMS[1], ALGORITHMS[1],
                                            "fixed", 1 << 30))
        self.assertTrue(resume.cursor_skips(PROBLEMS[1], ALGORITHMS[0],
                                            "adaptive", 1 << 30))
        # The named leg starts at N.
        self.assertTrue(resume.cursor_skips(PROBLEMS[1], ALGORITHMS[1],
                                            "adaptive", 128))
        self.assertFalse(resume.cursor_skips(PROBLEMS[1], ALGORITHMS[1],
                                             "adaptive", 512))
        # Later legs run in full.
        self.assertFalse(resume.cursor_skips(PROBLEMS[1], ALGORITHMS[2],
                                             "fixed", 8))
        # wp legs (no N) at the cursor's leg still run.
        self.assertFalse(resume.cursor_skips(PROBLEMS[1], ALGORITHMS[1],
                                             "adaptive"))
        self.assertTrue(resume.cursor_skips(PROBLEMS[1], ALGORITHMS[1],
                                            "fixed"))


class StoreSkipTests(EnvCase):
    def test_disabled_never_skips(self):
        leg = self.leg()
        leg.record_times(8, 1.0, 2.0, 0.0)
        self.assertFalse(resume.skip_point(leg, 8))

    def test_resume_skips_recorded_and_nan_rows(self):
        os.environ["BENCH_RESUME"] = "1"
        leg = self.leg()
        leg.record_times(8, 1.0, 2.0, 0.0)
        leg.nan_times([32])
        self.assertTrue(resume.skip_point(leg, 8))
        self.assertTrue(resume.skip_point(leg, 32))
        self.assertFalse(resume.skip_point(leg, 128))

    def test_no_overwrite_retries_nan_and_absent_rows(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        leg = self.leg()
        leg.record_times(8, 1.0, 2.0, 0.0)
        leg.nan_times([32])
        self.assertTrue(resume.skip_point(leg, 8))
        self.assertFalse(resume.skip_point(leg, 32))
        self.assertFalse(resume.skip_point(leg, 128))

    def test_resume_still_skips_nan_when_both_set(self):
        os.environ["BENCH_RESUME"] = "1"
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        leg = self.leg()
        leg.nan_times([32])
        self.assertTrue(resume.skip_point(leg, 32))

    def test_states_points_are_keyed_by_state_count(self):
        os.environ["BENCH_RESUME"] = "1"
        leg = self.leg("states", problem="lorenz96")
        leg.record_times(STATES_N, 1.0, 2.0, 0.0, build_s=0.5, states=16)
        self.assertTrue(resume.skip_point(leg, STATES_N, 16))
        self.assertFalse(resume.skip_point(leg, STATES_N, 32))

    def test_cursor_applies_to_the_states_count(self):
        os.environ["BENCH_RESUME_FROM"] = "lorenz96:32"
        leg = self.leg("states", problem="lorenz96")
        self.assertTrue(resume.skip_point(leg, STATES_N, 16))
        self.assertFalse(resume.skip_point(leg, STATES_N, 32))

    def test_a_store_of_another_leg_never_skips(self):
        os.environ["BENCH_RESUME"] = "1"
        self.leg(algorithm=ALGORITHMS[1]).record_times(8, 1.0, 2.0, 0.0)
        self.assertFalse(resume.skip_point(self.leg(), 8))


class WpLegTests(EnvCase):
    def wp(self):
        return self.leg("wp", mode="adaptive")

    def test_complete_leg_skips(self):
        os.environ["BENCH_RESUME"] = "1"
        leg = self.wp()
        for tol in TOLS:
            leg.record_wp(tol, 1.0, 0.1, 0.0)
        self.assertTrue(resume.skip_wp_leg(leg, TOLS))

    def test_partial_leg_reruns(self):
        os.environ["BENCH_RESUME"] = "1"
        leg = self.wp()
        for tol in TOLS[:-1]:
            leg.record_wp(tol, 1.0, 0.1, 0.0)
        self.assertFalse(resume.skip_wp_leg(leg, TOLS))

    def test_no_overwrite_reruns_a_leg_with_nan_rows(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        leg = self.wp()
        for tol in TOLS:
            leg.record_wp(tol, 1.0, 0.1, 0.0)
        leg.nan_wp([TOLS[-1]])
        self.assertFalse(resume.skip_wp_leg(leg, TOLS))
        os.environ["BENCH_RESUME"] = "1"
        self.assertTrue(resume.skip_wp_leg(leg, TOLS))

    def test_disabled_never_skips(self):
        leg = self.wp()
        for tol in TOLS:
            leg.record_wp(tol, 1.0, 0.1, 0.0)
        self.assertFalse(resume.skip_wp_leg(leg, TOLS))

    def test_status_reads_n_wp(self):
        leg = self.wp()
        leg.record_wp(TOLS[0], 1.0, 0.1, 0.0)
        self.assertEqual(leg.status(N_WP, setting=TOLS[0]), "finite")


class CliTests(EnvCase):
    def run_cli(self, argv):
        import io
        from contextlib import redirect_stdout
        import results
        out = io.StringIO()
        with mock.patch.object(results, "data_root", lambda: self.tmp), \
                redirect_stdout(out):
            resume._cli(argv)
        return out.getvalue().strip()

    def test_point_run_and_skip(self):
        os.environ["BENCH_RESUME"] = "1"
        self.leg().record_times(8, 1.0, 2.0, 0.0)
        base = ["point", "cubie", "test_key", "times", PROBLEMS[0],
                ALGORITHMS[0], "fixed"]
        self.assertEqual(self.run_cli(base + ["8"]), "skip")
        self.assertEqual(self.run_cli(base + ["32"]), "run")

    def test_leg(self):
        os.environ["BENCH_RESUME"] = "1"
        leg = self.leg("wp", mode="adaptive")
        argv = ["leg", "cubie", "test_key", PROBLEMS[0], ALGORITHMS[0],
                "adaptive"]
        self.assertEqual(self.run_cli(argv), "run")
        for tol in TOLS:
            leg.record_wp(tol, 1.0, 0.1, 0.0)
        self.assertEqual(self.run_cli(argv), "skip")

    def test_bad_usage_exits(self):
        with self.assertRaises(SystemExit):
            resume._cli(["point", "only"])


class ActiveTests(EnvCase):
    def test_inactive_by_default(self):
        self.assertFalse(resume.active())

    def test_resume_env_activates(self):
        os.environ["BENCH_RESUME"] = "1"
        self.assertTrue(resume.active())
        os.environ.pop("BENCH_RESUME")
        os.environ["BENCH_RESUME_FROM"] = PROBLEMS[0]
        resume._reset_cache()
        self.assertTrue(resume.active())


if __name__ == "__main__":
    unittest.main()
