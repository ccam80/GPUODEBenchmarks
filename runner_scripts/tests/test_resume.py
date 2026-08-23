"""Continuation tests: cursor parsing, run-order skipping and on-disk coverage."""

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

PROBLEMS = problem_names()
ALGORITHMS = algorithm_names()


class EnvCase(unittest.TestCase):
    """Every test starts with a clean environment and cursor cache."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("BENCH_RESUME", None)
        os.environ.pop("BENCH_NO_OVERWRITE", None)
        os.environ.pop("BENCH_RESUME_FROM", None)
        resume._reset_cache()
        self.addCleanup(resume._reset_cache)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def outfile(self, rows=""):
        path = os.path.join(self.tmp, "times.txt")
        with open(path, "w") as handle:
            handle.write(rows)
        return path


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


class DiskSkipTests(EnvCase):
    def test_disabled_never_skips(self):
        path = self.outfile("8 1.0 2.0\n")
        self.assertFalse(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                           "fixed", 8, path))

    def test_recorded_row_skips(self):
        os.environ["BENCH_RESUME"] = "1"
        path = self.outfile("8 1.0 2.0\n32 nan nan\n")
        self.assertTrue(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                          "fixed", 8, path))
        # A NaN row is a recorded failure, not a gap.
        self.assertTrue(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                          "fixed", 32, path))
        self.assertFalse(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                           "fixed", 128, path))

    def test_tab_separated_rows_count(self):
        os.environ["BENCH_RESUME"] = "1"
        path = self.outfile("8\t1.0\t2.0\n")
        self.assertTrue(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                          "fixed", 8, path))

    def test_torn_row_is_not_recorded(self):
        os.environ["BENCH_RESUME"] = "1"
        path = self.outfile("8\n")
        self.assertFalse(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                           "fixed", 8, path))

    def test_missing_file_never_skips(self):
        os.environ["BENCH_RESUME"] = "1"
        path = os.path.join(self.tmp, "absent.txt")
        self.assertFalse(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                           "fixed", 8, path))


class NoOverwriteTests(EnvCase):
    def test_finite_row_skips(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        path = self.outfile("8 1.0 2.0\n")
        self.assertTrue(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                          "fixed", 8, path))

    def test_nan_row_reruns(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        path = self.outfile("8 1.0 2.0\n32 nan nan\n")
        self.assertFalse(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                           "fixed", 32, path))

    def test_absent_row_reruns(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        path = self.outfile("8 1.0 2.0\n")
        self.assertFalse(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                           "fixed", 128, path))

    def test_missing_file_never_skips(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        path = os.path.join(self.tmp, "absent.txt")
        self.assertFalse(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                           "fixed", 8, path))

    def test_torn_row_reruns(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        path = self.outfile("8\n")
        self.assertFalse(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                           "fixed", 8, path))

    def test_resume_still_skips_nan_when_both_set(self):
        os.environ["BENCH_RESUME"] = "1"
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        path = self.outfile("32 nan nan\n")
        self.assertTrue(resume.skip_point(PROBLEMS[0], ALGORITHMS[0],
                                          "fixed", 32, path))

    def test_wp_leg_with_nan_rows_reruns(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        count = resume.wp_settings_count(PROBLEMS[0], ALGORITHMS[0], "fixed")
        rows = "".join("{0} nan nan\n".format(i) for i in range(count))
        path = self.outfile(rows)
        self.assertFalse(resume.skip_wp_leg(PROBLEMS[0], ALGORITHMS[0],
                                            "fixed", path))

    def test_wp_leg_with_finite_rows_skips(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        count = resume.wp_settings_count(PROBLEMS[0], ALGORITHMS[0], "fixed")
        rows = "".join("{0} 1.5 2.5\n".format(i) for i in range(count))
        path = self.outfile(rows)
        self.assertTrue(resume.skip_wp_leg(PROBLEMS[0], ALGORITHMS[0],
                                           "fixed", path))


class PruneRerunsTests(EnvCase):
    def test_inactive_leaves_the_file_alone(self):
        path = self.outfile("8 nan nan\n32 1.0 2.0\n")
        resume.prune_reruns(path, [8])
        with open(path) as handle:
            self.assertEqual(handle.read(), "8 nan nan\n32 1.0 2.0\n")

    def test_drops_only_the_rerun_rows(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        path = self.outfile("8 1.0 2.0\n32 nan nan\n128 1.0 2.0\n")
        resume.prune_reruns(path, [32])
        with open(path) as handle:
            self.assertEqual(handle.read(), "8 1.0 2.0\n128 1.0 2.0\n")

    def test_torn_row_survives(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        path = self.outfile("8\n32 nan nan\n")
        resume.prune_reruns(path, [8, 32])
        with open(path) as handle:
            self.assertEqual(handle.read(), "8\n")

    def test_missing_file_is_a_noop(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        path = os.path.join(self.tmp, "absent.txt")
        resume.prune_reruns(path, [8])
        self.assertFalse(os.path.exists(path))


class WpLegTests(EnvCase):
    def _complete_rows(self, problem, algorithm, mode):
        count = resume.wp_settings_count(problem, algorithm, mode)
        return "".join("{0} nan nan\n".format(i) for i in range(count)), count

    def test_complete_leg_skips(self):
        os.environ["BENCH_RESUME"] = "1"
        rows, _ = self._complete_rows(PROBLEMS[0], ALGORITHMS[0], "fixed")
        path = self.outfile(rows)
        self.assertTrue(resume.skip_wp_leg(PROBLEMS[0], ALGORITHMS[0],
                                           "fixed", path))

    def test_partial_leg_reruns(self):
        os.environ["BENCH_RESUME"] = "1"
        rows, count = self._complete_rows(PROBLEMS[0], ALGORITHMS[0], "fixed")
        partial = "".join(rows.splitlines(True)[:count - 1])
        path = self.outfile(partial)
        self.assertFalse(resume.skip_wp_leg(PROBLEMS[0], ALGORITHMS[0],
                                            "fixed", path))

    def test_adaptive_expects_tols(self):
        from wp_common import TOLS
        self.assertEqual(
            resume.wp_settings_count(PROBLEMS[0], ALGORITHMS[0], "adaptive"),
            len(TOLS))

    def test_disabled_never_skips(self):
        rows, _ = self._complete_rows(PROBLEMS[0], ALGORITHMS[0], "fixed")
        path = self.outfile(rows)
        self.assertFalse(resume.skip_wp_leg(PROBLEMS[0], ALGORITHMS[0],
                                            "fixed", path))


class CliTests(EnvCase):
    def _cli(self, *argv):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            resume._cli(list(argv))
        return buffer.getvalue().strip()

    def test_point_run_and_skip(self):
        os.environ["BENCH_RESUME"] = "1"
        path = self.outfile("8 1.0 2.0\n")
        self.assertEqual(
            self._cli("point", PROBLEMS[0], ALGORITHMS[0], "fixed", "8",
                      path), "skip")
        self.assertEqual(
            self._cli("point", PROBLEMS[0], ALGORITHMS[0], "fixed", "32",
                      path), "run")

    def test_leg(self):
        os.environ["BENCH_RESUME"] = "1"
        count = resume.wp_settings_count(PROBLEMS[0], ALGORITHMS[0], "fixed")
        path = self.outfile(
            "".join("{0} nan nan\n".format(i) for i in range(count)))
        self.assertEqual(
            self._cli("leg", PROBLEMS[0], ALGORITHMS[0], "fixed", path),
            "skip")

    def test_prune_verb_drops_a_point(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        path = self.outfile("8 1.0 2.0\n32 nan nan\n")
        self._cli("prune", "32", path)
        with open(path) as handle:
            self.assertEqual(handle.read(), "8 1.0 2.0\n")

    def test_bad_usage_exits(self):
        with self.assertRaises(SystemExit):
            resume._cli(["point", "too", "few"])


class ActiveTests(EnvCase):
    def test_inactive_by_default(self):
        self.assertFalse(resume.active())

    def test_resume_env_activates(self):
        os.environ["BENCH_RESUME"] = "1"
        self.assertTrue(resume.active())

    def test_zero_is_off(self):
        os.environ["BENCH_RESUME"] = "0"
        self.assertFalse(resume.active())

    def test_no_overwrite_activates(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        self.assertTrue(resume.active())

    def test_cursor_activates(self):
        os.environ["BENCH_RESUME_FROM"] = PROBLEMS[0]
        self.assertTrue(resume.active())


if __name__ == "__main__":
    unittest.main()
