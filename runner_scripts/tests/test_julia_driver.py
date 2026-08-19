"""States-driver cancellation and backfill tests with subprocess.Popen faked."""

import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "runner_scripts", "gpu"))

import julia_driver  # noqa: E402


class FakeProc(object):
    """Scripted bench process; 'hang' behavior never returns."""

    def __init__(self, nstates, algorithm, outfiles, behavior, ticks=1):
        self.nstates = nstates
        self.algorithm = algorithm
        self.outfiles = outfiles
        self.behavior = behavior
        self.ticks = ticks
        self.killed = False
        self._code = None

    def _write_rows(self, value):
        for (mode, alg), path in self.outfiles.items():
            if alg != self.algorithm:
                continue
            with open(path, "a") as handle:
                handle.write("{0} {1} {1} 1.0\n".format(self.nstates, value))

    def poll(self):
        if self.killed:
            return self._code
        if self._code is not None:
            return self._code
        if self.behavior == "hang":
            return None
        self.ticks -= 1
        if self.ticks > 0:
            return None
        if self.behavior == "ok":
            self._write_rows("12.5")
        elif self.behavior == "launch_failure":
            # The bench catches the launch error and records NaN rows.
            self._write_rows("NaN")
        elif self.behavior == "torn":
            # Killed mid-write: the size made it out, the timings did not.
            for (mode, alg), path in self.outfiles.items():
                if alg == self.algorithm:
                    with open(path, "a") as handle:
                        handle.write("{0}".format(self.nstates))
        self._code = 0
        return self._code

    def kill(self):
        self.killed = True
        self._code = -9

    def wait(self):
        return self._code


class DriverHarness(object):
    """Patches julia_driver so run_states drives FakeProcs into a tmp dir."""

    def __init__(self, case, behaviors, grid, algorithms=("tsit5",)):
        self.tmp = tempfile.mkdtemp(prefix="jd_test_")
        case.addCleanup(self._cleanup)
        self.behaviors = behaviors
        self.spawned = []
        self.outfiles = {}
        legs = [(mode, algorithm) for algorithm in algorithms
                for mode in ("fixed", "adaptive")]
        for leg in legs:
            self.outfiles[leg] = os.path.join(
                self.tmp, "states_{0}_{1}.txt".format(*leg))

        def fake_popen(cmd, cwd=None, env=None):
            spec = cmd[3]  # ["julia", "--project=.", BENCH, spec, algorithm]
            nstates = int(spec.split(":")[1])
            algorithm = cmd[4]
            behavior = self.behaviors.get((nstates, algorithm), "ok")
            proc = FakeProc(nstates, algorithm, self.outfiles, behavior)
            self.spawned.append((nstates, algorithm))
            marker = env.get("BENCH_STATES_MARKER", "")
            if behavior != "hang" and marker:
                open(marker, "w").close()
            return proc

        patches = [
            mock.patch.object(julia_driver.subprocess, "Popen", fake_popen),
            mock.patch.object(julia_driver.time, "sleep", lambda _s: None),
            mock.patch.object(julia_driver, "resolve_states_grid",
                              lambda token: list(grid)),
            mock.patch.object(julia_driver, "resolve_algorithms",
                              lambda request, fw: list(algorithms)),
            mock.patch.object(
                julia_driver, "supported_for",
                lambda fw, mode: tuple(algorithms)),
            mock.patch.object(julia_driver, "dataset_key", lambda: "test"),
            mock.patch.object(
                julia_driver, "states_outfile",
                lambda fdir, prefix, mode, algorithm, key:
                self.outfiles[(mode, algorithm)]),
        ]
        for patch in patches:
            patch.start()
            case.addCleanup(patch.stop)

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rows(self, mode, algorithm):
        with open(self.outfiles[(mode, algorithm)]) as handle:
            return [line.split() for line in handle if line.strip()]


class StatesDriverTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("BENCH_STATES_BUDGET", None)
        os.environ["BENCH_JULIA_JOBS"] = "2"
        self.addCleanup(os.environ.pop, "BENCH_JULIA_JOBS", None)

    def test_all_sizes_succeed(self):
        harness = DriverHarness(self, {}, grid=(4, 8, 16))
        self.assertEqual(julia_driver.run_states(["tsit5"]), 0)
        rows = harness.rows("fixed", "tsit5")
        self.assertEqual([r[0] for r in rows], ["4", "8", "16"])
        self.assertTrue(all(r[1] == "12.5" for r in rows))

    def test_launch_failure_cancels_larger_sizes(self):
        harness = DriverHarness(
            self, {(8, "tsit5"): "launch_failure"}, grid=(4, 8, 16, 32))
        self.assertEqual(julia_driver.run_states(["tsit5"]), 0)
        # 4 succeeded; 8 failed; 16/32 cancelled and NaN-backfilled.
        rows = harness.rows("fixed", "tsit5")
        self.assertEqual([r[0] for r in rows], ["4", "8", "16", "32"])
        self.assertEqual(rows[0][1], "12.5")
        for row in rows[1:]:
            self.assertEqual(row[1].lower(), "nan")
        # With 2 job slots, 16 may be in flight; 32 must never spawn.
        self.assertNotIn((32, "tsit5"), harness.spawned)

    def test_failure_leaves_other_algorithm_running(self):
        harness = DriverHarness(
            self, {(4, "tsit5"): "launch_failure"}, grid=(4, 8),
            algorithms=("tsit5", "rosenbrock23_sciml"))
        self.assertEqual(julia_driver.run_states(["all"]), 0)
        tsit5 = harness.rows("fixed", "tsit5")
        self.assertTrue(all(r[1].lower() == "nan" for r in tsit5))
        rosen = harness.rows("fixed", "rosenbrock23_sciml")
        self.assertEqual([r[1] for r in rosen], ["12.5", "12.5"])

    def test_cancelled_inflight_process_is_killed(self):
        # 8 hangs until the failure of 4 cancels it mid-poll.
        harness = DriverHarness(
            self, {(4, "tsit5"): "launch_failure", (8, "tsit5"): "hang"},
            grid=(4, 8))
        self.assertEqual(julia_driver.run_states(["tsit5"]), 0)
        rows = harness.rows("fixed", "tsit5")
        self.assertEqual([r[0] for r in rows], ["4", "8"])
        self.assertTrue(all(r[1].lower() == "nan" for r in rows))

    def test_budget_kills_markerless_process_and_cancels_larger(self):
        os.environ["BENCH_STATES_BUDGET"] = "0.000001"
        self.addCleanup(os.environ.pop, "BENCH_STATES_BUDGET", None)
        os.environ["BENCH_JULIA_JOBS"] = "1"
        harness = DriverHarness(
            self, {(4, "tsit5"): "hang"}, grid=(4, 8, 16))
        self.assertEqual(julia_driver.run_states(["tsit5"]), 0)
        rows = harness.rows("fixed", "tsit5")
        self.assertEqual([r[0] for r in rows], ["4", "8", "16"])
        self.assertTrue(all(r[1].lower() == "nan" for r in rows))
        self.assertNotIn((8, "tsit5"), harness.spawned)

    def test_torn_last_line_is_backfilled(self):
        harness = DriverHarness(
            self, {(8, "tsit5"): "torn"}, grid=(4, 8))
        self.assertEqual(julia_driver.run_states(["tsit5"]), 0)
        rows = harness.rows("fixed", "tsit5")
        self.assertEqual([r[0] for r in rows], ["4", "8"])
        self.assertEqual(rows[0][1], "12.5")
        self.assertEqual(rows[1][1].lower(), "nan")


if __name__ == "__main__":
    unittest.main()
