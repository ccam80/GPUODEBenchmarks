"""States-driver and performance-driver tests with subprocess.Popen faked."""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "runner_scripts", "gpu"))

import julia_driver  # noqa: E402
import resume  # noqa: E402
from problems import get_problem  # noqa: E402
from protocol import STATES_N  # noqa: E402
from results import Leg  # noqa: E402

NAN = float("nan")


class FakeProc(object):
    """Scripted bench process; 'hang' behavior never returns."""

    def __init__(self, nstates, algorithm, legs, behavior, ticks=1):
        self.nstates = nstates
        self.algorithm = algorithm
        self.legs = legs
        self.behavior = behavior
        self.ticks = ticks
        self.killed = False
        self._code = None

    def _write_rows(self, value):
        for (mode, alg), leg in self.legs.items():
            if alg == self.algorithm:
                leg.record_times(STATES_N, value, value, 0.0, build_s=1.0,
                                 states=self.nstates)

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
            self._write_rows(12.5)
        elif self.behavior == "launch_failure":
            # The bench catches the launch error and records NaN rows.
            self._write_rows(NAN)
        elif self.behavior == "silent":
            # Killed before any row was written.
            pass
        self._code = 0
        return self._code

    def kill(self):
        self.killed = True
        self._code = -9

    def wait(self):
        return self._code


class DriverHarness(object):
    """Patches julia_driver so run_states drives FakeProcs into a scratch store."""

    def __init__(self, case, behaviors, grid, algorithms=("tsit5",)):
        self.tmp = tempfile.mkdtemp(prefix="jd_test_")
        case.addCleanup(shutil.rmtree, self.tmp, True)
        self.behaviors = behaviors
        self.free_ram_gb = 999.0
        self.spawned = []
        self.live = []
        self.max_concurrent = 0
        self.legs = {(mode, algorithm): Leg("julia", "test", "states",
                                            "lorenz96", algorithm, mode,
                                            root=self.tmp)
                     for algorithm in algorithms
                     for mode in ("fixed", "adaptive")}

        def fake_popen(cmd, cwd=None, env=None):
            # [<julia launcher...>, "--project=.", BENCH, spec, algorithm]
            spec, algorithm = cmd[cmd.index(julia_driver.BENCH) + 1:][:2]
            nstates = int(spec.split(":")[1])
            behavior = self.behaviors.get((nstates, algorithm), "ok")
            proc = FakeProc(nstates, algorithm, self.legs, behavior)
            self.spawned.append((nstates, algorithm))
            self.live = [p for p in self.live if p.poll() is None]
            self.live.append(proc)
            self.max_concurrent = max(self.max_concurrent, len(self.live))
            return proc

        patches = [
            mock.patch.object(julia_driver.subprocess, "Popen", fake_popen),
            mock.patch.object(julia_driver.time, "sleep", lambda _s: None),
            mock.patch.object(julia_driver, "_available_ram_gb",
                              lambda: self.free_ram_gb),
            mock.patch.object(julia_driver, "STATES_GRID", tuple(grid)),
            mock.patch.object(julia_driver, "resolve_algorithms",
                              lambda request, fw: list(algorithms)),
            mock.patch.object(
                julia_driver, "supported_for",
                lambda fw, mode: tuple(algorithms)),
            mock.patch.object(julia_driver, "dataset_key", lambda: "test"),
            mock.patch.object(julia_driver, "DATA_ROOT", self.tmp),
        ]
        for patch in patches:
            patch.start()
            case.addCleanup(patch.stop)

    def rows(self, mode, algorithm):
        """[(states, min_ms)] of the host-path leg, by state count."""
        leg = self.legs[(mode, algorithm)]
        import results
        rows = [r for r in results.load(leg.path)
                if r["analysis"] == "states" and r["algorithm"] == algorithm
                and r["mode"] == mode and r["transfers"] == "both"]
        return sorted((int(r["states"]), r["min_ms"]) for r in rows)


class StatesDriverTests(unittest.TestCase):
    def setUp(self):
        os.environ["BENCH_JULIA_JOBS"] = "2"
        self.addCleanup(os.environ.pop, "BENCH_JULIA_JOBS", None)

    def test_all_sizes_succeed(self):
        harness = DriverHarness(self, {}, grid=(4, 8, 16))
        self.assertEqual(julia_driver.run_states(["tsit5"]), 0)
        rows = harness.rows("fixed", "tsit5")
        self.assertEqual([r[0] for r in rows], [4, 8, 16])
        self.assertTrue(all(r[1] == "12.5" for r in rows))

    def test_launch_failure_cancels_larger_sizes(self):
        harness = DriverHarness(
            self, {(8, "tsit5"): "launch_failure"}, grid=(4, 8, 16, 32))
        self.assertEqual(julia_driver.run_states(["tsit5"]), 0)
        # 4 succeeded; 8 failed; 16/32 cancelled and NaN-backfilled.
        rows = harness.rows("fixed", "tsit5")
        self.assertEqual([r[0] for r in rows], [4, 8, 16, 32])
        self.assertEqual(rows[0][1], "12.5")
        for row in rows[1:]:
            self.assertEqual(row[1], "nan")
        # With 2 job slots, 16 may be in flight; 32 must never spawn.
        self.assertNotIn((32, "tsit5"), harness.spawned)

    def test_failure_leaves_other_algorithm_running(self):
        harness = DriverHarness(
            self, {(4, "tsit5"): "launch_failure"}, grid=(4, 8),
            algorithms=("tsit5", "rosenbrock23_sciml"))
        self.assertEqual(julia_driver.run_states(["all"]), 0)
        tsit5 = harness.rows("fixed", "tsit5")
        self.assertTrue(all(r[1] == "nan" for r in tsit5))
        rosen = harness.rows("fixed", "rosenbrock23_sciml")
        self.assertEqual([r[1] for r in rosen], ["12.5", "12.5"])

    def test_cancelled_inflight_process_is_killed(self):
        # 8 hangs until the failure of 4 cancels it mid-poll.
        harness = DriverHarness(
            self, {(4, "tsit5"): "launch_failure", (8, "tsit5"): "hang"},
            grid=(4, 8))
        self.assertEqual(julia_driver.run_states(["tsit5"]), 0)
        rows = harness.rows("fixed", "tsit5")
        self.assertEqual([r[0] for r in rows], [4, 8])
        self.assertTrue(all(r[1] == "nan" for r in rows))

    def test_low_ram_serializes_spawns(self):
        harness = DriverHarness(self, {}, grid=(4, 8, 16))
        harness.free_ram_gb = 5.0
        self.assertEqual(julia_driver.run_states(["tsit5"]), 0)
        self.assertEqual(harness.max_concurrent, 1)
        rows = harness.rows("fixed", "tsit5")
        self.assertEqual([r[0] for r in rows], [4, 8, 16])
        self.assertTrue(all(r[1] == "12.5" for r in rows))

    def test_a_silent_process_is_backfilled(self):
        harness = DriverHarness(
            self, {(8, "tsit5"): "silent"}, grid=(4, 8))
        self.assertEqual(julia_driver.run_states(["tsit5"]), 0)
        rows = harness.rows("fixed", "tsit5")
        self.assertEqual([r[0] for r in rows], [4, 8])
        self.assertEqual(rows[0][1], "12.5")
        self.assertEqual(rows[1][1], "nan")


class PerfProc(object):
    """Scripted performance-leg process; exits with the given code."""

    def __init__(self, code):
        self._code = code
        self._ticks = 1

    def poll(self):
        if self._ticks > 0:
            self._ticks -= 1
            return None
        return self._code


class PerformanceDriverTests(unittest.TestCase):
    """One process per (problem, algorithm, mode); a hard exit fails the leg."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("BENCH_RESUME", None)
        os.environ.pop("BENCH_NO_OVERWRITE", None)
        os.environ.pop("BENCH_RESUME_FROM", None)
        resume._reset_cache()
        self.addCleanup(resume._reset_cache)
        os.environ["BENCH_JULIA_JOBS"] = "2"

        self.tmp = tempfile.mkdtemp(prefix="jd_perf_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.spawned = []
        self.exit_codes = {}

        def fake_popen(cmd, cwd=None, env=None):
            # [<julia launcher...>, "--project=.", BENCH, nlist, algorithm, "--problem", problem, "--mode", mode]
            args = cmd[cmd.index(julia_driver.BENCH) + 1:]
            self.spawned.append(args)
            key = (args[3], args[1], args[5])
            return PerfProc(self.exit_codes.get(key, 0))

        patches = [
            mock.patch.object(julia_driver.subprocess, "Popen", fake_popen),
            mock.patch.object(julia_driver.time, "sleep", lambda _s: None),
            mock.patch.object(julia_driver, "_available_ram_gb",
                              lambda: 999.0),
            mock.patch.object(julia_driver, "dataset_key", lambda: "test"),
            mock.patch.object(julia_driver, "resolve_algorithms",
                              lambda request, fw: ["tsit5"]),
            mock.patch.object(julia_driver, "resolve_problems",
                              lambda request, fw: [get_problem("lorenz")]),
            mock.patch.object(
                julia_driver, "supported_for",
                lambda fw, mode: ("tsit5",)),
            mock.patch.object(julia_driver, "DATA_ROOT", self.tmp),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def leg(self, mode):
        return Leg("julia", "test", "times", "lorenz", "tsit5", mode,
                   root=self.tmp)

    def modes_spawned(self):
        return [args[args.index("--mode") + 1] for args in self.spawned]

    def test_each_mode_gets_its_own_process(self):
        self.assertEqual(julia_driver.run_performance(["8,32"]), 0)
        self.assertEqual(len(self.spawned), 2)
        self.assertEqual(sorted(self.modes_spawned()), ["adaptive", "fixed"])
        for args in self.spawned:
            self.assertIn("--problem", args)
            self.assertIn("--mode", args)

    def test_mode_narrows_the_legs(self):
        self.assertEqual(julia_driver.run_performance(["8,32", "--mode", "fixed"]), 0)
        self.assertEqual(self.modes_spawned(), ["fixed"])
        self.assertNotIn("--mode", self.spawned[0][:3])
        with self.assertRaises(SystemExit):
            julia_driver.run_performance(["8,32", "--mode", "sideways"])

    def test_watchdog_hard_exit_fails_the_run(self):
        self.exit_codes[("lorenz", "tsit5", "adaptive")] = 3
        self.assertEqual(julia_driver.run_performance(["8,32"]), 1)
        # The sibling mode still gets its own process.
        self.assertEqual(len(self.spawned), 2)

    def test_covered_mode_is_pruned_alone(self):
        os.environ["BENCH_RESUME"] = "1"
        fixed = self.leg("fixed")
        fixed.record_times(8, 1.0, 2.0, 0.0)
        fixed.record_times(32, 1.0, 2.0, 0.0)
        self.assertEqual(julia_driver.run_performance(["8,32"]), 0)
        self.assertEqual(self.modes_spawned(), ["adaptive"])

    def test_no_overwrite_retries_the_nan_mode(self):
        os.environ["BENCH_NO_OVERWRITE"] = "1"
        fixed, adaptive = self.leg("fixed"), self.leg("adaptive")
        fixed.record_times(8, 1.0, 2.0, 0.0)
        fixed.record_times(32, 1.0, 2.0, 0.0)
        adaptive.record_times(8, 1.0, 2.0, 0.0)
        adaptive.nan_times([32])
        self.assertEqual(julia_driver.run_performance(["8,32"]), 0)
        self.assertEqual(self.modes_spawned(), ["adaptive"])


if __name__ == "__main__":
    unittest.main()
