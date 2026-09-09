"""bench.py: argument resolution, the plan output, the runner registry and the run loop with its hard-exit re-invocation, plus the clock verdicts."""

import glob
import math
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import bench  # noqa: E402
import clocks  # noqa: E402
import launch  # noqa: E402
import store  # noqa: E402
import trials  # noqa: E402
from protocol import TOLS, WATCHDOG_EXIT_CODE  # noqa: E402

FAKE_RUNNER = os.path.join(HERE, "fake_runner.py")


class ParseTests(unittest.TestCase):
    def test_a_command_is_required_and_the_flags_parse(self):
        with self.assertRaises(SystemExit):
            bench.parse_args([])
        args = bench.parse_args(["plan", "-p", "cubie,julia-gpu", "--for", "perf,wp", "-n", "32",
                                 "--point", "cubie/lorenz", "--point", "jax/pollu", "--tier", "default"])
        self.assertEqual(args.command, "plan")
        self.assertEqual(args.point, ["cubie/lorenz", "jax/pollu"])
        self.assertEqual(args.views, "perf,wp")
        self.assertEqual(bench.resolve_packages(args.package), ["cubie", "julia_gpu"])
        with self.assertRaises(SystemExit):
            bench.parse_args(["run", "--resume", "--no-overwrite"])
        with self.assertRaises(SystemExit):
            bench.parse_args(["sweep"])

    def test_packages_resolve_in_run_order_and_unknown_names_exit(self):
        self.assertEqual(bench.resolve_packages("all"), list(launch.PACKAGES))
        self.assertEqual(bench.resolve_packages("jax,cubie_mlir,cubie"), ["cubie", "cubie_mlir", "jax"])
        with self.assertRaises(SystemExit):
            bench.resolve_packages("fortran")
        with self.assertRaises(SystemExit):
            bench.resolve_packages("julia")

    def test_bad_axis_values_exit(self):
        for argv in (["plan", "-n", "x"], ["plan", "--tier", "gold"], ["plan", "--transfers", "d2h"],
                     ["plan", "--setting", "fast"], ["plan", "--states", "many"],
                     ["plan", "-s", "lorenz1000"], ["plan", "-g", "rk9"], ["plan", "--for", "plots"]):
            with self.assertRaises(SystemExit):
                bench.plan_trials(bench.parse_args(argv), "test", tempfile.gettempdir())


SHIPPED = {"step_controller": "i", "integral_gain": 0.2, "safety": 0.9,
           "min_step_shrink": 0.2, "max_step_growth": 10.0}
PI = {"step_controller": "pi", "integral_gain": 0.3, "proportional_gain": 0.1,
      "safety": 0.9, "min_step_shrink": 0.2, "max_step_growth": 10.0}


def without_cubie(case):
    """The controller hooks answer without importing cubie."""
    for name, value in (("shipped_controller", lambda row: dict(SHIPPED)),
                        ("pi_controller", lambda row: dict(PI))):
        patcher = mock.patch.object(trials, name, value)
        patcher.start()
        case.addCleanup(patcher.stop)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bench_plan_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        without_cubie(self)

    def plan(self, *argv):
        args = bench.parse_args(["plan"] + list(argv))
        return args, bench.plan_trials(args, "test", self.tmp)

    def test_plan_writes_one_file_per_package_and_prints_counts(self):
        args, selected = self.plan("-p", "cubie,jax", "-s", "lorenz", "-g", "tsit5", "--for", "perf", "-n", "32")
        summary = trials.counts(selected)
        self.assertEqual(summary["cubie"]["solve"], 4)
        self.assertEqual((summary["cubie"]["warm"], summary["cubie"]["optimize"]), (2, 2))
        self.assertEqual(summary["jax"]["solve"], 4)
        paths = bench.write_plan(selected, ["cubie", "jax", "pytorch"], self.tmp)
        self.assertEqual(sorted(paths), ["cubie", "jax"])
        self.assertEqual(len(trials.read_jsonl(paths["cubie"])), 8)
        text = bench.counts_text(selected)
        self.assertIn("cubie: 4 solve, 2 warm, 2 optimize", text)
        self.assertIn("  lorenz/tsit5/fixed/n: 2", text)
        self.assertEqual(bench.counts_text([]), "no trials")

    def test_the_real_tables_expand_every_view(self):
        _, selected = self.plan("-n", "32")
        summary = trials.counts(selected)
        self.assertEqual(set(summary), set(launch.PACKAGES))
        for package, entry in summary.items():
            self.assertGreater(entry["solve"], 0, package)
        self.assertEqual(summary["pytorch"]["warm"], 0)
        self.assertEqual(summary["julia_cpu"]["optimize"], 0)
        self.assertEqual({t.tier for t in selected if t.package == "julia_gpu"}, {"default"})
        self.assertIn("pi", {t.tier for t in selected if t.package == "cubie"})
        self.assertNotIn("matched", {t.tier for t in selected})
        self.assertEqual(len({t.id for t in selected}), len(selected))

    def test_filters_reach_the_plan(self):
        _, selected = self.plan("-p", "cubie", "-s", "lorenz", "-g", "tsit5", "--for", "perf,wp", "-n", "32",
                                "--transfers", "none", "--setting", str(TOLS[1]), "--mode", "adaptive")
        solve = [t for t in selected if t.kind == "solve"]
        self.assertEqual([(t.n, t.setting, t.transfers) for t in solve], [(131072, TOLS[1], ["none"])])
        _, resumed = self.plan("-p", "cubie", "-s", "lorenz", "-g", "tsit5", "--for", "perf", "-n", "8", "--resume")
        self.assertEqual(len([t for t in resumed if t.kind == "solve"]), 2)
        first = [t for t in resumed if t.kind == "solve"][0]
        for transfers in first.transfers:
            store.Store(self.tmp).record(dict(first.identity("test", transfers), min_ms=1.0))
        _, resumed = self.plan("-p", "cubie", "-s", "lorenz", "-g", "tsit5", "--for", "perf", "-n", "8", "--resume")
        remaining = [t for t in resumed if t.kind == "solve"]
        self.assertEqual(len(remaining), 1)
        self.assertNotEqual(remaining[0].id, first.id)


class RunLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bench_run_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = os.path.join(self.tmp, "data")
        self.logs = os.path.join(self.tmp, "logs")
        os.makedirs(self.data)
        patcher = mock.patch.object(clocks.ClockGuard, "start_monitor", lambda self, path: False)
        patcher.start()
        self.addCleanup(patcher.stop)
        without_cubie(self)
        self.store = store.Store(self.data)

    def register(self, modes):
        runners = {package: (lambda mode=mode: [sys.executable, FAKE_RUNNER, "--root", self.data,
                                                "--key", "test", "--mode", mode])
                   for package, mode in modes.items()}
        patcher = mock.patch.dict(launch.RUNNERS, runners, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_bench(self, *argv):
        args = bench.parse_args(["run", "--no-lock-clocks", "--cooldown", "0"] + list(argv))
        packages = bench.resolve_packages(args.package)
        selected = bench.plan_trials(args, "test", self.data)
        run = bench.Run(args, "test", data_root=self.data, log_root=self.logs)
        code = run.execute(packages, selected)
        with open(run.summary, encoding="utf-8") as handle:
            summary = {line.split("\t")[0]: line.rstrip("\n").split("\t")[1:] for line in handle if line.strip()}
        return code, summary, run

    def test_an_ok_runner_records_every_trial_once(self):
        self.register({"cubie": "ok"})
        code, summary, run = self.run_bench("-p", "cubie", "-s", "lorenz", "-g", "tsit5", "--for", "perf", "-n", "32")
        self.assertEqual(code, 0)
        self.assertEqual(summary["run:cubie"][:2], ["OK", "4 solve trials, 1 attempt(s)"])
        rows = self.store.rows(package="cubie")
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(math.isfinite(r["min_ms"]) for r in rows))
        self.assertEqual(sorted(glob.glob(os.path.join(run.log_dir, "*.jsonl"))),
                         [os.path.join(run.log_dir, "cubie.jsonl")])
        self.assertTrue(os.path.isfile(os.path.join(run.log_dir, "cubie.log")))
        with open(os.path.join(run.log_dir, "run_manifest.txt"), encoding="utf-8") as handle:
            manifest = handle.read()
        self.assertIn("dataset_key=test", manifest)
        self.assertIn("finished_utc=", manifest)

    def test_a_hard_exit_abandons_the_leg_and_re_invokes_with_the_rest(self):
        self.register({"cubie": "hang-once"})
        code, summary, run = self.run_bench("-p", "cubie", "-s", "lorenz", "-g", "tsit5", "--for", "perf", "-n", "128")
        self.assertEqual(code, 0)
        self.assertEqual(summary["run:cubie"][:2], ["OK", "6 solve trials, 2 attempt(s)"])
        fixed = {(r["n"], r["transfers"]): r for r in self.store.rows(package="cubie", mode="fixed")}
        self.assertEqual(len(fixed), 6)
        self.assertTrue(math.isfinite(fixed[(8, "both")]["min_ms"]))
        for n in (32, 128):
            for transfers in ("both", "none"):
                row = fixed[(n, transfers)]
                self.assertTrue(math.isnan(row["min_ms"]))
                self.assertEqual(row["reason"], "abandoned: hard-exit at ordinal 1")
                self.assertEqual(row["suite_rev"], store.suite_rev(ROOT))
        adaptive = self.store.rows(package="cubie", mode="adaptive")
        self.assertEqual(len(adaptive), 6)
        self.assertTrue(all(math.isfinite(r["min_ms"]) for r in adaptive))
        second = trials.read_jsonl(os.path.join(run.log_dir, "cubie_1.jsonl"))
        self.assertEqual({t.leg for t in second}, {"lorenz/tsit5/adaptive/n"})
        self.assertEqual(sorted(t.ordinal for t in second if t.kind == "solve"), [1, 2])
        self.assertEqual(sorted(t.kind for t in second if t.kind != "solve"), ["optimize", "warm"])

    def test_a_hard_exit_without_progress_and_a_crash_fail_the_package_only(self):
        self.register({"cubie": "hang-silent", "jax": "crash", "pytorch": "ok"})
        code, summary, _ = self.run_bench("-p", "cubie,jax,pytorch", "-s", "lorenz", "-g", "tsit5",
                                          "--for", "perf", "-n", "8")
        self.assertEqual(code, 1)
        self.assertEqual(summary["run:cubie"][0], "FAILED")
        self.assertIn("without a progress file", summary["run:cubie"][1])
        self.assertEqual(summary["run:jax"][:2][0], "FAILED")
        self.assertEqual(summary["run:jax"][2], "1")
        self.assertEqual(summary["run:pytorch"][0], "OK")
        self.assertEqual(len(self.store.rows(package="pytorch")), 2)
        self.assertEqual(self.store.rows(package="cubie"), [])

    def test_an_unported_package_is_reported_and_fails_the_run(self):
        self.register({})
        code, summary, _ = self.run_bench("-p", "myokit_cuda", "-s", "lorenz", "--for", "perf", "-n", "8")
        self.assertEqual(code, 1)
        self.assertEqual(summary["run:myokit_cuda"][0], "UNPORTED")
        self.assertIn("launch.RUNNERS", summary["run:myokit_cuda"][1])
        with self.assertRaises(launch.UnportedPackage):
            launch.runner_argv("cpp")
        with self.assertRaises(ValueError):
            launch.runner_argv("fortran")
        self.assertEqual(launch.runner_env("jax"), {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})

    def test_floor_reaches_the_runner_and_no_trials_is_skipped(self):
        self.register({"cubie": "ok"})
        code, summary, run = self.run_bench("-p", "cubie,jax", "-s", "nand_gate", "-g", "tsit5",
                                            "--for", "perf", "-n", "8", "--floor")
        self.assertEqual(summary["run:jax"][:2], ["SKIPPED", "no trials"])
        with open(os.path.join(run.log_dir, "cubie.log"), encoding="utf-8") as handle:
            self.assertIn("floor=True", handle.read())
        self.assertEqual(code, 0)


class LaunchTests(unittest.TestCase):
    def test_ordering_and_the_package_list(self):
        self.assertEqual(launch.ordered(["jax", "cubie_mlir", "cubie"]), ["cubie", "cubie_mlir", "jax"])
        self.assertEqual(launch.PACKAGES, store.PACKAGES)
        self.assertNotIn("julia", launch.PACKAGES)

    def test_julia_command_is_the_1_13_channel_unless_overridden(self):
        saved = os.environ.pop("JULIA", None)
        try:
            self.assertEqual(launch.julia_command(), ["julia", "+1.13"])
            os.environ["JULIA"] = "/opt/julia/bin/julia"
            self.assertEqual(launch.julia_command(), ["/opt/julia/bin/julia"])
        finally:
            os.environ.pop("JULIA", None)
            if saved is not None:
                os.environ["JULIA"] = saved


class ClockVerdictTests(unittest.TestCase):
    HEADER = "timestamp, clocks.sm [MHz], clocks.mem [MHz], temperature.gpu, power.draw [W], utilization.gpu [%], clocks_event_reasons.active\n"

    def write(self, rows):
        path = os.path.join(self.tmp, "clocks.csv")
        with open(path, "w") as handle:
            handle.write(self.HEADER)
            for second, sm, reasons in rows:
                handle.write("2026/09/08 10:00:{0:02d}.123, {1}, 6801, 60, 120.5, 99, {2}\n".format(second, sm, reasons))
        return path

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_steady_samples_are_ok_and_idle_ones_do_not_count(self):
        path = self.write([(s, 1470, "0x0000000000000000") for s in range(10)]
                          + [(11, 300, "0x0000000000000001")])
        verdict = clocks.slice_verdict(path, "2026/09/08 10:00:00.000", "2026/09/08 10:00:20.999", 1470)
        self.assertEqual(verdict["verdict"], "OK")
        self.assertEqual((verdict["n"], verdict["busy"], verdict["drift"]), (11, 10, 0))

    def test_a_few_off_samples_are_a_blip_and_many_are_drift(self):
        path = self.write([(s, 1470 if s != 3 else 1400, "0x0") for s in range(10)])
        blip = clocks.slice_verdict(path, "2026/09/08 10:00:00.000", "2026/09/08 10:00:20.999", 1470)
        self.assertEqual(blip["verdict"], "BLIP")
        path = self.write([(s, 1300, "0x0") for s in range(10)])
        drift = clocks.slice_verdict(path, "2026/09/08 10:00:00.000", "2026/09/08 10:00:20.999", 1470)
        self.assertEqual(drift["verdict"], "DRIFT")
        self.assertEqual(drift["worst"], 170)

    def test_throttle_bits_are_drift_and_windows_exclude_other_samples(self):
        path = self.write([(s, 1470, "0x0000000000000004") for s in range(3)])
        self.assertEqual(clocks.slice_verdict(path, "2026/09/08 10:00:00.000", "2026/09/08 10:00:20.999", 1470)["verdict"], "DRIFT")
        self.assertIsNone(clocks.slice_verdict(path, "2026/09/08 11:00:00.000", "2026/09/08 11:00:20.999", 1470))


if __name__ == "__main__":
    unittest.main()
