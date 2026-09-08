"""bench.py: argument resolution, points, the per-package command plans and the clock verdicts."""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import bench  # noqa: E402
import clocks  # noqa: E402
import launch  # noqa: E402
from protocol import STATES_GRID, WATCHDOG_EXIT_CODE  # noqa: E402


class ResolveTests(unittest.TestCase):
    def plan(self, *argv):
        return bench.resolve(bench.parse_args(list(argv)))

    def test_defaults_run_every_package_and_the_default_analyses(self):
        plan = self.plan()
        self.assertEqual(plan["packages"], list(launch.PACKAGES))
        self.assertEqual(plan["analyses"], list(bench.DEFAULT_ANALYSES))
        self.assertEqual(plan["nlist"][:3], [8, 32, 128])
        self.assertEqual(plan["algorithm"], "all")
        self.assertEqual(plan["problem"], "all")

    def test_cubie_packages_run_first_and_hyphens_are_accepted(self):
        plan = self.plan("-p", "julia,cubie-mlir,cubie")
        self.assertEqual(plan["packages"], ["cubie", "cubie_mlir", "julia"])

    def test_timed_analyses_pull_in_plots(self):
        plan = self.plan("-a", "performance")
        self.assertEqual(plan["analyses"], ["performance", "plots"])
        self.assertFalse(plan["plot_all"])
        self.assertTrue(self.plan("-a", "plots")["plot_all"])
        self.assertEqual(self.plan("-a", "optimize,warm")["analyses"], ["optimize", "warm"])

    def test_ne_and_overlap_take_either_cubie_package(self):
        self.assertEqual(bench.ne_package(["julia", "cubie_mlir"]), "all")
        self.assertEqual(bench.ne_package(["cubie_mlir"]), "cubie")
        self.assertEqual(bench.ne_package(["pytorch"]), "")
        self.assertEqual(bench.cubie_packages(["cubie", "julia", "cubie_mlir"]), ["cubie", "cubie_mlir"])

    def test_exact_counts_and_bad_values(self):
        self.assertEqual(self.plan("-n", "32768,8")["nlist"], [8, 32768])
        with self.assertRaises(SystemExit):
            self.plan("-n", "x")
        with self.assertRaises(SystemExit):
            self.plan("-p", "fortran")
        with self.assertRaises(SystemExit):
            self.plan("-a", "warmup")
        with self.assertRaises(SystemExit):
            self.plan("-g", "rk9")
        with self.assertRaises(SystemExit):
            self.plan("-s", "lorenz1000")

    def test_mode_axis(self):
        self.assertEqual(self.plan()["mode"], "all")
        self.assertEqual(self.plan("--mode", "adaptive")["mode"], "adaptive")
        self.assertEqual(self.plan("--mode", "fixed,adaptive")["mode"], "all")
        with self.assertRaises(SystemExit):
            self.plan("--mode", "sideways")

    def test_a_run_clears_only_what_it_records(self):
        ident = bench.clear_identity("cubie", "k", "performance", "euler,tsit5", "lorenz", "fixed", [8, 32])
        self.assertEqual(ident, {"package": "cubie", "key": "k", "analysis": "times",
                                 "algorithm": ["euler", "tsit5"], "problem": ["lorenz"],
                                 "mode": "fixed", "n": ["8", "32"]})
        states = bench.clear_identity("cpp", "k", "states", "all", "all", "all", [8])
        self.assertEqual(states, {"package": "cpp", "key": "k", "analysis": "states",
                                  "states": [str(s) for s in STATES_GRID]})
        wp = bench.clear_identity("julia", "k", "work-precision", "all", "pollu", "all", [8])
        self.assertEqual(wp, {"package": "julia", "key": "k", "analysis": "wp", "problem": ["pollu"]})

    def test_continuation_flags_imply_keep(self):
        for flag in ("--resume", "--no-overwrite", "--floor"):
            self.assertTrue(bench.parse_args([flag]).keep)
        args = bench.parse_args(["--resume-from", "cubie:pollu:tsit5:adaptive:262144"])
        self.assertTrue(args.keep)
        plan = bench.resolve(args)
        self.assertEqual((plan["resume_pkg"], plan["resume_tail"]),
                         ("cubie", "pollu:tsit5:adaptive:262144"))
        with self.assertRaises(SystemExit):
            self.plan("--resume-from", "fortran:lorenz")


class PointTests(unittest.TestCase):
    def test_times_point_carries_mode_and_n(self):
        point = bench.Point("times:cubie:lorenz:tsit5:fixed:32768")
        self.assertEqual(point.stage, "performance")
        self.assertEqual(point.identity("k"), {
            "package": "cubie", "key": "k", "analysis": "times", "problem": "lorenz",
            "algorithm": "tsit5", "mode": "fixed", "n": "32768"})

    def test_wp_point_needs_no_n_and_states_point_carries_the_count(self):
        wp = bench.Point("wp:julia:pollu:kvaerno3")
        self.assertEqual(wp.stage, "work-precision")
        self.assertEqual(wp.mode, "all")
        self.assertNotIn("n", wp.identity("k"))
        self.assertNotIn("mode", wp.identity("k"))
        states = bench.Point("states:cpp:lorenz96:classical-rk4:16")
        self.assertEqual(states.identity("k")["states"], "16")

    def test_bad_points_are_refused(self):
        for spec in ("times:cubie:lorenz:tsit5", "wp:fortran:lorenz:tsit5",
                     "perf:cubie:lorenz:tsit5:8", "times:cubie:lorenz:tsit5:sideways:8"):
            with self.assertRaises(SystemExit):
                bench.Point(spec)

    def test_points_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "points.txt")
            with open(path, "w") as handle:
                handle.write("# retakes\nwp:cubie:lorenz:tsit5\n\ntimes:julia:pollu:kvaerno3:adaptive:8\n")
            plan = bench.resolve(bench.parse_args(["--points-file", path]))
        self.assertEqual([p.analysis for p in plan["points"]], ["wp", "times"])


class LaunchTests(unittest.TestCase):
    def labels(self, package, analysis, algorithm="all", problem="all"):
        return [c.label for c in launch.commands(package, analysis, [8, 32], "32", algorithm, problem)]

    def test_cubie_performance_optimises_then_warms_then_sweeps(self):
        self.assertEqual(self.labels("cubie", "performance"), ["optimize", "warm", "performance"])
        self.assertEqual(self.labels("cubie_mlir", "work-precision"), ["optimize", "wp"])
        self.assertEqual(self.labels("cubie", "optimize"), ["optimize"])

    def test_other_packages_have_no_optimize_step(self):
        for package in ("julia", "cpp", "pytorch", "jax", "myokit_cuda"):
            self.assertEqual(launch.commands(package, "optimize", [8], "8", "all", "all"), [])
        self.assertEqual(self.labels("pytorch", "performance"), ["performance"])
        self.assertEqual(self.labels("jax", "performance"), ["warm", "performance"])

    def test_python_wp_legs_run_one_process_each(self):
        labels = self.labels("pytorch", "work-precision", "euler", "lorenz,pollu")
        self.assertEqual(labels, ["wp lorenz euler", "wp pollu euler"])
        command = launch.commands("pytorch", "work-precision", [8], "8", "euler", "lorenz")[0]
        self.assertIn(WATCHDOG_EXIT_CODE, command.ok)
        self.assertEqual(command.argv[-2:], ["--problem", "lorenz"])

    def test_julia_and_cpp_go_through_their_drivers(self):
        julia = launch.commands("julia", "performance", [8, 32], "32", "tsit5", "lorenz")[0]
        self.assertTrue(julia.argv[1].endswith("julia_driver.py"))
        self.assertEqual(julia.argv[2:], ["performance", "8,32", "tsit5", "lorenz"])
        cpp = launch.commands("cpp", "states", [8], "8", "all", "all")[0]
        self.assertEqual(cpp.argv[-10:], ["-a", "states", "-n", "8", "-g", "all", "-s", "all", "-m", "all"])

    def test_mode_reaches_every_package(self):
        cubie = launch.commands("cubie", "performance", [8], "8", "all", "all", "fixed")
        self.assertTrue(all(c.argv[-2:] == ["--mode", "fixed"] for c in cubie))
        julia = launch.commands("julia", "work-precision", [8], "8", "all", "all", "adaptive")[0]
        self.assertEqual(julia.argv[-2:], ["--mode", "adaptive"])
        cpp = launch.commands("cpp", "performance", [8], "8", "all", "all", "adaptive")[0]
        self.assertEqual(cpp.argv[-2:], ["-m", "adaptive"])
        # An adaptive-only run has no pytorch wp legs and the cubie warm step names no mode when every mode runs.
        self.assertEqual(launch.commands("pytorch", "work-precision", [8], "8", "all", "all", "adaptive"), [])
        self.assertNotIn("--mode", launch.commands("cubie", "warm", [8], "8", "all", "all")[0].argv)
        with self.assertRaises(ValueError):
            launch.commands("cubie", "performance", [8], "8", "all", "all", "sideways")

    def test_ordering_and_store_names(self):
        self.assertEqual(launch.ordered(["jax", "cubie_mlir", "cubie"]), ["cubie", "cubie_mlir", "jax"])
        self.assertEqual(launch.store_analysis("work-precision"), "wp")
        self.assertEqual(launch.store_analysis("performance"), "times")

    def test_numerical_is_the_cubie_ne_legs(self):
        self.assertEqual(self.labels("cubie", "numerical"), ["optimize", "ne"])
        self.assertEqual(launch.commands("cubie", "numerical", [8], "8", "all", "pollu")[1].argv[-3:],
                         ["all", "--problem", "pollu"])
        for package in ("julia", "pytorch", "cpp"):
            self.assertEqual(launch.commands(package, "numerical", [8], "8", "all", "all"), [])

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
