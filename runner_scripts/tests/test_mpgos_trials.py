"""mpgos_trials.py: the builds and points of a cpp trial file, the reasons for trials the package cannot run, NaN rows through the store, the context lines, and trial.cuh reading a trial file bit for bit."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import mpgos_trials  # noqa: E402
import sets  # noqa: E402
import store  # noqa: E402
import trials  # noqa: E402
from test_grid import nvcc_build  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
KEY = "windows_RTX-4070-SUPER"
NAN = float("nan")
CPP_TEST = os.path.join(REPO_ROOT, "GPU_ODE_MPGOS", "tests", "test_trial.cu")
SCRIPT = os.path.join(os.path.dirname(HERE), "mpgos_trials.py")


def cpp_trials(names, problems=None, n=(8, 32), root=None):
    """The cpp trials of the named sets at the given n list."""
    specs = sets.expand(list(names), KEY, root or os.path.join(REPO_ROOT, "data"), packages=["cpp"],
                        problems=problems, n=list(n))
    return [t for t in trials.build_trials(specs) if t["package"] == "cpp"]


def spec(**overrides):
    fields = dict(problem="lorenz", system_params="{}", duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0, n=8,
                  grid_dtype="float32", algorithm="classical-rk4", controller="fixed",
                  dt=2.0 ** -10, dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains="{}",
                  newton_atol=NAN, newton_rtol=NAN, package="cpp", transfers=["both", "none"],
                  finals=False, axis="n", build="warm", optimize=None)
    fields.update(overrides)
    return fields


class ListingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mpgos_trials_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, trial_list):
        path = os.path.join(self.tmp, "cpp.jsonl")
        trials.write_jsonl(path, trial_list)
        return path

    def test_builds_are_one_binary_per_problem_solver_n_states_and_precision(self):
        trial_list = cpp_trials(["perf", "golden_grid", "states"], n=(8, 32, 131072))
        builds = mpgos_trials.builds(trial_list)
        keys = [(b["problem"], b["solver"], b["nt"], b["sd"], b["precision"]) for b in builds]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual({b["precision"] for b in builds}, {"float32"})
        self.assertEqual({b["solver"] for b in builds}, {"RK4", "RKCK45"})
        lorenz = [b for b in builds if b["problem"] == "lorenz"]
        self.assertEqual(sorted((b["solver"], b["nt"]) for b in lorenz),
                         [("RK4", 8), ("RK4", 32), ("RK4", 131072), ("RKCK45", 8), ("RKCK45", 32),
                          ("RKCK45", 131072)])
        self.assertTrue(all(b["sd"] == "-" and not b["cold"] for b in lorenz))
        # Every states point, the default 32 included, builds its binary cold.
        states = [b for b in builds if b["problem"] == "lorenz96" and b["cold"]]
        self.assertEqual(sorted({int(b["sd"]) for b in states}), [4, 8, 16, 32, 64, 128])
        self.assertEqual({b["nt"] for b in states}, {131072})
        merged = [b for b in builds if b["problem"] == "lorenz96" and b["sd"] == "32"]
        self.assertEqual(len(merged), 6)
        self.assertEqual(sorted((b["nt"], b["cold"]) for b in merged),
                         [(8, False), (8, False), (32, False), (32, False), (131072, True), (131072, True)])
        # Every trial has a build.
        for t in trial_list:
            self.assertIn(mpgos_trials.build_key(t), keys, t["trial_id"])

    def test_points_follow_the_file_order_with_the_build_key_transfers_and_finals(self):
        trial_list = cpp_trials(["perf", "golden_grid"], problems=["lorenz"], n=(8, 32, 131072))
        points = mpgos_trials.points(trial_list)
        self.assertEqual([p["trial_id"] for p in points], [t["trial_id"] for t in trial_list])
        for p, t in zip(points, trial_list):
            self.assertEqual(p["solver"], "RK4" if t["algorithm"] == "classical-rk4" else "RKCK45")
            self.assertEqual(p["nt"], t["n"])
            self.assertEqual(p["transfers"], ",".join(t["transfers"]))
            self.assertEqual(p["finals"], t["finals"])
            self.assertEqual(p["cold"], t["cold"])
            self.assertEqual(p["reason"], "")
        self.assertIn("both,none", {p["transfers"] for p in points})
        self.assertIn("none", {p["transfers"] for p in points})
        self.assertTrue(any(p["finals"] for p in points))
        # The thirteen RK4 steps at 131072: the 2^-10 point carries perf's transfers too.
        steps = [p for p in points if p["nt"] == 131072 and p["solver"] == "RK4"]
        self.assertEqual(len(steps), 13)
        self.assertEqual([p["transfers"] for p in steps], ["none"] * 9 + ["both,none"] + ["none"] * 3)
        self.assertEqual({p["finals"] for p in steps}, {True})
        # The abandon rule over the file: a failure at 131072 gives up nothing easier.
        hung = [t for t in trial_list if t["algorithm"] == "classical-rk4" and t["n"] == 131072][9]
        harder = mpgos_trials.harder(trial_list, hung["trial_id"])
        self.assertEqual(len(harder), 3)
        self.assertTrue(all(t["dt"] < hung["dt"] for t in trial_list if t["trial_id"] in harder))
        with self.assertRaises(ValueError):
            mpgos_trials.harder(trial_list, "0000000000000000")

    def test_trials_the_package_cannot_run_carry_a_reason_and_no_build(self):
        trial_list = trials.build_trials([
            spec(algorithm="tsit5", controller="fixed"),
            spec(problem="nand_gate", parameter="c9", grid_min=2.5e-5, grid_max=1e-4, duration=80.0),
            spec(precision="float64"),
        ])
        points = mpgos_trials.points(trial_list)
        self.assertEqual([p["reason"] for p in points],
                         ["error: ValueError: cpp has no solver for tsit5", "",
                          "error: ValueError: cpp has no problem header for nand_gate"])
        self.assertEqual(points[0]["solver"], "-")
        builds = mpgos_trials.builds(trial_list)
        self.assertEqual(len(builds), 1)
        self.assertEqual((builds[0]["problem"], builds[0]["solver"], builds[0]["precision"]),
                         ("lorenz", "RK4", "float64"))

    def test_states_come_from_system_params_or_the_problem_header(self):
        self.assertEqual(mpgos_trials.header_states("lorenz"), 3)
        self.assertEqual(mpgos_trials.header_states("lorenz96"), 32)
        self.assertEqual(mpgos_trials.header_states("pleiades"), 28)
        self.assertIsNone(mpgos_trials.header_states("nand_gate"))
        self.assertEqual(mpgos_trials.states_of(spec()), 3)
        self.assertEqual(mpgos_trials.states_of(spec(problem="lorenz96", system_params='{"states":64}')), 64)
        self.assertEqual(mpgos_trials.build_key(spec(problem="lorenz96", system_params='{"states":64}'))[3], "64")

    def test_nan_rows_record_the_reason_states_build_s_and_versions(self):
        root = os.path.join(self.tmp, "data")
        trial_list = trials.build_trials([spec(), spec(problem="lorenz96", parameter="F",
                                                       grid_max=16.0, system_params='{"states":64}')])
        path = self.write(trial_list)
        solves = trial_list
        rows = mpgos_trials.nan_rows(trial_list, solves[0]["trial_id"], KEY, ["both", "none"],
                                     "error: BuildError: nvcc failed", build_s="1.250",
                                     src_hash="abcdef123456", suite_rev="deadbeef")
        self.assertEqual([r["transfers"] for r in rows], ["both", "none"])
        self.assertEqual(rows[0]["states"], 3)
        self.assertEqual(rows[0]["build_s"], 1.25)
        self.assertTrue(rows[0]["package_version"].startswith("abcdef123456+nvcc"))
        self.assertEqual(rows[0]["suite_rev"], "deadbeef")
        with self.assertRaises(ValueError):
            mpgos_trials.nan_rows(trial_list, "0000000000000000", KEY, ["both"], "x")
        proc = subprocess.run([sys.executable, SCRIPT, "--root", root, "nan", path, solves[1]["trial_id"],
                               KEY, "none", "error: ProcessError: Bench.exe exit 1"],
                              capture_output=True, text=True, cwd=REPO_ROOT)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        stored = store.Store(root).rows(package="cpp")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["states"], 64)
        self.assertEqual(stored[0]["transfers"], "none")
        self.assertEqual(stored[0]["reason"], "error: ProcessError: Bench.exe exit 1")
        self.assertTrue(stored[0]["min_ms"] != stored[0]["min_ms"])
        self.assertEqual(stored[0]["run_id"], store.run_id(dict(solves[1], transfers="none", key=KEY)))

    def test_the_cli_prints_tab_separated_builds_and_points(self):
        path = self.write(cpp_trials(["perf"], problems=["lorenz"], n=(8,)))
        builds = subprocess.run([sys.executable, SCRIPT, "builds", path], capture_output=True, text=True)
        self.assertEqual(builds.returncode, 0, builds.stderr)
        rows = [line.split("\t") for line in builds.stdout.splitlines()]
        self.assertEqual(sorted(r[:6] for r in rows), [["lorenz", "RK4", "8", "-", "float32", "false"],
                                                       ["lorenz", "RKCK45", "8", "-", "float32", "false"]])
        points = subprocess.run([sys.executable, SCRIPT, "points", path], capture_output=True, text=True)
        self.assertEqual(points.returncode, 0, points.stderr)
        rows = [line.split("\t") for line in points.stdout.splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual([len(r) for r in rows], [len(mpgos_trials.POINT_COLUMNS)] * 2)
        self.assertEqual(rows[0][6:], ["both,none", "false", "false", ""])
        harder = subprocess.run([sys.executable, SCRIPT, "harder", path, rows[0][0]], capture_output=True, text=True)
        self.assertEqual((harder.returncode, harder.stdout), (0, ""))

    def test_context_lists_the_run_constants(self):
        context = mpgos_trials.context()
        self.assertEqual(set(context), {"key", "source_hash", "package_version", "suite_rev", "watchdog_exit"})
        self.assertTrue(re.fullmatch(r"[0-9a-f]{12}", context["source_hash"]))
        self.assertTrue(context["package_version"].startswith(context["source_hash"] + "+nvcc"))
        self.assertEqual(context["watchdog_exit"], 3)
        self.assertTrue(os.path.isfile(mpgos_trials.PROTOCOL_HEADER))
        names = [os.path.basename(p) for p in mpgos_trials.source_files()]
        for name in ("Bench.cu", "grid.cuh", "trial.cuh", "protocol.h", "makefile", "lorenz.cuh", "stubs.cuh"):
            self.assertIn(name, names)


class CppTrialTests(unittest.TestCase):
    def test_trial_cuh_reads_a_trial_file_and_emits_a_row_the_store_hashes_to_the_same_ids(self):
        if shutil.which("nvcc") is None:
            self.skipTest("nvcc is not on PATH")
        tmp = tempfile.mkdtemp(prefix="trial_cuh_")
        self.addCleanup(shutil.rmtree, tmp, True)
        exe = os.path.join(tmp, "test_trial.exe" if os.name == "nt" else "test_trial")
        built = nvcc_build(CPP_TEST, exe)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        trial_list = cpp_trials(["perf", "golden_grid"], problems=["lorenz"], n=(8,))
        path = os.path.join(tmp, "cpp.jsonl")
        trials.write_jsonl(path, trial_list)
        target = [t for t in trial_list if t["controller"] == "fixed" and len(t["transfers"]) == 2][0]
        row_path, spec_path = os.path.join(tmp, "row.json"), os.path.join(tmp, "spec.json")
        proc = subprocess.run([exe, path, target["trial_id"], KEY, row_path, spec_path],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        fields = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
        self.assertEqual(fields["count"], str(len(trial_list)))
        self.assertEqual(fields["problem"], "lorenz")
        self.assertEqual(fields["algorithm"], "classical-rk4")
        self.assertEqual(fields["controller"], "fixed")
        self.assertEqual(fields["n"], "8")
        self.assertEqual(fields["states_param"], "-1")
        self.assertEqual(float(fields["dt"]), target["dt"])
        self.assertEqual(fields["atol_nan"], "1")
        self.assertEqual(fields["finals"], "1" if target["finals"] else "0")
        self.assertEqual(fields["cold"], "0")
        self.assertEqual(float(fields["watchdog_s"]), target["watchdog_s"])
        self.assertEqual(fields["transfers"], "both,none")
        self.assertEqual(fields["lists_none"], "1")
        with open(row_path, encoding="utf-8") as handle:
            row = json.load(handle)
        self.assertEqual(list(row)[:len(store.TRIAL_FIELDS)], list(store.TRIAL_FIELDS))
        self.assertEqual(row["transfers"], "both")
        self.assertEqual(row["key"], KEY)
        self.assertIsNone(row["dt_min"])
        self.assertEqual(row["samples_ms"], [2.0, 1.5])
        self.assertIsNone(row["build_s"])
        made = store.make_row(**row)
        self.assertEqual(made["trial_id"], target["trial_id"])
        self.assertEqual(made["run_id"], store.run_id(dict(target, transfers="both", key=KEY)))
        self.assertEqual(made["min_ms"], 1.5)
        self.assertEqual(made["package_version"], "abcdef123456+nvcc13.3")
        with open(spec_path, encoding="utf-8") as handle:
            finals_spec = json.load(handle)
        self.assertEqual(set(finals_spec), set(store.FINALS_FIELDS))
        self.assertEqual(store.trial_id(finals_spec), target["trial_id"])
        states_list = trials.build_trials([spec(problem="lorenz96", parameter="F", grid_max=16.0,
                                                system_params='{"states":64}', finals=True)])
        trials.write_jsonl(path, states_list)
        proc = subprocess.run([exe, path, states_list[-1]["trial_id"], KEY, row_path, spec_path],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        fields = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
        self.assertEqual(fields["states_param"], "64")
        self.assertEqual(fields["finals"], "1")


if __name__ == "__main__":
    unittest.main()
