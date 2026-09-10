"""bench.py: flag resolution, the plan output, the -n check over heterogeneous grids, the completeness-aware continuation filters, the runner registry, and the watchdog hard-exit loop against a fake runner."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import bench  # noqa: E402
import completeness  # noqa: E402
import cubie_adapter  # noqa: E402
import launch  # noqa: E402
import sets  # noqa: E402
import store  # noqa: E402
import trials  # noqa: E402
from protocol import WATCHDOG_EXIT_CODE  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
NAN = float("nan")

# A runner that records every solve trial it reaches and exits 3 (once) while a chosen trial is in progress.
FAKE_RUNNER = '''
import json, os, sys
sys.path.insert(0, r"{runner_scripts}")
import store
argv = sys.argv[1:]
path = argv[argv.index("--trials") + 1]
trials = [json.loads(l) for l in open(path) if l.strip()]
marker = os.path.join(os.path.dirname(path), "plan.json")
plan = json.load(open(marker))
log = os.path.join(os.path.dirname(path), "calls.jsonl")
with open(log, "a") as h:
    h.write(json.dumps({{"path": os.path.basename(path), "argv": argv,
                         "ids": [t["trial_id"] for t in trials]}}) + "\\n")
data = store.Store(plan["root"])
for t in trials:
    with open(path + ".progress", "w") as h:
        json.dump({{"trial_id": t["trial_id"], "stage": "solve", "started_utc": "2026-09-09T00:00:00Z"}}, h)
    if not t["transfers"]:
        continue
    if t["trial_id"] == plan.get("trial_id") and not plan.get("done"):
        json.dump(dict(plan, done=True), open(marker, "w"))
        sys.exit(plan.get("code", 3))
    spec = {{f: t[f] for f in store.TRIAL_FIELDS}}
    data.record_batch([dict(spec, transfers=x, key=plan["key"], states=3, min_ms=1.0)
                       for x in t["transfers"]])
sys.exit(0)
'''.format(runner_scripts=os.path.dirname(HERE))


class ResolveTests(unittest.TestCase):
    def plan(self, *argv):
        return bench.resolve(bench.parse_args(list(argv)))

    def test_plan_and_run_take_sets_and_the_narrowing_flags(self):
        plan = self.plan("plan", "--set", "perf,golden_grid", "-p", "julia-gpu,cubie", "-s", "lorenz,pollu",
                         "-g", "tsit5", "--mode", "adaptive", "--controller", "default,matched",
                         "-n", "32,8", "--tol", "1e-5,1e-6", "--dt", "0.5")
        self.assertEqual(plan["sets"], ["perf", "golden_grid"])
        self.assertEqual(plan["packages"], ["cubie", "julia_gpu"])
        self.assertEqual(plan["problems"], ["lorenz", "pollu"])
        self.assertEqual(plan["algorithms"], ["tsit5"])
        self.assertEqual(plan["mode"], "adaptive")
        self.assertEqual(plan["controllers"], ["default", "matched"])
        self.assertEqual(plan["n"], [8, 32])
        self.assertEqual(plan["tols"], [1e-5, 1e-6])
        self.assertEqual(plan["dts"], [0.5])
        bare = self.plan("run", "--set", "golden")
        for key in ("packages", "problems", "algorithms", "mode", "controllers", "n", "tols", "dts"):
            self.assertIsNone(bare[key], key)

    def test_bad_flags_exit(self):
        for argv in (["--set", "perf"], ["plan"], ["plan", "--set", "nosuchset"],
                     ["plan", "--set", "perf", "-p", "fortran"], ["plan", "--set", "perf", "-s", "lorenz1000"],
                     ["plan", "--set", "perf", "-g", "rk9"], ["plan", "--set", "perf", "--mode", "sideways"],
                     ["plan", "--set", "perf", "-n", "x"], ["plan", "--set", "perf", "-n", "1"],
                     ["plan", "--set", "perf", "--tol", "tight"],
                     ["plan", "--set", "perf", "--resume", "--no-overwrite"], ["sweep", "--set", "perf"]):
            with self.assertRaises(SystemExit, msg=argv):
                self.plan(*argv)
        with self.assertRaises(SystemExit) as caught:
            self.plan("-h")
        self.assertEqual(caught.exception.code, 0)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bench_plan_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "data")

    def plan(self, *argv, **kw):
        return bench.plan_trials(bench.resolve(bench.parse_args(["plan"] + list(argv))), KEY, self.root, **kw)

    def test_plan_groups_trials_per_package_in_run_order_and_writes_the_files(self):
        by_package = self.plan("--set", "perf", "-p", "jax,pytorch,cpp", "-s", "lorenz", "-g", "classical-rk4",
                               "-n", "8,32")
        self.assertEqual(list(by_package), ["jax", "pytorch", "cpp"])
        for rows in by_package.values():
            self.assertEqual(trials.counts(rows), (2, 0, 0, 1))
        paths = bench.write_plan(os.path.join(self.tmp, "trials"), by_package)
        self.assertEqual(sorted(os.path.basename(p) for p in paths.values()),
                         ["cpp.jsonl", "jax.jsonl", "pytorch.jsonl"])
        back = trials.read_jsonl(paths["jax"])
        self.assertEqual([t["n"] for t in back], [8, 32])
        self.assertEqual(set(back[0]), set(trials.TRIAL_KEYS))

    def test_resume_drops_trials_whose_rows_exist_and_no_overwrite_those_finite(self):
        cpp = self.plan("--set", "perf", "-p", "cpp", "-s", "lorenz", "-n", "8,32")["cpp"]
        self.assertEqual(len(cpp), 4)
        recorded, nan_row, partial, absent = cpp
        data = store.Store(self.root)
        for transfers in ("both", "none"):
            spec = {f: recorded[f] for f in store.TRIAL_FIELDS}
            data.record(dict(spec, transfers=transfers, key=KEY, states=3, min_ms=1.0))
            spec = {f: nan_row[f] for f in store.TRIAL_FIELDS}
            data.record(dict(spec, transfers=transfers, key=KEY, states=3, min_ms=NAN, reason="error: x"))
        spec = {f: partial[f] for f in store.TRIAL_FIELDS}
        data.record(dict(spec, transfers="both", key=KEY, states=3, min_ms=2.0, finals="finals/p.parquet"))
        resumed = {t["trial_id"]: t for t in bench.continue_filter(cpp, KEY, self.root, resume=True)}
        self.assertEqual(set(resumed), {partial["trial_id"], absent["trial_id"]})
        # A partial trial runs its missing transfers alone and keeps asking finals once a row carries them.
        self.assertEqual((resumed[partial["trial_id"]]["transfers"], resumed[partial["trial_id"]]["finals"]),
                         (["none"], True))
        self.assertEqual(resumed[absent["trial_id"]]["transfers"], ["both", "none"])
        fresh = {t["trial_id"]: t for t in bench.continue_filter(cpp, KEY, self.root, no_overwrite=True)}
        self.assertEqual(set(fresh), {nan_row["trial_id"], partial["trial_id"], absent["trial_id"]})
        self.assertEqual(fresh[nan_row["trial_id"]]["transfers"], ["both", "none"])
        # A recorded package's lines drop while another package's stay.
        both = self.plan("--set", "perf", "-p", "cpp,pytorch", "-s", "lorenz", "-g", "classical-rk4", "-n", "8")
        for t in both["cpp"]:
            spec = {f: t[f] for f in store.TRIAL_FIELDS}
            for transfers in ("both", "none"):
                data.record(dict(spec, transfers=transfers, key=KEY, states=3, min_ms=1.0))
        mixed = bench.continue_filter(both["pytorch"] + both["cpp"], KEY, self.root, resume=True)
        self.assertEqual([t["package"] for t in mixed], ["pytorch"])
        # A trial that asks finals over rows without them runs its last transfers again.
        wants = dict(recorded, finals=True)
        again = bench.continue_filter([wants], KEY, self.root, resume=True)
        self.assertEqual([(t["transfers"], t["finals"]) for t in again], [(["none"], True)])
        spec = {f: recorded[f] for f in store.TRIAL_FIELDS}
        relative = data.record_finals(dict(spec, key=KEY), np.zeros((8, 3)), np.full(8, 1.0))
        for transfers in ("both", "none"):
            data.record(dict(spec, transfers=transfers, key=KEY, states=3, min_ms=1.0, finals=relative))
        self.assertEqual(bench.continue_filter([wants], KEY, self.root, resume=True), [])
        # A line without transfers never runs again; without a flag the plan stands.
        self.assertEqual(bench.continue_filter([dict(recorded, transfers=[])], KEY, self.root, resume=True), [])
        self.assertEqual(bench.continue_filter(cpp, KEY, self.root), cpp)

    def test_n_is_checked_against_every_grid_of_the_named_sets_not_the_filtered_specs(self):
        # The per-package smoke counts: 1024 is julia_cpu's golden_grid count, the others perf's.
        julia = self.plan("--set", "perf,golden_grid", "-p", "julia_cpu", "-s", "lorenz", "-g", "tsit5",
                          "-n", "128,1024,131072")
        self.assertEqual(list(julia), ["julia_cpu"])
        self.assertEqual({t["n"] for t in julia["julia_cpu"]}, {1024})
        cpp = self.plan("--set", "perf,golden_grid", "-p", "cpp", "-s", "lorenz", "-g", "classical-rk4",
                        "-n", "128,1024,131072")
        self.assertEqual({t["n"] for t in cpp["cpp"]}, {128, 131072})
        # A count declared by a grid no filtered package uses yields no trials and no error.
        self.assertEqual(self.plan("--set", "golden_grid", "-p", "cpp", "-s", "lorenz", "-n", "1024"), {})
        # A count no grid of the named sets lists exits.
        for argv in (("--set", "perf,golden_grid", "-p", "julia_cpu", "-n", "16"),
                     ("--set", "perf", "-p", "cpp", "-n", "1024"),
                     ("--set", "states,golden", "-n", "8")):
            with self.assertRaises(SystemExit, msg=argv) as caught:
                self.plan(*argv)
            self.assertIn("no grid of", str(caught.exception))

    def test_plan_merges_a_point_with_its_declarations_in_every_set_file(self):
        for argv in (("--set", "perf"), ("--set", "states"), ("--set", "golden_grid"),
                     ("--set", "golden_grid,states,perf")):
            by_package = self.plan(*argv, "-p", "cpp", "-s", "lorenz96", "-g", "classical-rk4", "-n", "131072",
                                   "--mode", "fixed", "--dt", str(2.0 ** -10))
            point = [t for t in by_package["cpp"] if t["system_params"] == '{"states":32}']
            self.assertEqual([(t["cold"], t["finals"], t["transfers"], t["optimize"], t["sets"]) for t in point],
                             [(True, True, ["both", "none"], None, ["golden_grid", "perf", "states"])], argv)

    def test_plan_cli_writes_the_trial_files_and_prints_counts(self):
        out = subprocess.run([sys.executable, os.path.join(ROOT, "bench.py"), "plan", "--set", "perf",
                              "-p", "cpp", "-s", "lorenz", "-n", "8", "--allow-unknown-gpu"],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("cpp: 2 trials, 0 optimize, 0 cold, 2 builds", out.stdout)
        self.assertIn("lorenz/{}/float32/classical-rk4/fixed/{}  1", out.stdout)
        self.assertIn("2 trials", out.stdout)
        written = [line for line in out.stdout.splitlines() if line.endswith("cpp.jsonl")]
        self.assertEqual(len(written), 1)
        path = os.path.join(ROOT, written[0])
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(len(trials.read_jsonl(path)), 2)
        os.remove(path)


class CompletenessTests(unittest.TestCase):
    """continue_filter over partially populated rows: the cold build time, the finals file and the optimize record are artifacts a row needs before it is reused."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bench_complete_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "data")
        self.data = store.Store(self.root)

    def plan(self, *argv):
        return bench.plan_trials(bench.resolve(bench.parse_args(["plan"] + list(argv))), KEY, self.root)

    def record(self, trial, transfers, **values):
        spec = {f: trial[f] for f in store.TRIAL_FIELDS}
        return self.data.record(dict(spec, transfers=transfers, key=KEY, states=3, min_ms=1.0, **values))

    def kept(self, trial_list, **kw):
        kw.setdefault("sources", lambda package, systems: {s: "S" for s in systems})
        return {t["trial_id"]: t for t in bench.continue_filter(trial_list, KEY, self.root, **kw)}

    def test_a_cold_line_needs_a_finite_build_time_on_every_row(self):
        states = self.plan("--set", "states", "-p", "cpp", "-g", "classical-rk4")["cpp"]
        self.assertEqual(len(states), 6)
        self.assertEqual({t["cold"] for t in states}, {True})
        for transfers in ("both", "none"):
            self.record(states[0], transfers)
            self.record(states[1], transfers, build_s=12.5 if transfers == "both" else NAN)
            self.record(states[2], transfers, build_s=12.5)
        kept = self.kept(states, resume=True)
        self.assertEqual(kept[states[0]["trial_id"]]["transfers"], ["both", "none"])
        self.assertEqual(kept[states[1]["trial_id"]]["transfers"], ["none"])
        self.assertNotIn(states[2]["trial_id"], kept)
        audits = completeness.audit(states, KEY, self.data, "resume")
        self.assertEqual(audits[states[0]["trial_id"]].reasons(), ["build:both", "build:none"])
        self.assertEqual(audits[states[1]["trial_id"]].reasons(), ["build:none"])
        self.assertTrue(audits[states[2]["trial_id"]].complete())
        # A warm line's rows never need one.
        perf = self.plan("--set", "perf", "-p", "cpp", "-s", "lorenz", "-g", "classical-rk4", "-n", "8")["cpp"]
        for transfers in ("both", "none"):
            self.record(perf[0], transfers)
        self.assertEqual(self.kept(perf, resume=True), {})

    def test_finals_need_a_readable_file(self):
        golden = self.plan("--set", "golden_grid", "-p", "cpp", "-s", "lorenz", "-g", "classical-rk4",
                           "--dt", "0.5,0.25")["cpp"]
        self.assertEqual([(t["dt"], t["finals"]) for t in golden], [(0.5, True), (0.25, True)])
        spec = {f: golden[0][f] for f in store.TRIAL_FIELDS}
        relative = self.data.record_finals(dict(spec, key=KEY), np.zeros((131072, 3)), np.full(131072, 1.0))
        self.record(golden[0], "none", finals=relative)
        self.record(golden[1], "none", finals="finals/gone.parquet")
        kept = self.kept(golden, resume=True)
        self.assertEqual(list(kept), [golden[1]["trial_id"]])
        self.assertEqual((kept[golden[1]["trial_id"]]["transfers"], kept[golden[1]["trial_id"]]["finals"]),
                         (["none"], True))
        self.assertEqual(completeness.audit(golden, KEY, self.data)[golden[1]["trial_id"]].reasons(), ["finals"])
        # A corrupt file is as absent.
        path = os.path.join(self.data.package_dir("cpp", KEY), *relative.split("/"))
        with open(path, "wb") as handle:
            handle.write(b"not parquet")
        self.assertIn(golden[0]["trial_id"], self.kept(golden, resume=True))
        # A canonical finals requirement over rows recorded without finals reruns the last transfers.
        perf = self.plan("--set", "perf", "-p", "cpp", "-s", "lorenz", "-g", "classical-rk4", "-n", "131072",
                         "--mode", "fixed")["cpp"]
        point = perf[0]
        self.assertTrue(point["finals"])
        for transfers in ("both", "none"):
            self.record(point, transfers)
        kept = self.kept(perf, resume=True)
        self.assertEqual((kept[point["trial_id"]]["transfers"], kept[point["trial_id"]]["finals"]), (["none"], True))

    def test_an_optimize_record_must_exist_from_the_current_source(self):
        perf = self.plan("--set", "perf", "-p", "cubie", "-s", "lorenz", "-g", "tsit5", "--mode", "fixed",
                         "-n", "8,32")["cubie"]
        self.assertEqual([(t["n"], t["optimize"]) for t in perf], [(8, "solve"), (32, "solve")])
        for trial in perf:
            for transfers in ("both", "none"):
                self.record(trial, transfers)
        # Complete rows, no record: every transfers of both lines runs again.
        kept = self.kept(perf, resume=True)
        self.assertEqual(sorted(kept), sorted(t["trial_id"] for t in perf))
        self.assertEqual({tuple(t["transfers"]) for t in kept.values()}, {("both", "none")})
        audits = completeness.audit(perf, KEY, self.data, "resume", lambda p, s: {x: "S" for x in s})
        self.assertEqual(audits[perf[0]["trial_id"]].reasons(), ["optimize:absent"])
        result = FakeOptimizeResult()
        for trial in perf:
            cubie_adapter.record_optimized(trial, KEY, result, trial["n"], root=self.root, source="S")
        self.assertEqual(self.kept(perf, resume=True), {})
        self.assertEqual(self.kept(perf, no_overwrite=True), {})
        # A per-solve record serves its own n alone.
        cubie_adapter.clear_optimized("cubie", KEY, "tsit5", "lorenz", root=self.root)
        cubie_adapter.record_optimized(perf[0], KEY, result, 8, root=self.root, source="S")
        self.assertEqual(list(self.kept(perf, resume=True)), [perf[1]["trial_id"]])
        cubie_adapter.record_optimized(perf[1], KEY, result, 32, root=self.root, source="S")
        # Recorded from another source: stale, so both lines run again.
        stale = self.kept(perf, resume=True, sources=lambda p, s: {x: "T" for x in s})
        self.assertEqual(sorted(stale), sorted(t["trial_id"] for t in perf))
        audits = completeness.audit(perf, KEY, self.data, "resume", lambda p, s: {x: "T" for x in s})
        self.assertEqual(audits[perf[1]["trial_id"]].reasons(), ["optimize:source S"])
        # A timed-out optimize stands under --resume and is attempted again under --no-overwrite.
        cubie_adapter.record_optimize_timeout(perf[1], KEY, self.root)
        self.assertEqual(self.kept(perf, resume=True), {})
        self.assertEqual(list(self.kept(perf, no_overwrite=True)), [perf[1]["trial_id"]])
        self.assertEqual(completeness.audit(perf, KEY, self.data, "no_overwrite")[perf[1]["trial_id"]].reasons(),
                         ["optimize:timeout"])
        # Without a current source (the analyses) a record of any source stands.
        for trial in perf:
            cubie_adapter.record_optimized(trial, KEY, result, trial["n"], root=self.root, source="old")
        self.assertTrue(all(m.complete() for m in completeness.audit(perf, KEY, self.data).values()))
        # The source hashes are asked once per cubie package for the systems of its optimizing lines.
        asked = []
        bench.continue_filter(perf, KEY, self.root, resume=True,
                              sources=lambda p, s: asked.append((p, list(s))) or {x: "old" for x in s})
        self.assertEqual(asked, [("cubie", [("lorenz", "{}", "float32")])])
        with mock.patch.object(cubie_adapter, "source_hashes", side_effect=RuntimeError("no cubie")):
            with self.assertRaises(SystemExit) as caught:
                bench.continue_filter(perf, KEY, self.root, resume=True)
        self.assertIn("no cubie", str(caught.exception))

    def test_each_kernel_has_its_own_optimize_record(self):
        golden = self.plan("--set", "golden_grid", "-p", "cubie", "-s", "lorenz", "-g", "classical-rk4",
                           "--dt", "0.5,0.25")["cubie"]
        self.assertEqual([(t["dt"], t["optimize"]) for t in golden], [(0.5, "kernel"), (0.25, "kernel")])
        result = FakeOptimizeResult()
        for trial in golden:
            spec = {f: trial[f] for f in store.TRIAL_FIELDS}
            relative = self.data.record_finals(dict(spec, key=KEY), np.zeros((131072, 3)), np.full(131072, 1.0))
            self.record(trial, "none", finals=relative)
        cubie_adapter.record_optimized(golden[0], KEY, result, 71680, root=self.root, source="S")
        # The 0.5 kernel stands with its record at any batch; the 0.25 kernel runs again.
        self.assertEqual(list(self.kept(golden, resume=True)), [golden[1]["trial_id"]])
        self.assertEqual(completeness.summary(completeness.audit(golden, KEY, self.data, "resume",
                                                                 lambda p, s: {x: "S" for x in s})),
                         {"optimize:absent": 1})


class FakeOptimizeResult:
    class Best:
        label, best_ms, blocksize, resident_blocks = "state=shared @bs128", 1.0, 128, 2

    best = Best()
    applied_settings = {"blocksize": 128}


class LaunchTests(unittest.TestCase):
    def test_every_package_has_a_runner_taking_the_trial_file(self):
        self.assertEqual(set(launch.RUNNERS), set(store.PACKAGES))
        for package in store.PACKAGES:
            command = launch.runner_command(package, "trials/k/x.jsonl")
            self.assertEqual(command.argv[-2:], ["--trials", "trials/k/x.jsonl"], package)
            self.assertEqual(command.ok, (0,))
        floored = launch.runner_command("cubie", "x.jsonl", floor=True)
        self.assertEqual(floored.argv[-3:], ["--trials", "x.jsonl", "--floor"])
        self.assertEqual(floored.env, {"CUBIE_MAX_CACHE_ENTRIES": "0"})
        self.assertEqual(launch.runner_command("jax", "x.jsonl").env, {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
        self.assertTrue(launch.runner_command("julia_gpu", "x.jsonl").argv[1].endswith("julia_driver.py"))
        julia_cpu = launch.runner_command("julia_cpu", "x.jsonl").argv
        self.assertEqual(julia_cpu[:2], launch.julia_command())
        self.assertTrue(julia_cpu[-3].endswith("bench_ode_cpu.jl"))
        cpp = launch.runner_command("cpp", "x.jsonl").argv
        self.assertTrue(cpp[-3].endswith("run_ode_cpp.ps1") or cpp[-3].endswith("run_ode_cpp.sh"))
        with self.assertRaises(ValueError):
            launch.runner_command("fortran", "x.jsonl")

    def test_ordering_and_the_julia_channel(self):
        self.assertEqual(launch.ordered(["jax", "cubie_mlir", "cubie"]), ["cubie", "cubie_mlir", "jax"])
        saved = os.environ.pop("JULIA", None)
        try:
            self.assertEqual(launch.julia_command(), ["julia", "+1.13"])
            os.environ["JULIA"] = "/opt/julia/bin/julia"
            self.assertEqual(launch.julia_command(), ["/opt/julia/bin/julia"])
        finally:
            os.environ.pop("JULIA", None)
            if saved is not None:
                os.environ["JULIA"] = saved


class HardExitTests(unittest.TestCase):
    """The runner loop against a fake runner: a hard exit abandons the harder runs of the family and the rest re-runs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bench_run_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "data")
        self.logs = os.path.join(self.tmp, "logs")
        self.runner = os.path.join(self.tmp, "fake_runner.py")
        with open(self.runner, "w", encoding="utf-8") as handle:
            handle.write(FAKE_RUNNER)
        saved = launch.RUNNERS["cpp"]
        launch.RUNNERS["cpp"] = lambda: [sys.executable, self.runner]
        self.addCleanup(launch.RUNNERS.__setitem__, "cpp", saved)

    def run_bench(self, hung=None, code=3, *argv):
        args = bench.parse_args(["run", "--set", "perf", "-p", "cpp", "-s", "lorenz", "-n", "8,32,128",
                                 "--no-lock-clocks", "--cooldown", "0"] + list(argv))
        run = bench.Run(args, bench.resolve(args), key=KEY, data_root=self.root, logs_root=self.logs)
        with open(os.path.join(run.log_dir, "plan.json"), "w") as handle:
            json.dump({"trial_id": hung, "code": code, "root": self.root, "key": KEY}, handle)
        status = run.execute()
        with open(os.path.join(run.log_dir, "calls.jsonl")) as handle:
            calls = [json.loads(line) for line in handle]
        with open(run.summary) as handle:
            summary = [line.rstrip("\n").split("\t") for line in handle]
        return status, run, calls, summary

    def planned(self):
        return bench.plan_trials(bench.resolve(bench.parse_args(
            ["run", "--set", "perf", "-p", "cpp", "-s", "lorenz", "-n", "8,32,128"])), KEY, self.root)["cpp"]

    def line(self, algorithm, n):
        return [t for t in self.planned() if t["algorithm"] == algorithm and t["n"] == n][0]

    def test_a_clean_runner_is_one_call(self):
        status, run, calls, summary = self.run_bench()
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["path"], "cpp.jsonl")
        self.assertEqual(summary, [["cpp", "OK", "-", "0"]])
        self.assertTrue(os.path.isfile(os.path.join(run.log_dir, "run_manifest.txt")))
        rows = store.Store(self.root).rows()
        self.assertEqual(len(rows), 12)
        self.assertEqual({r["min_ms"] for r in rows}, {1.0})

    def test_a_hard_exit_abandons_the_harder_runs_and_reruns_the_rest(self):
        # The cash-karp-54 build runs first; hang it at n = 32.
        hung = self.line("cash-karp-54", 32)
        status, run, calls, summary = self.run_bench(hung["trial_id"])
        self.assertEqual(status, 0)
        self.assertEqual(summary, [["cpp", "PARTIAL", "1 hard exit(s)", "0"]])
        self.assertEqual([c["path"] for c in calls], ["cpp.jsonl", "cpp.retry1.jsonl"])
        rows = store.Store(self.root).rows()
        abandoned = [r for r in rows if r["min_ms"] != r["min_ms"]]
        self.assertEqual(sorted((r["n"], r["transfers"]) for r in abandoned),
                         [(32, "both"), (32, "none"), (128, "both"), (128, "none")])
        self.assertEqual({r["reason"] for r in abandoned}, {"abandoned: hard-exit at " + hung["trial_id"]})
        self.assertEqual({r["algorithm"] for r in abandoned}, {"cash-karp-54"})
        self.assertEqual({r["states"] for r in abandoned}, {3})
        self.assertTrue(all(r["suite_rev"] for r in abandoned))
        # The rows the runner recorded before the hard exit stand, and the other build ran in the retry.
        finite = [r for r in rows if r["min_ms"] == 1.0]
        self.assertEqual(sorted((r["algorithm"], r["n"]) for r in finite if r["transfers"] == "both"),
                         [("cash-karp-54", 8), ("classical-rk4", 8), ("classical-rk4", 32), ("classical-rk4", 128)])
        retry = trials.read_jsonl(os.path.join(run.log_dir, "cpp.retry1.jsonl"))
        self.assertEqual([(t["algorithm"], t["n"]) for t in retry],
                         [("classical-rk4", 8), ("classical-rk4", 32), ("classical-rk4", 128)])

    def test_a_hard_exit_on_the_last_build_ends_the_package_without_a_retry(self):
        hung = self.line("classical-rk4", 8)
        status, run, calls, summary = self.run_bench(hung["trial_id"])
        self.assertEqual(status, 0)
        self.assertEqual([c["path"] for c in calls], ["cpp.jsonl"])
        rows = store.Store(self.root).rows()
        self.assertEqual({(r["algorithm"], r["min_ms"] == r["min_ms"]) for r in rows},
                         {("cash-karp-54", True), ("classical-rk4", False)})
        self.assertEqual({r["reason"] for r in rows if r["algorithm"] == "classical-rk4"},
                         {"abandoned: hard-exit at " + hung["trial_id"]})
        self.assertEqual(summary, [["cpp", "PARTIAL", "1 hard exit(s)", "3"]])

    def test_other_exit_codes_fail_the_package(self):
        hung = self.planned()[0]
        status, run, calls, summary = self.run_bench(hung["trial_id"], 2)
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary, [["cpp", "FAILED", "runner exit 2", "2"]])
        self.assertEqual(store.Store(self.root).rows(), [])

    def test_floor_reaches_the_runner(self):
        status, run, calls, summary = self.run_bench(None, 3, "--floor")
        self.assertEqual(calls[0]["argv"][-1], "--floor")
        self.assertEqual(WATCHDOG_EXIT_CODE, 3)


if __name__ == "__main__":
    unittest.main()
