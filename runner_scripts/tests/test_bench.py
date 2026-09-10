"""bench.py: flag resolution, the plan output, the continuation filters, the runner registry, and the watchdog hard-exit loop against a fake runner."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import bench  # noqa: E402
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
                         "ids": [t["trial_id"] + ":" + t["kind"] for t in trials]}}) + "\\n")
data = store.Store(plan["root"])
for t in trials:
    with open(path + ".progress", "w") as h:
        json.dump({{"trial_id": t["trial_id"], "started_utc": "2026-09-09T00:00:00Z"}}, h)
    if t["kind"] != "solve":
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
            self.assertEqual(trials.counts(rows)[0], {"solve": 2, "warm": 1, "optimize": 0})
        paths = bench.write_plan(os.path.join(self.tmp, "trials"), by_package)
        self.assertEqual(sorted(os.path.basename(p) for p in paths.values()),
                         ["cpp.jsonl", "jax.jsonl", "pytorch.jsonl"])
        back = trials.read_jsonl(paths["jax"])
        self.assertEqual([t["kind"] for t in back], ["warm", "solve", "solve"])
        self.assertEqual([t["n"] for t in back if t["kind"] == "solve"], [8, 32])
        self.assertEqual(set(back[0]), set(trials.TRIAL_KEYS))

    def test_resume_drops_trials_whose_rows_exist_and_no_overwrite_those_finite(self):
        by_package = self.plan("--set", "perf", "-p", "cpp", "-s", "lorenz", "-n", "8,32")
        cpp = by_package["cpp"]
        solves = [t for t in cpp if t["kind"] == "solve"]
        self.assertEqual(len(solves), 4)
        recorded = solves[0]
        nan_row = solves[1]
        partial = solves[2]
        data = store.Store(self.root)
        for transfers in ("both", "none"):
            spec = {f: recorded[f] for f in store.TRIAL_FIELDS}
            data.record(dict(spec, transfers=transfers, key=KEY, states=3, min_ms=1.0))
            spec = {f: nan_row[f] for f in store.TRIAL_FIELDS}
            data.record(dict(spec, transfers=transfers, key=KEY, states=3, min_ms=NAN, reason="error: x"))
        spec = {f: partial[f] for f in store.TRIAL_FIELDS}
        data.record(dict(spec, transfers="both", key=KEY, states=3, min_ms=2.0, finals="finals/p.parquet"))
        resumed = bench.continue_filter(cpp, KEY, self.root, resume=True)
        ids = {(t["trial_id"], t["kind"]) for t in resumed}
        self.assertNotIn((recorded["trial_id"], "solve"), ids)
        self.assertNotIn((nan_row["trial_id"], "solve"), ids)
        self.assertIn((partial["trial_id"], "solve"), ids)
        self.assertIn((solves[3]["trial_id"], "solve"), ids)
        # A partial trial runs its missing transfers alone and keeps asking finals once a row carries them.
        kept = {t["trial_id"]: t for t in resumed if t["kind"] == "solve"}
        self.assertEqual((kept[partial["trial_id"]]["transfers"], kept[partial["trial_id"]]["finals"]),
                         (["none"], True))
        self.assertEqual(kept[solves[3]["trial_id"]]["transfers"], ["both", "none"])
        fresh = bench.continue_filter(cpp, KEY, self.root, no_overwrite=True)
        ids = {(t["trial_id"], t["kind"]) for t in fresh}
        self.assertNotIn((recorded["trial_id"], "solve"), ids)
        self.assertIn((nan_row["trial_id"], "solve"), ids)
        self.assertIn((partial["trial_id"], "solve"), ids)
        kept = {t["trial_id"]: t for t in fresh if t["kind"] == "solve"}
        self.assertEqual(kept[nan_row["trial_id"]]["transfers"], ["both", "none"])
        # A leg name repeats across packages; a covered package's warm line is dropped while another's stays.
        both = self.plan("--set", "perf", "-p", "cpp,pytorch", "-s", "lorenz", "-g", "classical-rk4", "-n", "8")
        for t in both["cpp"]:
            if t["kind"] == "solve":
                spec = {f: t[f] for f in store.TRIAL_FIELDS}
                for transfers in ("both", "none"):
                    data.record(dict(spec, transfers=transfers, key=KEY, states=3, min_ms=1.0))
        mixed = bench.continue_filter(both["pytorch"] + both["cpp"], KEY, self.root, resume=True)
        self.assertEqual({t["package"] for t in mixed}, {"pytorch"})
        self.assertEqual([t["kind"] for t in mixed], ["warm", "solve"])
        # A trial that asks finals over rows without them runs its last transfers again.
        wants = dict(recorded, finals=True)
        again = bench.continue_filter([wants], KEY, self.root, resume=True)
        self.assertEqual([(t["transfers"], t["finals"]) for t in again], [(["none"], True)])
        for transfers in ("both", "none"):
            spec = {f: recorded[f] for f in store.TRIAL_FIELDS}
            data.record(dict(spec, transfers=transfers, key=KEY, states=3, min_ms=1.0, finals="finals/r.parquet"))
        self.assertEqual(bench.continue_filter([wants], KEY, self.root, resume=True), [])
        # Warm trials follow their leg: a fully recorded leg loses its warm trial.
        leg = recorded["leg"]
        other = [t for t in cpp if t["kind"] == "solve" and t["leg"] == leg and t is not recorded]
        for t in other:
            spec = {f: t[f] for f in store.TRIAL_FIELDS}
            for transfers in ("both", "none"):
                data.record(dict(spec, transfers=transfers, key=KEY, states=3, min_ms=1.0))
        resumed = bench.continue_filter(cpp, KEY, self.root, resume=True)
        self.assertNotIn(leg, {t["leg"] for t in resumed})
        self.assertIn("warm", {t["kind"] for t in resumed})
        self.assertEqual(bench.continue_filter(cpp, KEY, self.root), cpp)

    def test_plan_cli_writes_the_trial_files_and_prints_counts(self):
        out = subprocess.run([sys.executable, os.path.join(ROOT, "bench.py"), "plan", "--set", "perf",
                              "-p", "cpp", "-s", "lorenz", "-n", "8", "--allow-unknown-gpu"],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("cpp: 2 solve, 2 warm, 0 optimize trials in 2 legs", out.stdout)
        self.assertIn("lorenz/{}/classical-rk4/fixed/float32/n  1", out.stdout)
        self.assertIn("2 solve trials", out.stdout)
        written = [line for line in out.stdout.splitlines() if line.endswith("cpp.jsonl")]
        self.assertEqual(len(written), 1)
        path = os.path.join(ROOT, written[0])
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(len(trials.read_jsonl(path)), 4)
        os.remove(path)


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
        self.assertIn("--project=" + launch.julia_project(), julia_cpu)
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

    def test_the_julia_project_is_the_checkout_unless_JULIA_PROJECT_names_one(self):
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gpu"))
        import julia_driver
        saved = os.environ.pop("JULIA_PROJECT", None)
        try:
            self.assertEqual(launch.julia_project(), launch.REPO_ROOT)
            os.environ["JULIA_PROJECT"] = "/srv/GPUODEBenchmarks"
            self.assertEqual(launch.julia_project(), "/srv/GPUODEBenchmarks")
            self.assertIn("--project=/srv/GPUODEBenchmarks", launch.runner_command("julia_cpu", "x.jsonl").argv)
            self.assertEqual(julia_driver.julia_command()[-1], "--project=/srv/GPUODEBenchmarks")
        finally:
            os.environ.pop("JULIA_PROJECT", None)
            if saved is not None:
                os.environ["JULIA_PROJECT"] = saved

    def test_a_shared_project_whose_julia_sources_differ_stops_the_julia_runners(self):
        other = tempfile.mkdtemp(prefix="julia_project_")
        self.addCleanup(shutil.rmtree, other, True)
        for name in launch._julia_source_files(ROOT):
            target = os.path.join(other, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(os.path.join(ROOT, name), target)
        saved = os.environ.pop("JULIA_PROJECT", None)
        os.environ["JULIA_PROJECT"] = other
        try:
            self.assertEqual(launch.julia_sources_differing(other), [])
            self.assertEqual(launch.check_julia_project(), other)
            self.assertIn("--project=" + other, launch.runner_command("julia_cpu", "x.jsonl").argv)
            with open(os.path.join(other, "runner_scripts", "julia_systems.jl"), "a") as handle:
                handle.write("# edited\n")
            self.assertEqual(launch.julia_sources_differing(other), ["runner_scripts/julia_systems.jl"])
            with self.assertRaises(SystemExit) as caught:
                launch.runner_command("julia_gpu", "x.jsonl")
            self.assertIn("julia_systems.jl", str(caught.exception))
            self.assertIn("Unset JULIA_PROJECT", str(caught.exception))
            launch.runner_command("cubie", "x.jsonl")
        finally:
            os.environ.pop("JULIA_PROJECT", None)
            if saved is not None:
                os.environ["JULIA_PROJECT"] = saved


class HardExitTests(unittest.TestCase):
    """The runner loop against a fake runner: a hard exit abandons the leg's higher ordinals and the rest re-runs."""

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

    def solves(self, run):
        return [t for t in trials.read_jsonl(os.path.join(run.log_dir, "cpp.jsonl")) if t["kind"] == "solve"]

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

    def test_a_hard_exit_abandons_the_higher_ordinals_and_reruns_the_rest(self):
        # Hang the classical-rk4 leg at n = 32 (ordinal 1).
        first = bench.plan_trials(bench.resolve(bench.parse_args(
            ["run", "--set", "perf", "-p", "cpp", "-s", "lorenz", "-n", "8,32,128"])), KEY, self.root)["cpp"]
        hung = [t for t in first if t["kind"] == "solve" and t["algorithm"] == "classical-rk4" and t["n"] == 32][0]
        status, run, calls, summary = self.run_bench(hung["trial_id"])
        self.assertEqual(status, 0)
        self.assertEqual(summary, [["cpp", "PARTIAL", "1 hard exit(s)", "0"]])
        self.assertEqual([c["path"] for c in calls], ["cpp.jsonl", "cpp.retry1.jsonl"])
        rows = store.Store(self.root).rows()
        abandoned = [r for r in rows if r["min_ms"] != r["min_ms"]]
        self.assertEqual(sorted((r["n"], r["transfers"]) for r in abandoned),
                         [(32, "both"), (32, "none"), (128, "both"), (128, "none")])
        self.assertEqual({r["reason"] for r in abandoned}, {"abandoned: hard-exit at ordinal 1"})
        self.assertEqual({r["algorithm"] for r in abandoned}, {"classical-rk4"})
        self.assertEqual({r["states"] for r in abandoned}, {3})
        self.assertTrue(all(r["suite_rev"] for r in abandoned))
        # The rows the runner recorded before the hard exit stand, and the other leg ran in the retry.
        finite = [r for r in rows if r["min_ms"] == 1.0]
        self.assertEqual(sorted((r["algorithm"], r["n"]) for r in finite if r["transfers"] == "both"),
                         [("cash-karp-54", 8), ("cash-karp-54", 32), ("cash-karp-54", 128), ("classical-rk4", 8)])
        retry = trials.read_jsonl(os.path.join(run.log_dir, "cpp.retry1.jsonl"))
        self.assertEqual({t["algorithm"] for t in retry}, {"cash-karp-54"})
        self.assertEqual([t["kind"] for t in retry], ["warm", "solve", "solve", "solve"])

    def test_a_hard_exit_on_the_last_leg_ends_the_package_without_a_retry(self):
        first = bench.plan_trials(bench.resolve(bench.parse_args(
            ["run", "--set", "perf", "-p", "cpp", "-s", "lorenz", "-n", "8,32,128"])), KEY, self.root)["cpp"]
        hung = [t for t in first if t["kind"] == "solve" and t["algorithm"] == "cash-karp-54" and t["n"] == 8][0]
        status, run, calls, summary = self.run_bench(hung["trial_id"])
        self.assertEqual(status, 0)
        self.assertEqual([c["path"] for c in calls], ["cpp.jsonl"])
        rows = store.Store(self.root).rows()
        self.assertEqual({(r["algorithm"], r["min_ms"] == r["min_ms"]) for r in rows},
                         {("classical-rk4", True), ("cash-karp-54", False)})
        self.assertEqual({r["reason"] for r in rows if r["algorithm"] == "cash-karp-54"},
                         {"abandoned: hard-exit at ordinal 0"})
        self.assertEqual(summary, [["cpp", "PARTIAL", "1 hard exit(s)", "3"]])

    def test_other_exit_codes_fail_the_package(self):
        first = bench.plan_trials(bench.resolve(bench.parse_args(
            ["run", "--set", "perf", "-p", "cpp", "-s", "lorenz", "-n", "8,32,128"])), KEY, self.root)["cpp"]
        hung = [t for t in first if t["kind"] == "solve"][0]
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
