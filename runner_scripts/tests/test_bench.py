"""bench.py: flag resolution, the plan output, the -n check over heterogeneous grids, the completeness-aware continuation filters, the runner registry, the cubie precompile step and the watchdog hard-exit loop against a fake runner."""

import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import bench  # noqa: E402
import clocks  # noqa: E402
import completeness  # noqa: E402
import cubie_adapter  # noqa: E402
import launch  # noqa: E402
import sets  # noqa: E402
import store  # noqa: E402
import trials  # noqa: E402
from protocol import WATCHDOG_EXIT_CODE  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
NAN = float("nan")

# A runner that records every solve trial it reaches, exits 3 (once) while a chosen trial is in progress and lists plan.json's crashed builds in its progress file; under --precompile it logs the call and exits plan.json's precompile_code.
FAKE_RUNNER = '''
import json, os, sys, time
from datetime import datetime, timezone
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
if "--precompile" in argv:
    sys.exit(plan.get("precompile_code", 0))
data = store.Store(plan["root"])
for t in trials:
    with open(path + ".progress", "w") as h:
        json.dump({{"trial_id": t["trial_id"], "stage": "solve", "started_utc": "2026-09-09T00:00:00Z",
                   "failed": plan.get("failed", [])}}, h)
    if not t["transfers"]:
        continue
    if t["trial_id"] == plan.get("trial_id") and not plan.get("done"):
        json.dump(dict(plan, done=True), open(marker, "w"))
        sys.exit(plan.get("code", 3))
    spec = {{f: t[f] for f in store.TRIAL_FIELDS}}
    started = datetime.now(timezone.utc)
    time.sleep(0.003)
    ended = datetime.now(timezone.utc)
    data.record_batch([dict(spec, transfers=x, key=plan["key"], states=3, min_ms=1.0, samples_ms=[2.0, 1.0],
                            timed_start_utc=started, timed_end_utc=ended)
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
        # An unreadable finals file reruns the whole trial; readable finals rerun the missing transfers alone.
        self.assertEqual((resumed[partial["trial_id"]]["transfers"], resumed[partial["trial_id"]]["finals"]),
                         (["both", "none"], True))
        self.assertEqual(resumed[absent["trial_id"]]["transfers"], ["both", "none"])
        spec = {f: partial[f] for f in store.TRIAL_FIELDS}
        readable = data.record_finals(dict(spec, key=KEY), np.zeros((8, 3)), np.full(8, 1.0))
        data.record(dict(spec, transfers="both", key=KEY, states=3, min_ms=2.0, finals=readable))
        resumed = {t["trial_id"]: t for t in bench.continue_filter(cpp, KEY, self.root, resume=True)}
        self.assertEqual((resumed[partial["trial_id"]]["transfers"], resumed[partial["trial_id"]]["finals"]),
                         (["none"], True))
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
        # A trial that asks finals over rows without them runs whole, so its finals and timings share an execution.
        wants = dict(recorded, finals=True)
        again = bench.continue_filter([wants], KEY, self.root, resume=True)
        self.assertEqual([(t["transfers"], t["finals"]) for t in again], [(["both", "none"], True)])
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
                             [(True, True, ["both", "none"], False, ["golden_grid", "perf", "states"])], argv)

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
        row = dict(spec, transfers=transfers, key=KEY, states=3, min_ms=1.0)
        row.update(values)
        return self.data.record(row)

    def kept(self, trial_list, **kw):
        return {t["trial_id"]: t for t in bench.continue_filter(trial_list, KEY, self.root, **kw)}

    @staticmethod
    def day(d):
        return "2026-09-{0:02d}T00:00:00Z".format(d)

    def optimized(self, trial, runs=71680, timeout=False, on=1):
        """A kernel record stamped on day `on`."""
        with mock.patch.object(cubie_adapter, "_stamp", return_value=self.day(on)):
            if timeout:
                return cubie_adapter.record_optimize_timeout(trial, KEY, self.root)
            return cubie_adapter.record_optimized(trial, KEY, FakeOptimizeResult(runs), root=self.root)

    def legacy_records(self, trial, rows):
        """An optimize.csv in the per-solve layout with its source column: (per, n, settings, source) rows for the trial's kernel."""
        kernel = cubie_adapter.kernel_ident(trial, KEY)
        text = ("package,key,problem,states,precision,algorithm,controller,gains,stepping,per,n,duration,source,"
                "label,best_ms,blocksize,resident_blocks,settings,recorded_utc\n")
        for per, n, settings, source in rows:
            text += ",".join([kernel["package"], kernel["key"], kernel["problem"], kernel["states"],
                              kernel["precision"], kernel["algorithm"], kernel["controller"], '"{}"',
                              kernel["stepping"], per, n, "1", source, "bs", "1.0", "64", "", '"' + settings + '"',
                              "2026-09-01T00:00:00Z"]) + "\n"
        with open(cubie_adapter.optimize_path("cubie", KEY, self.root), "w", newline="", encoding="utf-8") as handle:
            handle.write(text)

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
        # A build time lacking on one row reruns the whole trial: its timing and build time come from one execution.
        self.assertEqual(kept[states[1]["trial_id"]]["transfers"], ["both", "none"])
        self.assertNotIn(states[2]["trial_id"], kept)
        audits = completeness.audit(states, KEY, self.data, "resume")
        self.assertEqual(audits[states[0]["trial_id"]].reasons(), ["build:both", "build:none"])
        self.assertEqual(audits[states[1]["trial_id"]].reasons(), ["build:none"])
        self.assertTrue(audits[states[2]["trial_id"]].complete())
        # A cold line recorded NaN wants no build time: complete under --resume, its rows under --no-overwrite.
        for transfers in ("both", "none"):
            self.record(states[3], transfers, min_ms=NAN, reason="error: BuildError: out of memory")
        self.assertNotIn(states[3]["trial_id"], self.kept(states, resume=True))
        self.assertEqual(completeness.audit(states, KEY, self.data, "no_overwrite")[states[3]["trial_id"]].reasons(),
                         ["row:both", "row:none"])
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
        # Finals wanted over rows recorded without them: the whole trial runs again.
        perf = self.plan("--set", "perf", "-p", "cpp", "-s", "lorenz", "-g", "classical-rk4", "-n", "131072",
                         "--mode", "fixed")["cpp"]
        point = perf[0]
        self.assertTrue(point["finals"])
        for transfers in ("both", "none"):
            self.record(point, transfers)
        kept = self.kept(perf, resume=True)
        self.assertEqual((kept[point["trial_id"]]["transfers"], kept[point["trial_id"]]["finals"]),
                         (["both", "none"], True))
        self.assertEqual(completeness.audit(perf, KEY, self.data, "resume")[point["trial_id"]].reasons(), ["finals"])
        # Rows all NaN carry no finals: complete under --resume, the row alone under --no-overwrite.
        timed_out = self.plan("--set", "golden_grid", "-p", "cpp", "-s", "lorenz", "-g", "classical-rk4",
                              "--dt", "0.125")["cpp"]
        spec = {f: timed_out[0][f] for f in store.TRIAL_FIELDS}
        self.data.record(dict(spec, transfers="none", key=KEY, states=3, min_ms=NAN,
                              reason="timeout: 130000.0 ms exceeded the 120 s cap"))
        self.assertEqual(self.kept(timed_out, resume=True), {})
        self.assertTrue(completeness.audit(timed_out, KEY, self.data)[timed_out[0]["trial_id"]].complete())
        self.assertEqual(completeness.audit(timed_out, KEY, self.data, "no_overwrite")[timed_out[0]["trial_id"]].reasons(),
                         ["row:none"])

    def test_an_optimize_record_must_exist_for_the_kernel(self):
        perf = self.plan("--set", "perf", "-p", "cubie", "-s", "lorenz", "-g", "tsit5", "--mode", "fixed",
                         "-n", "8,32")["cubie"]
        self.assertEqual([(t["n"], t["optimize"]) for t in perf], [(8, True), (32, True)])
        for trial in perf:
            for transfers in ("both", "none"):
                self.record(trial, transfers, recorded_utc=self.day(2))
        # Complete rows, no record: every transfers of both lines runs again.
        kept = self.kept(perf, resume=True)
        self.assertEqual(sorted(kept), sorted(t["trial_id"] for t in perf))
        self.assertEqual({tuple(t["transfers"]) for t in kept.values()}, {("both", "none")})
        audits = completeness.audit(perf, KEY, self.data, "resume")
        self.assertEqual(audits[perf[0]["trial_id"]].reasons(), ["optimize:absent"])
        for trial in perf:
            self.optimized(trial, runs=trial["n"])
        self.assertEqual(self.kept(perf, resume=True), {})
        self.assertEqual(self.kept(perf, no_overwrite=True), {})
        # The two lines share a kernel: one record, from either line, serves both.
        cubie_adapter.clear_optimized("cubie", KEY, "tsit5", "lorenz", root=self.root)
        self.optimized(perf[0])
        self.assertEqual(self.kept(perf, resume=True), {})
        self.assertEqual(len(cubie_adapter.optimize_rows("cubie", KEY, self.root)), 1)
        # A timed-out optimize stands under --resume; --no-overwrite reruns every line of its kernel.
        self.optimized(perf[1], timeout=True)
        self.assertEqual(self.kept(perf, resume=True), {})
        self.assertEqual(sorted(self.kept(perf, no_overwrite=True)), sorted(t["trial_id"] for t in perf))
        self.assertEqual(completeness.audit(perf, KEY, self.data, "no_overwrite")[perf[1]["trial_id"]].reasons(),
                         ["optimize:timeout"])
        # A record carrying a source column from an earlier suite stands, whatever hash it names.
        self.legacy_records(perf[0], [("kernel", "71680", '{""blocksize"": 256}', "0" * 16)])
        for mode in (None, "resume", "no_overwrite"):
            self.assertTrue(all(m.complete() for m in completeness.audit(perf, KEY, self.data, mode).values()), mode)
        # Nothing about the source is asked or checked on the way.
        self.assertFalse(hasattr(bench, "source_hashes"))
        self.assertFalse(hasattr(cubie_adapter, "source_hashes"))
        self.assertNotIn("sources", inspect.signature(completeness.audit).parameters)
        self.assertNotIn("sources", inspect.signature(bench.continue_filter).parameters)

    def test_a_timed_out_optimize_stands_under_resume_and_reruns_under_no_overwrite(self):
        perf = self.plan("--set", "perf", "-p", "cubie", "-s", "lorenz", "-g", "tsit5", "--mode", "fixed",
                         "-n", "8,32")["cubie"]
        self.optimized(perf[0], timeout=True)
        # No rows: both lines run; under --resume they solve without optimizing the timed-out kernel.
        self.assertEqual({t["optimize"] for t in self.kept(perf, resume=True).values()}, {False})
        self.assertEqual({t["optimize"] for t in self.kept(perf, no_overwrite=True).values()}, {True})
        self.assertEqual({t["optimize"] for t in self.kept(perf).values()}, {True})

    def test_each_kernel_has_its_own_optimize_record(self):
        golden = self.plan("--set", "golden_grid", "-p", "cubie", "-s", "lorenz", "-g", "backwards_euler",
                           "--dt", "0.5,0.25")["cubie"]
        self.assertEqual([(t["dt"], t["optimize"]) for t in golden], [(0.5, True), (0.25, True)])
        for trial in golden:
            spec = {f: trial[f] for f in store.TRIAL_FIELDS}
            relative = self.data.record_finals(dict(spec, key=KEY), np.zeros((131072, 3)), np.full(131072, 1.0))
            self.record(trial, "none", finals=relative, recorded_utc=self.day(2))
        self.optimized(golden[0])
        # The 0.5 kernel stands with its record at any batch; the 0.25 kernel runs again.
        self.assertEqual(list(self.kept(golden, resume=True)), [golden[1]["trial_id"]])
        self.assertEqual(completeness.summary(completeness.audit(golden, KEY, self.data, "resume")),
                         {"optimize:absent": 1})

    def test_an_explicit_fixed_step_build_shares_one_optimize_record_across_dt(self):
        golden = self.plan("--set", "golden_grid", "-p", "cubie", "-s", "lorenz", "-g", "classical-rk4",
                           "--dt", "0.5,0.25")["cubie"]
        for trial in golden:
            spec = {f: trial[f] for f in store.TRIAL_FIELDS}
            relative = self.data.record_finals(dict(spec, key=KEY), np.zeros((131072, 3)), np.full(131072, 1.0))
            self.record(trial, "none", finals=relative, recorded_utc=self.day(2))
        self.optimized(golden[0])
        self.assertEqual(self.kept(golden, resume=True), {})
        self.assertEqual(trials.optimizes_of(golden), 1)

    def test_a_recorded_row_stands_whatever_was_optimized_or_changed_after_it(self):
        # Legacy per-solve records with their rows: the per-solve rows are dropped, the timings stay.
        perf = self.plan("--set", "perf", "-p", "cubie", "-s", "lorenz", "-g", "tsit5", "--mode", "fixed",
                         "-n", "8,32")["cubie"]
        self.legacy_records(perf[0], [("solve", "8", '{""blocksize"": 256}', "S"),
                                      ("solve", "32", '{""blocksize"": 32}', "S")])
        for trial in perf:
            for transfers in ("both", "none"):
                self.record(trial, transfers, recorded_utc=self.day(2))
        # No kernel record: both lines run again, for the record alone.
        self.assertEqual(sorted(self.kept(perf, resume=True)), sorted(t["trial_id"] for t in perf))
        self.assertEqual({tuple(m.reasons()) for m in completeness.audit(perf, KEY, self.data, "resume").values()},
                         {("optimize:absent",)})
        # A kernel record later than every row: the rows stand under every mode, no row is dated.
        self.optimized(perf[0], on=3)
        for mode in (None, "resume", "no_overwrite"):
            audits = completeness.audit(perf, KEY, self.data, mode)
            self.assertTrue(all(m.complete() for m in audits.values()), mode)
            self.assertEqual({r for m in audits.values() for r in m.reasons() if r.startswith("stale")}, set())
        self.assertEqual(self.kept(perf, resume=True), {})
        self.assertEqual(self.kept(perf, no_overwrite=True), {})
        # A re-optimize after the rows changes nothing about them either.
        self.optimized(perf[1], on=6)
        self.assertEqual(self.kept(perf, resume=True), {})
        self.assertEqual(self.kept(perf, no_overwrite=True), {})
        # A NaN row recorded before the record is kept under --resume and rerun alone under --no-overwrite.
        self.record(perf[0], "both", min_ms=NAN, reason="abandoned: hard-exit at x", recorded_utc=self.day(1))
        self.assertEqual(self.kept(perf, resume=True), {})
        self.assertEqual(self.kept(perf, no_overwrite=True)[perf[0]["trial_id"]]["transfers"], ["both"])
        self.assertEqual(completeness.audit(perf, KEY, self.data, "no_overwrite")[perf[0]["trial_id"]].reasons(),
                         ["row:both"])
        # A timed-out record after the rows: they stand under --resume; --no-overwrite reruns the kernel's lines.
        self.optimized(perf[1], on=7, timeout=True)
        self.assertEqual(self.kept(perf, resume=True), {})
        self.assertEqual(sorted(self.kept(perf, no_overwrite=True)), sorted(t["trial_id"] for t in perf))
        for gone in ("predates", "optimize_systems"):
            self.assertFalse(hasattr(completeness, gone), gone)
        self.assertFalse(hasattr(completeness.Missing(perf[0], False), "stale"))


class FakeOptimizeResult:
    """A winner timed over `runs` runs at the whole duration."""

    class Best:
        label, best_ms, blocksize, resident_blocks = "state=shared @bs128", 1.0, 128, 2

    best = Best()
    applied_settings = {"blocksize": 128}

    def __init__(self, runs):
        self.runs = runs
        self.duration = 1.0


class LaunchTests(unittest.TestCase):
    def test_every_package_has_a_runner_taking_the_trial_file(self):
        self.assertEqual(set(launch.RUNNERS), set(store.PACKAGES))
        for package in store.PACKAGES:
            command = launch.runner_command(package, "trials/k/x.jsonl")
            self.assertEqual(command.argv[-2:], ["--trials", "trials/k/x.jsonl"], package)
            self.assertEqual(command.ok, (0,))
        for package in launch.CUBIE_PACKAGES:
            precompile = launch.precompile_command(package, "trials/k/x.jsonl")
            self.assertEqual(precompile.argv[:2], launch.runner_command(package, "x.jsonl").argv[:2])
            self.assertEqual(precompile.argv[2:], ["--trials", "trials/k/x.jsonl", "--precompile", "--jobs", "4",
                                                   "--per-worker", "8", "--memory-gb", "6"])
            self.assertEqual(precompile.env, {"CUBIE_MAX_CACHE_ENTRIES": "0"})
            self.assertEqual(precompile.label, package + " precompile")
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


class FakeSampler:
    """Stands in for the nvidia-smi sampler: a thread writes a 25 Hz log of one SM reading and reason mask until stopped."""

    def __init__(self, sm=2310, reasons=0):
        self.sm, self.reasons = sm, reasons
        self.stopped = threading.Event()
        self.thread = None

    def start_monitor(self, guard, csv_path, sample_ms=40):
        guard.csv = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        def write():
            with open(csv_path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(clocks.HEADER + "\n")
                while not self.stopped.is_set():
                    stamp = datetime.now(timezone.utc).strftime(clocks.UTC_STAMP)
                    handle.write("{0},{1},10251,60,170,100,0x{2:016x}\n".format(stamp, self.sm, self.reasons))
                    handle.flush()
                    self.stopped.wait(sample_ms / 1000.0)
        self.thread = threading.Thread(target=write, daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def stop_monitor(self, guard):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(5)


class HardExitTests(unittest.TestCase):
    """The runner loop against a fake runner under a fake lock and sampler: a hard exit abandons the harder runs of the family and the rest re-runs; every row records the lock and the clocks its window showed; drift fails the run."""

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
        # A Run exports its context into the environment; the tests leave none behind.
        context = {name: os.environ.pop(name, None)
                   for name in (store.RUN_ENV, store.DRIVER_ENV, store.CLOCK_LOCK_ENV, store.OVERWRITE_ENV)}

        def restore():
            for name, value in context.items():
                os.environ.pop(name, None)
                if value is not None:
                    os.environ[name] = value
        self.addCleanup(restore)
        # The card's conf row, the lock and the sampler are faked: no nvidia-smi is driven.
        self.real_configure = bench.configure_clocks
        self.real_lock = clocks.ClockGuard.lock
        self.resets = []
        self.sampler = FakeSampler()

        def lock(guard):
            guard.locked = True
            return True
        for target, name, value in (
                (bench, "configure_clocks", lambda key, explicit="": ("2310", None)),
                (clocks.ClockGuard, "lock", lock),
                (clocks.ClockGuard, "reset", lambda guard: self.resets.append(guard.locked)),
                (clocks.ClockGuard, "start_monitor",
                 lambda guard, path, sample_ms=40: self.sampler.start_monitor(guard, path, sample_ms)),
                (clocks.ClockGuard, "stop_monitor", lambda guard: self.sampler.stop_monitor(guard))):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_bench(self, hung=None, code=3, *argv, lock="", failed=(), package="cpp", precompile_code=0):
        args = bench.parse_args(["run", "--set", "perf", "-p", package, "-s", "lorenz", "-n", "8,32,128",
                                 "--cooldown", "0"] + ([lock] if lock else []) + list(argv))
        run = bench.Run(args, bench.resolve(args), key=KEY, data_root=self.root, logs_root=self.logs)
        with open(os.path.join(run.log_dir, "plan.json"), "w") as handle:
            json.dump({"trial_id": hung, "code": code, "root": self.root, "key": KEY, "failed": list(failed),
                       "precompile_code": precompile_code}, handle)
        status = run.execute()
        with open(os.path.join(run.log_dir, "calls.jsonl")) as handle:
            calls = [json.loads(line) for line in handle]
        with open(run.summary) as handle:
            summary = [line.rstrip("\n").split("\t") for line in handle]
        return status, run, calls, summary

    def planned(self, package="cpp", *argv):
        return bench.plan_trials(bench.resolve(bench.parse_args(
            ["run", "--set", "perf", "-p", package, "-s", "lorenz", "-n", "8,32,128"] + list(argv))),
            KEY, self.root)[package]

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
        # Every row names the run it was recorded in and the lock, and the run's clock log sits in the mirror.
        self.assertEqual(run.run, os.path.basename(run.log_dir))
        self.assertTrue(run.run.startswith(KEY + "_"))
        self.assertEqual({r["run"] for r in rows}, {run.run})
        self.assertEqual({r["clock_lock_mhz"] for r in rows}, {2310})
        self.assertEqual({r["driver"] for r in rows}, {run.driver})
        self.assertEqual(run.clocks_csv, os.path.join(self.root, "clocks", run.run + ".csv"))
        self.assertNotIn(store.RUN_ENV, {k for k in os.environ if k == "no-such"})
        self.assertEqual(os.environ[store.OVERWRITE_ENV], "1")
        with open(os.path.join(run.log_dir, "run_manifest.txt")) as handle:
            manifest = handle.read()
        self.assertIn("run=" + run.run + "\n", manifest)
        self.assertIn("clocks=locked SM=2310\n", manifest)
        self.assertTrue(all(r["timed_start_utc"] <= r["timed_end_utc"] <= r["recorded_utc"] for r in rows))
        # The sampler covered every timing window, so each row carries the clocks it showed.
        self.assertEqual(run.clock_lines, ["cpp: 12 rows annotated, SM median 2310 MHz, low 2310 MHz, 0 drifted"])
        self.assertEqual({(r["clock_sm_mhz"], r["clock_sm_min_mhz"], r["clock_throttled"]) for r in rows},
                         {(2310.0, 2310.0, 0)})
        # The lock was released once, after the run.
        self.assertEqual(self.resets, [True])

    def test_a_row_below_the_lock_or_throttled_fails_the_run(self):
        self.sampler.sm = 2200
        status, run, calls, summary = self.run_bench()
        self.assertEqual(status, 1)
        self.assertEqual(summary, [["cpp", "OK", "-", "0"]])
        self.assertEqual(run.clock_failures, 1)
        self.assertEqual(run.clock_lines, ["cpp: 12 rows annotated, SM median 2200 MHz, low 2200 MHz, 12 drifted"])
        rows = store.Store(self.root).rows()
        self.assertEqual({r["clock_sm_min_mhz"] for r in rows}, {2200.0})
        # A window within the tolerance passes; a throttle reason in it fails the run.
        self.sampler = FakeSampler(sm=2300)
        status, run, _, _ = self.run_bench()
        self.assertEqual((status, run.clock_failures), (0, 0))
        self.sampler = FakeSampler(sm=2310, reasons=0x4)
        status, run, _, _ = self.run_bench()
        self.assertEqual((status, run.clock_failures), (1, 1))
        self.assertTrue(run.clock_lines[0].endswith("12 drifted"), run.clock_lines[0])

    def test_the_clock_lock_is_released_when_planning_fails(self):
        args = bench.parse_args(["run", "--set", "perf", "-p", "cpp", "-s", "lorenz", "-n", "8", "--cooldown", "0"])
        run = bench.Run(args, bench.resolve(args), key=KEY, data_root=self.root, logs_root=self.logs)
        self.assertTrue(run.clocks.locked)
        released = []
        run.clocks.reset = lambda: released.append("reset")
        run.clocks.stop_monitor = lambda: released.append("stop")
        # A failure while planning still reaches the guard's reset.
        with mock.patch.object(bench, "plan_trials", side_effect=RuntimeError("no store")):
            with self.assertRaises(RuntimeError):
                run.execute()
        self.assertEqual(released, ["stop", "reset"])
        self.assertEqual(store.Store(self.root).rows(), [])

    def test_a_sampler_that_dies_at_once_allows_the_run_and_releases_the_lock(self):
        with mock.patch.object(clocks.ClockGuard, "start_monitor", return_value=False):
            status, _, _, _ = self.run_bench()
        self.assertEqual(status, 0)
        self.assertEqual(self.resets, [True])
        rows = store.Store(self.root).rows()
        self.assertEqual(len(rows), 12)
        self.assertEqual({r["min_ms"] for r in rows}, {1.0})
        self.assertTrue(all(np.isnan(r[field]) for r in rows for field in ("clock_sm_mhz", "clock_sm_min_mhz")))
        self.assertTrue(all(r["clock_throttled"] is None for r in rows))

    def test_a_run_that_cannot_lock_refuses_to_start(self):
        # There is no flag that runs unlocked.
        with self.assertRaises(SystemExit):
            bench.parse_args(["run", "--set", "perf", "--no-lock-clocks"])
        self.assertNotIn("--no-lock-clocks", bench.__doc__)
        with mock.patch.object(bench, "configure_clocks", self.real_configure):
            with self.assertRaises(SystemExit) as caught:
                self.run_bench()
        self.assertIn("No clock target for 'RTX-4070-SUPER'", str(caught.exception))
        self.assertNotIn("--no-lock-clocks", str(caught.exception))
        self.assertNotIn(store.RUN_ENV, os.environ)
        self.assertEqual(store.Store(self.root).rows(), [])
        self.assertEqual(self.resets, [])
        # An explicit target from a shell that is not elevated refuses the same way.
        for patcher in (mock.patch.object(clocks, "is_admin", lambda: False),
                        mock.patch.object(clocks, "supported", lambda kind, mhz: True)):
            patcher.start()
            self.addCleanup(patcher.stop)
        with mock.patch.object(bench, "configure_clocks", self.real_configure):
            with mock.patch.object(clocks.ClockGuard, "lock", self.real_lock):
                with self.assertRaises(SystemExit) as caught:
                    self.run_bench(lock="--lock-clocks=2310")
        self.assertIn("elevated", str(caught.exception))
        self.assertEqual(self.resets, [])

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

    def test_a_cubie_package_precompiles_its_kernels_then_runs_a_fresh_runner_per_part_of_whole_families(self):
        two = ("-g", "cash-karp-54,classical-rk4")
        cubie_lines = self.planned("cubie", *two)
        families = {trials.family_key(t) for t in cubie_lines}
        # Each family's n sweep shares one kernel, so a budget of one kernel gives one part per family.
        parts = trials.family_parts(cubie_lines, 1)
        self.assertEqual(len(parts), len(families))
        self.assertEqual([t for part in parts for t in part], cubie_lines)
        self.assertEqual(trials.family_parts(cubie_lines, len(families)), [cubie_lines])
        self.assertEqual(trials.family_parts(cubie_lines), [cubie_lines])
        # A family of several kernels stays whole: the part closes at the boundary past the budget.
        tols = bench.plan_trials(bench.resolve(bench.parse_args(
            ["run", "--set", "golden_grid", "-p", "cubie", "-s", "lorenz", "-g", "cash-karp-54",
             "--controller", "default"])), KEY, self.root)["cubie"]
        self.assertEqual(len({trials.family_key(t) for t in tols}), 1)
        self.assertGreater(len({trials.kernel_key(t) for t in tols}), 1)
        self.assertEqual(trials.family_parts(tols, 1), [tols])
        # Only the cubie packages restart.
        self.assertEqual(sorted(launch.RESTART_KERNELS), ["cubie", "cubie_mlir"])
        self.assertEqual(set(launch.RESTART_KERNELS.values()), {8})
        self.assertIsNone(launch.RESTART_KERNELS.get("cpp"))
        saved = launch.RUNNERS["cubie"]
        launch.RUNNERS["cubie"] = lambda: [sys.executable, self.runner]
        self.addCleanup(launch.RUNNERS.__setitem__, "cubie", saved)
        with mock.patch.dict(launch.RESTART_KERNELS, {"cubie": 1}):
            status, run, calls, summary = self.run_bench(None, 3, *two, package="cubie")
        self.assertEqual(status, 0)
        # The precompile pass takes the whole file first, with the worker geometry.
        self.assertEqual(calls[0]["path"], "cubie.jsonl")
        self.assertEqual(calls[0]["argv"][-7:], ["--precompile", "--jobs", "4", "--per-worker", "1", "--memory-gb", "6"])
        self.assertEqual([c["path"] for c in calls[1:]],
                         ["cubie.part{0}.jsonl".format(n + 1) for n in range(len(parts))])
        self.assertTrue(all("--precompile" not in c["argv"] for c in calls[1:]))
        self.assertEqual([t["trial_id"] for part in parts for t in part],
                         [i for c in calls[1:] for i in c["ids"]])
        self.assertEqual(summary, [["cubie", "OK", "-", "0"]])
        self.assertEqual(len(store.Store(self.root).rows()), 2 * len(cubie_lines))
        self.assertEqual(calls[0]["ids"], [t["trial_id"] for t in cubie_lines])
        # A hard exit retries within its part, and the later parts still run.
        shutil.rmtree(self.root, ignore_errors=True)
        hung = [t for t in cubie_lines if t["algorithm"] == "cash-karp-54" and t["n"] == 32][0]
        with mock.patch.dict(launch.RESTART_KERNELS, {"cubie": 1}):
            status, run, calls, summary = self.run_bench(hung["trial_id"], 3, *two, package="cubie")
        self.assertEqual([c["path"] for c in calls[1:]],
                         ["cubie.part{0}.jsonl".format(n + 1) for n in range(len(parts))])
        self.assertEqual(summary, [["cubie", "PARTIAL", "1 hard exit(s)", "0"]])
        abandoned = [r for r in store.Store(self.root).rows() if r["min_ms"] != r["min_ms"]]
        self.assertEqual(sorted((r["n"], r["transfers"]) for r in abandoned),
                         [(32, "both"), (32, "none"), (128, "both"), (128, "none")])
        self.assertEqual({r["algorithm"] for r in abandoned}, {"cash-karp-54"})

    def test_a_failed_precompile_pass_still_runs_the_runners(self):
        saved = launch.RUNNERS["cubie"]
        launch.RUNNERS["cubie"] = lambda: [sys.executable, self.runner]
        self.addCleanup(launch.RUNNERS.__setitem__, "cubie", saved)
        status, run, calls, summary = self.run_bench(None, 3, package="cubie", precompile_code=1)
        self.assertEqual(status, 0)
        self.assertEqual(calls[0]["path"], "cubie.jsonl")
        self.assertIn("--precompile", calls[0]["argv"])
        self.assertTrue(all("--precompile" not in c["argv"] and c["path"].startswith("cubie.part") for c in calls[1:]))
        self.assertEqual(summary, [["cubie", "OK", "-", "0"]])
        # A package without a precompile pass runs its runner at once.
        status, run, calls, summary = self.run_bench(precompile_code=1)
        self.assertEqual(status, 0)
        self.assertEqual(summary, [["cpp", "OK", "-", "0"]])
        self.assertEqual(calls[-1]["path"], "cpp.jsonl")
        self.assertTrue(all("--precompile" not in c["argv"] for c in calls if c["path"] == "cpp.jsonl"))

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

    def test_a_build_crashed_before_a_hard_exit_fails_the_package_after_the_relaunch(self):
        # The relaunch still runs the rest; the crashed build fails the package.
        hung = self.line("cash-karp-54", 32)
        status, run, calls, summary = self.run_bench(hung["trial_id"], 3, failed=["lorenz/{}/float32/euler/fixed/{}"])
        self.assertEqual(status, 1)
        self.assertEqual([c["path"] for c in calls], ["cpp.jsonl", "cpp.retry1.jsonl"])
        self.assertEqual(summary, [["cpp", "FAILED",
                                    "1 hard exit(s); crashed before a hard exit: lorenz/{}/float32/euler/fixed/{}", "0"]])
        finite = [r for r in store.Store(self.root).rows() if r["min_ms"] == 1.0]
        self.assertEqual(sorted((r["algorithm"], r["n"]) for r in finite if r["transfers"] == "both"),
                         [("cash-karp-54", 8), ("classical-rk4", 8), ("classical-rk4", 32), ("classical-rk4", 128)])
        # A hard exit on the last build ends the package the same way, with the watchdog code.
        shutil.rmtree(self.root, ignore_errors=True)
        hung = self.line("classical-rk4", 8)
        status, run, calls, summary = self.run_bench(hung["trial_id"], 3, failed=["a", "b"])
        self.assertEqual(status, 1)
        self.assertEqual([c["path"] for c in calls], ["cpp.jsonl"])
        self.assertEqual(summary, [["cpp", "FAILED", "1 hard exit(s); crashed before a hard exit: a, b", "3"]])

    def test_consecutive_hard_exits_accumulate_crashed_builds(self):
        args = bench.parse_args(["run", "--set", "perf"])
        run = bench.Run(args, bench.resolve(args), key=KEY, data_root=self.root, logs_root=self.logs)
        planned = self.planned()
        path = os.path.join(run.log_dir, "cpp.jsonl")
        trials.write_jsonl(path, planned)
        calls = []

        def step(label, logfile, command):
            current = command.argv[command.argv.index("--trials") + 1]
            calls.append(os.path.basename(current))
            if len(calls) == 3:
                return 0
            with open(current + ".progress", "w") as handle:
                json.dump({"failed": ["first"] if len(calls) == 1 else ["second", "first"]}, handle)
            return WATCHDOG_EXIT_CODE

        with mock.patch.object(run, "step", side_effect=step), \
                mock.patch.object(bench, "abandon_after_hard_exit", side_effect=[planned[1:], planned[2:]]):
            run.run_package("cpp", planned, path)
        self.assertEqual(calls, ["cpp.jsonl", "cpp.retry1.jsonl", "cpp.retry2.jsonl"])
        with open(run.summary) as handle:
            self.assertEqual(handle.read(),
                             "cpp\tFAILED\t2 hard exit(s); crashed before a hard exit: first, second\t0\n")
        self.assertEqual(run.failures, 1)

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


class PullStore(unittest.TestCase):
    """bench.pull_store: a mirror without this key's files pulls at once; one with files asks the box for unpushed files first and any refuses the run before anything is pulled."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def pull(self, check_code=0, box=""):
        def fake_run(command, root, key, *args, **kwargs):
            self.calls.append(command)
            return check_code if command == "unpushed" else 0
        with mock.patch.object(bench.sync, "unavailable", return_value=""),                 mock.patch.object(bench.sync, "box_ready", return_value=box),                 mock.patch.object(bench.sync, "run", side_effect=fake_run):
            bench.pull_store(KEY, self.root)

    def test_a_box_that_cannot_prune_refuses_before_anything_is_pulled(self):
        # The push needs the box-side script.
        with self.assertRaises(SystemExit) as raised:
            self.pull(box="the box cannot run box_prune.py")
        self.assertIn("box_prune.py", str(raised.exception))
        self.assertIn("--no-sync", str(raised.exception))
        self.assertEqual(self.calls, [])

    def test_empty_partition_pulls_without_a_check(self):
        os.makedirs(os.path.join(self.root, "key=" + KEY, "package=cubie"))
        self.pull()
        self.assertEqual(self.calls, ["pull"])

    def test_matching_partition_is_checked_then_pulled(self):
        path = os.path.join(self.root, "key=" + KEY, "package=cubie", "results", "lorenz__rk4.parquet")
        os.makedirs(os.path.dirname(path))
        open(path, "w").close()
        self.pull()
        self.assertEqual(self.calls, ["unpushed", "pull"])

    def test_differing_partition_refuses_before_pulling(self):
        path = os.path.join(self.root, "key=" + KEY, "package=cubie", "results", "lorenz__rk4.parquet")
        os.makedirs(os.path.dirname(path))
        open(path, "w").close()
        with self.assertRaises(SystemExit) as raised:
            self.pull(check_code=1)
        self.assertIn("files the box lacks or differs from", str(raised.exception))
        self.assertIn("sync/sync.py push", str(raised.exception))
        self.assertEqual(self.calls, ["unpushed"])

    def test_no_sync_skips_the_store(self):
        with mock.patch.object(bench.sync, "run") as run:
            bench.pull_store(KEY, self.root, skip=True)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
