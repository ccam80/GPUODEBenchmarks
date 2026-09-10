"""runner.py against a fake adapter: every outcome, the abandon rule, finals kept and not kept, unknown controllers, build and optimize failures, cold builds, builds kept across lines, the progress file and the CLI."""

import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import runner  # noqa: E402
import store  # noqa: E402
import trials  # noqa: E402
import wp_common  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
NAN = float("nan")
CAP_S = 0.05


def spec(n=8, transfers=("both", "none"), finals=False, build="warm", optimize=None, **overrides):
    """A run spec with the expansion's own fields: lorenz, fixed tsit5 at dt 2^-10, cubie."""
    fields = dict(problem="lorenz", system_params="{}", duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0, n=n,
                  grid_dtype="float32", algorithm="tsit5", controller="fixed", dt=2.0 ** -10,
                  dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains="{}", newton_atol=NAN,
                  newton_rtol=NAN, package="cubie", transfers=list(transfers), finals=finals,
                  build=build, optimize=optimize, watchdog_s=CAP_S, set="test", stepping="fixed")
    fields.update(overrides)
    return fields


class FakeBuild:
    def __init__(self, trial):
        self.states = 3
        self.trial = trial
        self.closed = False

    def close(self):
        self.closed = True


class FakeAdapter:
    """Solve behaviour by (n, transfers): ok, slow (past the cap), oom, error; an optimize fails when ("optimize", n) says error."""

    controllers = ("fixed", "default", "pi")

    def __init__(self, behaviour=None, fail_build=False, bad_row=None):
        self.behaviour = dict(behaviour or {})
        self.fail_build = fail_build
        self.bad_row = bad_row
        self.calls = []
        self.builds = []

    def version(self):
        return "fake 1.0"

    def states(self, trial):
        return 3

    def build(self, trial, cold=False):
        self.calls.append(("build", trial["n"], cold))
        if self.fail_build:
            raise RuntimeError("no such system")
        if cold:
            time.sleep(0.01)
        build = FakeBuild(trial)
        self.builds.append(build)
        return build

    def compile(self, leg, trial, values):
        self.calls.append(("compile", trial["n"], None))

    def reset(self, leg, trial, values, transfers):
        self.calls.append(("reset", trial["n"], transfers))

    def optimize(self, build, trial, values):
        batch = int(values.shape[0])
        self.calls.append(("optimize", batch, None))
        if self.behaviour.get(("optimize", batch)) == "error":
            raise RuntimeError("no launch timed")

    def solve(self, leg, trial, values, transfers):
        self.calls.append(("solve", trial["n"], transfers))
        what = self.behaviour.get((trial["n"], transfers), "ok")
        if what == "slow":
            time.sleep(CAP_S * 1.5)
        elif what == "oom":
            raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY: allocation failed")
        elif what == "error":
            raise ValueError("bad " + transfers)
        return {"n": int(values.shape[0]), "values": np.asarray(values)}

    def finals(self, leg, result):
        n = result["n"]
        finals = np.tile(result["values"][:, None], (1, 3)).astype(np.float32)
        t_final = np.full(n, 1.0)
        retcode = [""] * n
        if self.bad_row is not None:
            finals[self.bad_row, 1] = np.nan
            retcode[self.bad_row] = "STEP_TOO_SMALL"
        return finals, t_final, retcode


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="runner_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "data")
        self.saved = (wp_common.WATCHDOG_SECONDS, runner.WATCHDOG_SECONDS)
        wp_common.WATCHDOG_SECONDS = runner.WATCHDOG_SECONDS = CAP_S
        self.addCleanup(self.restore)

    def restore(self):
        wp_common.WATCHDOG_SECONDS, runner.WATCHDOG_SECONDS = self.saved

    def run_specs(self, specs, adapter, floor=False):
        """(status, rows by (n, transfers), trial file path) after running the specs' trials through the adapter."""
        trial_list = trials.build_trials(specs)
        path = trials.write_jsonl(os.path.join(self.tmp, "cubie.jsonl"), trial_list)
        status = runner.Runner(adapter, KEY, self.root, floor=floor, repeats=3).run_file(path)
        rows = {(r["n"], r["transfers"]): r for r in store.Store(self.root).rows()}
        return status, rows, path


class OutcomeTests(RunnerCase):
    def test_clean_legs_record_every_transfers_row_and_close(self):
        adapter = FakeAdapter()
        status, rows, path = self.run_specs([spec(8), spec(32)], adapter)
        self.assertEqual(status, 0)
        self.assertEqual(sorted(rows), [(8, "both"), (8, "none"), (32, "both"), (32, "none")])
        for row in rows.values():
            self.assertTrue(math.isfinite(row["min_ms"]))
            self.assertEqual(len(row["samples_ms"]), 4)       # warm-up + 3 repeats
            self.assertEqual(row["min_ms"], min(row["samples_ms"][1:]))
            self.assertEqual(row["errored_pct"], 0.0)
            self.assertEqual((row["reason"], row["finals"]), ("", ""))
            self.assertEqual(row["package_version"], "fake 1.0")
            self.assertTrue(row["suite_rev"])
            self.assertEqual(row["states"], 3)
            self.assertTrue(math.isnan(row["build_s"]))
        self.assertEqual(adapter.calls[:1], [("build", 8, False)])
        self.assertNotIn("compile", {c[0] for c in adapter.calls})
        solves = [c for c in adapter.calls if c[0] == "solve"]
        self.assertEqual([(n, t) for _, n, t in solves][::4], [(8, "both"), (8, "none"), (32, "both"), (32, "none")])
        self.assertTrue(all(build.closed for build in adapter.builds))
        with open(path + ".progress") as handle:
            progress = json.load(handle)
        self.assertEqual(progress["trial_id"], [t for t in trials.read_jsonl(path)][-1]["trial_id"])
        self.assertEqual(progress["stage"], "solve")
        self.assertTrue(progress["started_utc"].endswith("Z"))

    def test_timeout_abandons_the_harder_runs_of_the_same_transfers(self):
        adapter = FakeAdapter({(32, "none"): "slow"})
        status, rows, _ = self.run_specs([spec(8), spec(32), spec(128)], adapter)
        self.assertEqual(status, 0)
        hit = rows[(32, "none")]
        self.assertTrue(math.isnan(hit["min_ms"]))
        self.assertTrue(hit["reason"].startswith("timeout: "), hit["reason"])
        self.assertIn("over the 0.05s cap", hit["reason"])
        self.assertEqual(len(hit["samples_ms"]), 1)
        # The run returned, so its finals still give the errored share.
        self.assertEqual(hit["errored_pct"], 0.0)
        after = rows[(128, "none")]
        self.assertTrue(math.isnan(after["min_ms"]))
        self.assertEqual(after["reason"], "abandoned: timeout at " + hit["trial_id"])
        self.assertEqual(after["samples_ms"], [])
        self.assertTrue(math.isfinite(rows[(128, "both")]["min_ms"]))
        self.assertTrue(math.isfinite(rows[(32, "both")]["min_ms"]))
        self.assertNotIn(("solve", 128, "none"), adapter.calls)
        self.assertIn(("solve", 128, "both"), adapter.calls)

    def test_reset_runs_before_every_attempt_after_the_first(self):
        adapter = FakeAdapter()
        status, rows, _ = self.run_specs([spec(8, transfers=("none",))], adapter)
        calls = [c for c in adapter.calls if c[0] in ("solve", "reset")]
        self.assertEqual(calls, [("solve", 8, "none")] + [("reset", 8, "none"), ("solve", 8, "none")] * 3)

    def test_the_trials_own_budget_sets_the_cap(self):
        adapter = FakeAdapter({(32, "none"): "slow"})
        status, rows, _ = self.run_specs([spec(8, watchdog_s=1.0), spec(32, watchdog_s=1.0),
                                          spec(128, watchdog_s=1.0)], adapter)
        self.assertTrue(all(math.isfinite(r["min_ms"]) for r in rows.values()))
        self.assertEqual({r["reason"] for r in rows.values()}, {""})
        adapter = FakeAdapter({(32, "none"): "slow"})
        status, rows, _ = self.run_specs([spec(32, watchdog_s=CAP_S)], adapter)
        self.assertEqual(rows[(32, "none")]["reason"][:9], "timeout: ")
        self.assertIn("over the {0:g}s cap".format(CAP_S), rows[(32, "none")]["reason"])

    def test_oom_abandons_like_a_timeout_and_the_other_transfers_continue(self):
        adapter = FakeAdapter({(32, "both"): "oom"})
        status, rows, _ = self.run_specs([spec(8), spec(32), spec(128)], adapter)
        hit = rows[(32, "both")]
        self.assertTrue(math.isnan(hit["min_ms"]))
        self.assertEqual(hit["reason"], "oom: RuntimeError: CUDA_ERROR_OUT_OF_MEMORY: allocation failed")
        self.assertEqual(hit["samples_ms"], [])
        self.assertTrue(math.isnan(hit["errored_pct"]))
        self.assertTrue(math.isfinite(rows[(32, "none")]["min_ms"]))
        self.assertEqual(rows[(128, "both")]["reason"], "abandoned: oom at " + hit["trial_id"])
        self.assertTrue(math.isfinite(rows[(128, "none")]["min_ms"]))

    def test_an_error_marks_its_row_and_the_build_continues(self):
        adapter = FakeAdapter({(32, "both"): "error", (32, "none"): "error"})
        status, rows, _ = self.run_specs([spec(8), spec(32), spec(128)], adapter)
        self.assertEqual(rows[(32, "both")]["reason"], "error: ValueError: bad both")
        self.assertEqual(rows[(32, "none")]["reason"], "error: ValueError: bad none")
        for key in ((8, "both"), (8, "none"), (128, "both"), (128, "none")):
            self.assertTrue(math.isfinite(rows[key]["min_ms"]), key)
            self.assertEqual(rows[key]["reason"], "")

    def test_a_none_failure_after_a_good_both_marks_the_none_row_only(self):
        adapter = FakeAdapter({(8, "none"): "error"})
        status, rows, _ = self.run_specs([spec(8)], adapter)
        self.assertTrue(math.isfinite(rows[(8, "both")]["min_ms"]))
        self.assertEqual(rows[(8, "both")]["errored_pct"], 0.0)
        self.assertTrue(math.isnan(rows[(8, "none")]["min_ms"]))
        # The finals were read from the good leg, so the errored share stands on both rows.
        self.assertEqual(rows[(8, "none")]["errored_pct"], 0.0)


class FinalsTests(RunnerCase):
    def test_finals_kept_land_one_file_per_trial_with_the_errored_share(self):
        adapter = FakeAdapter(bad_row=2)
        status, rows, _ = self.run_specs([spec(8, finals=True)], adapter)
        both, none = rows[(8, "both")], rows[(8, "none")]
        self.assertEqual(both["finals"], "finals/" + both["trial_id"] + ".parquet")
        self.assertEqual(none["finals"], both["finals"])
        self.assertEqual(both["errored_pct"], 12.5)
        self.assertEqual(none["errored_pct"], 12.5)
        traj, states, t_final, retcode = store.Store(self.root).load_finals("cubie", KEY, both["finals"])
        self.assertEqual(states.shape, (8, 3))
        self.assertEqual(states.dtype, np.float32)
        self.assertTrue(np.isnan(states[2, 1]))
        self.assertEqual(list(retcode), [""] * 2 + ["STEP_TOO_SMALL"] + [""] * 5)
        self.assertEqual(list(t_final), [1.0] * 8)
        # The finals are the grid's values: v[i] = 0 + i * 3 in float32.
        self.assertEqual(list(states[:, 0]), [float(3 * i) for i in range(8)])

    def test_finals_not_kept_leave_no_file(self):
        adapter = FakeAdapter(bad_row=0)
        status, rows, _ = self.run_specs([spec(8)], adapter)
        self.assertEqual({r["finals"] for r in rows.values()}, {""})
        self.assertEqual({r["errored_pct"] for r in rows.values()}, {12.5})
        self.assertFalse(os.path.isdir(os.path.join(self.root, "key=" + KEY, "package=cubie", "finals")))


class BuildTests(RunnerCase):
    def test_an_unknown_controller_records_the_reason_without_running(self):
        adapter = FakeAdapter()
        status, rows, _ = self.run_specs([spec(8, controller="gustafsson", dt=NAN, atol=1e-5, rtol=1e-5,
                                               stepping="gustafsson")], adapter)
        self.assertEqual(status, 0)
        self.assertEqual({r["reason"] for r in rows.values()}, {"error: unknown controller gustafsson"})
        self.assertTrue(all(math.isnan(r["min_ms"]) for r in rows.values()))
        self.assertEqual({r["states"] for r in rows.values()}, {3})
        self.assertEqual(adapter.calls, [])

    def test_a_build_failure_records_every_row_of_its_lines_once(self):
        adapter = FakeAdapter(fail_build=True)
        status, rows, _ = self.run_specs([spec(8), spec(32)], adapter)
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["reason"] for r in rows.values()}, {"error: RuntimeError: no such system"})
        self.assertEqual([c[0] for c in adapter.calls], ["build"])

    def test_a_cold_line_rebuilds_and_carries_its_own_build_time(self):
        adapter = FakeAdapter()
        status, rows, path = self.run_specs([spec(8, build="cold"), spec(32, build="cold"), spec(128)], adapter)
        self.assertEqual([c for c in adapter.calls if c[0] in ("build", "compile")],
                         [("build", 8, True), ("compile", 8, None), ("build", 32, True), ("compile", 32, None)])
        self.assertEqual([t["cold"] for t in trials.read_jsonl(path)], [True, True, False])
        self.assertGreater(rows[(8, "both")]["build_s"], 0.0)
        self.assertEqual(rows[(8, "both")]["build_s"], rows[(8, "none")]["build_s"])
        self.assertGreater(rows[(32, "none")]["build_s"], 0.0)
        self.assertNotEqual(rows[(8, "both")]["build_s"], rows[(32, "both")]["build_s"])
        self.assertTrue(math.isnan(rows[(128, "both")]["build_s"]))
        self.assertEqual(len(adapter.builds), 2)

    def test_an_optimize_failure_leaves_the_solves_running(self):
        adapter = FakeAdapter({("optimize", 64): "error"})
        status, rows, _ = self.run_specs([spec(8, optimize={"n": 64}), spec(32, optimize={"n": 64})], adapter)
        self.assertEqual(adapter.calls[:2], [("build", 8, False), ("optimize", 64, None)])
        self.assertTrue(all(math.isfinite(r["min_ms"]) for r in rows.values()))
        self.assertEqual({r["reason"] for r in rows.values()}, {""})

    def test_each_line_optimizes_at_its_batch_before_its_solves(self):
        adapter = FakeAdapter()
        self.run_specs([spec(8, optimize={"n": 64}, build="cold")], adapter)
        self.assertEqual(adapter.calls[:3], [("build", 8, True), ("compile", 8, None), ("optimize", 64, None)])
        adapter = FakeAdapter()
        self.run_specs([spec(8, optimize={"n": "solve"}), spec(32, optimize={"n": "solve"})], adapter)
        self.assertEqual([c for c in adapter.calls if c[0] != "solve" and c[0] != "reset"],
                         [("build", 8, False), ("optimize", 8, None), ("optimize", 32, None)])
        # A line without transfers warms the build alone.
        adapter = FakeAdapter()
        self.run_specs([spec(8, transfers=())], adapter)
        self.assertEqual(adapter.calls, [("build", 8, False), ("compile", 8, None)])

    def test_only_the_optimize_runs_under_the_watchdog(self):
        table = {"n": 64}
        budgets = []

        def recording(run, on_breach, budget_s=None):
            budgets.append(budget_s)
            return run()

        saved = runner.run_watchdogged
        runner.run_watchdogged = recording
        self.addCleanup(setattr, runner, "run_watchdogged", saved)
        adapter = FakeAdapter()
        status, rows, path = self.run_specs([spec(8, optimize=table)], adapter)
        self.assertEqual(adapter.calls[:2], [("build", 8, False), ("optimize", 64, None)])
        self.assertEqual(budgets, [runner.OPTIMIZE_SECONDS])
        self.assertGreater(runner.OPTIMIZE_SECONDS, runner.WATCHDOG_SECONDS)
        with open(path + ".progress") as handle:
            self.assertEqual(json.load(handle)["stage"], "solve")

    def test_a_build_is_kept_while_consecutive_lines_share_it(self):
        adapter = FakeAdapter()
        specs = [spec(8), spec(32), spec(8, algorithm="euler"), spec(32, algorithm="euler")]
        status, rows, _ = self.run_specs(specs, adapter)
        builds = [c for c in adapter.calls if c[0] == "build"]
        self.assertEqual(len(builds), 2)
        self.assertEqual(len(adapter.builds), 2)
        self.assertEqual([build.trial["algorithm"] for build in adapter.builds], ["euler", "tsit5"])
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["algorithm"] for r in store.Store(self.root).rows()}, {"tsit5", "euler"})


class RuleTests(unittest.TestCase):
    def test_classification_and_reasons(self):
        self.assertEqual(runner.classify(MemoryError("x")), "oom")
        self.assertEqual(runner.classify(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")), "oom")
        self.assertEqual(runner.classify(RuntimeError("RESOURCE_EXHAUSTED: out of memory")), "oom")

        class OutOfMemoryError(Exception):
            pass

        self.assertEqual(runner.classify(OutOfMemoryError("cuda")), "oom")
        self.assertEqual(runner.classify(ValueError("chunked")), "error")
        long_text = "m" * 300
        reason = runner.failure_reason("error", ValueError(long_text))
        self.assertEqual(reason, "error: ValueError: " + "m" * 200)
        self.assertEqual(runner.failure_reason("timeout", elapsed_s=130.25),
                         "timeout: 130.2s over the {0:g}s cap".format(runner.WATCHDOG_SECONDS))


class CliTests(RunnerCase):
    def test_main_builds_the_adapter_for_the_key_and_honours_floor(self):
        trial_list = trials.build_trials([spec(8, transfers=("both",))])
        path = trials.write_jsonl(os.path.join(self.tmp, "cubie.jsonl"), trial_list)
        solve = trial_list[0]
        data = store.Store(self.root)
        fields = {f: solve[f] for f in store.TRIAL_FIELDS}
        data.record(dict(fields, transfers="both", key=KEY, states=3, min_ms=1e-9))
        made = []

        def factory(key, root):
            made.append((key, root))
            return FakeAdapter()

        status = runner.main(["--trials", path, "--floor"], factory, key=KEY, root=self.root)
        self.assertEqual(status, 0)
        self.assertEqual(made, [(KEY, self.root)])
        rows = data.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["min_ms"], 1e-9)
        status = runner.main(["--trials", path], factory, key=KEY, root=self.root)
        self.assertNotEqual(data.rows()[0]["min_ms"], 1e-9)
        with self.assertRaises(SystemExit):
            runner.parse_args([])


if __name__ == "__main__":
    unittest.main()
