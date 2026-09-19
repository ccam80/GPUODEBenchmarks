"""abandon.py: the abandon rule over a family, the rows a hard exit implies while solving or optimizing, the compile timeouts the store records per compile_key, and the trials left to run."""

import json
import math
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import abandon  # noqa: E402
import store  # noqa: E402
import trials  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
NAN = float("nan")


def spec(n, optimize=None, **overrides):
    fields = dict(problem="lorenz", system_params="{}", duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0, n=n,
                  grid_dtype="float32", algorithm="tsit5", controller="fixed", dt=2.0 ** -10,
                  dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains="{}", newton_atol=NAN,
                  newton_rtol=NAN, package="cubie", transfers=["both", "none"], finals=False,
                  build="warm", optimize=optimize, set="test", stepping="fixed")
    fields.update(overrides)
    return fields


class AbandonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="abandon_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = store.Store(os.path.join(self.tmp, "data"))
        self.progress = os.path.join(self.tmp, "cubie.jsonl.progress")

    def progress_for(self, trial, stage="solve"):
        with open(self.progress, "w", encoding="utf-8") as handle:
            json.dump({"trial_id": trial["trial_id"], "stage": stage, "started_utc": "x"}, handle)

    def record_ok(self, trial):
        fields = {f: trial[f] for f in store.TRIAL_FIELDS}
        for transfers in trial["transfers"]:
            self.data.record(dict(fields, transfers=transfers, key=KEY, states=3, min_ms=1.0))

    def test_the_abandon_rule_gives_up_the_harder_runs_of_a_family(self):
        def line(n, **overrides):
            return trials.build_trials([spec(n, **overrides)])[0]

        base = line(32)
        failures = [(base, "timeout")]
        self.assertIsNone(abandon.abandon_reason(line(8), failures))
        self.assertIsNone(abandon.abandon_reason(line(32), failures))
        self.assertEqual(abandon.abandon_reason(line(128), failures), "abandoned: timeout at " + base["trial_id"])
        self.assertEqual(abandon.abandon_reason(line(32, dt=2.0 ** -12), failures), "abandoned: timeout at " + base["trial_id"])
        self.assertIsNone(abandon.abandon_reason(line(128, dt=0.5), failures))
        self.assertIsNone(abandon.abandon_reason(line(128, algorithm="euler"), failures))
        self.assertIsNone(abandon.abandon_reason(line(128, precision="float64"), failures))
        harder = line(32, problem="lorenz96", system_params='{"states":64}', parameter="F", grid_max=16.0)
        easier = line(32, problem="lorenz96", system_params='{"states":32}', parameter="F", grid_max=16.0)
        self.assertEqual(abandon.abandon_reason(harder, [(easier, "oom")]), "abandoned: oom at " + easier["trial_id"])
        self.assertIsNone(abandon.abandon_reason(easier, [(harder, "oom")]))
        history = abandon.History()
        history.add(base, "none", "timeout")
        history.add(line(8), "both", "error")
        self.assertEqual(history.reason(line(128), "none"), "abandoned: timeout at " + base["trial_id"])
        self.assertIsNone(history.reason(line(128), "both"))

    def test_a_solve_hard_exit_abandons_the_harder_runs_and_keeps_the_rest(self):
        trial_list = trials.build_trials([spec(8), spec(32), spec(128), spec(128, dt=0.5), spec(8, algorithm="euler")])
        # File order: the two n = 8 lines ran before the hung n = 32 line; the n = 128, dt = 0.5 line after it
        # already holds rows from an earlier run and runs again all the same.
        for trial in trial_list:
            if trial["n"] == 8 or trial["dt"] == 0.5:
                self.record_ok(trial)
        hung = [t for t in trial_list if t["n"] == 32][0]
        self.record_ok(hung)
        self.progress_for(hung)
        remaining = abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev")
        rows = self.data.rows()
        self.assertEqual(sorted((r["n"], r["dt"], r["transfers"]) for r in rows if math.isnan(r["min_ms"])),
                         [(32, 2.0 ** -10, "both"), (32, 2.0 ** -10, "none"), (128, 2.0 ** -10, "both"),
                          (128, 2.0 ** -10, "none")])
        self.assertEqual({r["reason"] for r in rows if math.isnan(r["min_ms"])},
                         {"abandoned: hard-exit at " + hung["trial_id"]})
        self.assertEqual(sorted((r["n"], r["dt"]) for r in rows if not math.isnan(r["min_ms"])),
                         [(8, 2.0 ** -10)] * 4 + [(128, 0.5)] * 2)
        self.assertEqual([(t["n"], t["dt"], t["algorithm"]) for t in remaining], [(128, 0.5, "tsit5")])

    def test_an_optimize_hard_exit_records_a_timeout_row_and_drops_the_optimize(self):
        trial_list = trials.build_trials([spec(8, True), spec(32, True)])
        self.progress_for(trial_list[0], "optimize")
        remaining = abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev")
        # The store records the compile timeout of the group: a NaN row per transfers of every line of it.
        rows = self.data.rows()
        self.assertEqual(sorted((r["n"], r["transfers"]) for r in rows),
                         [(8, "both"), (8, "none"), (32, "both"), (32, "none")])
        self.assertEqual({(r["compile"], r["reason"], r["suite_rev"]) for r in rows},
                         {("compile_timeout", "compile timeout at " + trial_list[0]["trial_id"], "rev")})
        self.assertTrue(all(math.isnan(r["min_ms"]) for r in rows))
        self.assertEqual(abandon.compile_timeouts(self.data, KEY), {trials.compile_key(trial_list[0])})
        self.assertEqual(abandon.compile_timeouts(self.data, KEY, "cubie_mlir"), set())
        # The n = 32 line runs on the hung kernel, so its optimize goes with it; both lines are marked.
        self.assertEqual([(t["n"], t["optimize"], t["compile"]) for t in remaining],
                         [(8, False, "timeout"), (32, False, "timeout")])
        import csv
        with open(os.path.join(self.data.root, "key=" + KEY, "package=cubie", "optimize.csv"),
                  newline="", encoding="utf-8") as handle:
            recorded = list(csv.DictReader(handle))
        self.assertEqual(len(recorded), 1)
        self.assertEqual((recorded[0]["label"], recorded[0]["n"], recorded[0]["stepping"].split(";")[0],
                          recorded[0]["settings"]),
                         ("timeout", "8", "dt=", ""))
        self.assertNotIn("per", recorded[0])
        # Only the later lines of the hung kernel lose their optimize; euler and dt = 0.5 precede it in the file.
        trial_list = trials.build_trials([spec(8, True), spec(32, True), spec(8, True, dt=0.5),
                                          spec(8, True, algorithm="euler")])
        hung = [t for t in trial_list if t["algorithm"] == "tsit5" and t["dt"] == 2.0 ** -10 and t["n"] == 8][0]
        self.progress_for(hung, "optimize")
        remaining = abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev")
        self.assertEqual([(t["n"], t["dt"], t["algorithm"], t["optimize"]) for t in remaining],
                         [(8, 2.0 ** -10, "tsit5", False), (32, 2.0 ** -10, "tsit5", False)])
        # A hang in a line of another kernel leaves this kernel's lines optimizing.
        other = [t for t in trial_list if t["algorithm"] == "euler"][0]
        self.progress_for(other, "optimize")
        remaining = abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev")
        self.assertEqual([(t["algorithm"], t["optimize"]) for t in remaining],
                         [("euler", False), ("tsit5", True), ("tsit5", True), ("tsit5", True)])

    def test_a_compile_timeout_is_recorded_for_the_lines_of_its_group_that_have_no_row(self):
        trial_list = trials.build_trials([spec(8, True), spec(32, True), spec(8, True, controller="default", atol=1e-5,
                                                                              rtol=1e-5, dt=NAN),
                                          spec(8, True, algorithm="euler"), spec(8, True, package="cubie_mlir")])
        fixed = [t for t in trial_list if t["controller"] == "fixed" and t["algorithm"] == "tsit5"
                 and t["package"] == "cubie"]
        self.record_ok(fixed[1])
        written = abandon.abandon_compile(self.data, KEY, trial_list, fixed[0], "rev")
        # The adaptive tsit5 line, euler and cubie_mlir are other groups; the n = 32 line's rows stand.
        self.assertEqual(trials.COMPILE_FIELDS, ("package", "problem", "system_params", "precision", "algorithm",
                                                 "controller"))
        self.assertEqual(sorted((r["n"], r["transfers"]) for r in written), [(8, "both"), (8, "none")])
        rows = self.data.rows()
        self.assertEqual(sorted((r["n"], r["compile"]) for r in rows),
                         [(8, "compile_timeout"), (8, "compile_timeout"), (32, ""), (32, "")])
        self.assertTrue(abandon.compile_timed_out(self.data, KEY, fixed[1]))
        self.assertFalse(any(abandon.compile_timed_out(self.data, KEY, t) for t in trial_list if t not in fixed))
        # Marking touches the group's lines alone and hands back the same list when none is in it.
        marked = trials.mark_compile_timeouts(trial_list, abandon.compile_timeouts(self.data, KEY))
        self.assertEqual([(t["package"], t["algorithm"], t["controller"], t["optimize"], t["compile"]) for t in marked],
                         [("cubie", "euler", "fixed", True, ""), ("cubie", "tsit5", "default", True, ""),
                          ("cubie", "tsit5", "fixed", False, "timeout"), ("cubie", "tsit5", "fixed", False, "timeout"),
                          ("cubie_mlir", "tsit5", "fixed", True, "")])
        self.assertIs(trials.mark_compile_timeouts(marked, abandon.compile_timeouts(self.data, KEY)), marked)
        self.assertIs(trials.mark_compile_timeouts(trial_list, set()), trial_list)
        self.assertEqual(trials.compile_timeouts_of(marked), {trials.compile_key(fixed[0])})
        # Recording the same group again writes nothing more.
        self.assertEqual(abandon.abandon_compile(self.data, KEY, trial_list, fixed[1], "rev"), [])
        # The marks survive the trial file, and a file written before the column reads "".
        path = os.path.join(self.tmp, "cubie.jsonl")
        trials.write_jsonl(path, marked)
        self.assertEqual([t["compile"] for t in trials.read_jsonl(path)], ["", "", "timeout", "timeout", ""])
        with open(path, "w", encoding="utf-8") as handle:
            for line in open(trials.write_jsonl(path + ".old", trial_list), encoding="utf-8"):
                record = json.loads(line)
                del record["compile"]
                handle.write(json.dumps(record) + "\n")
        self.assertEqual({t["compile"] for t in trials.read_jsonl(path)}, {""})

    def test_a_progress_file_naming_no_trial_returns_none(self):
        trial_list = trials.build_trials([spec(8)])
        with open(self.progress, "w", encoding="utf-8") as handle:
            json.dump({"trial_id": "none", "started_utc": "x"}, handle)
        self.assertIsNone(abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev"))
        os.remove(self.progress)
        self.assertIsNone(abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev"))

    def test_the_progress_file_names_the_builds_that_crashed_before_the_hard_exit(self):
        trial_list = trials.build_trials([spec(8), spec(32)])
        self.progress_for(trial_list[1])
        self.assertEqual(abandon.crashed_builds(self.progress), [])
        with open(self.progress, "w", encoding="utf-8") as handle:
            json.dump({"trial_id": trial_list[1]["trial_id"], "stage": "solve", "started_utc": "x",
                       "failed": ["lorenz/{}/float32/vern7/fixed/{}"]}, handle)
        self.assertEqual(abandon.crashed_builds(self.progress), ["lorenz/{}/float32/vern7/fixed/{}"])
        # The abandonment reads the same file.
        self.assertEqual([t["n"] for t in abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress,
                                                                            "rev")], [])
        os.remove(self.progress)
        self.assertEqual(abandon.crashed_builds(self.progress), [])


if __name__ == "__main__":
    unittest.main()
