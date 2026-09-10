"""abandon.py: the abandon rule over a family, the rows a hard exit implies while solving or optimizing, the store-driven abandonment and the trials left to run."""

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
        self.record_ok([t for t in trial_list if t["n"] == 8 and t["algorithm"] == "tsit5"][0])
        hung = [t for t in trial_list if t["n"] == 32][0]
        self.progress_for(hung)
        remaining = abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev")
        rows = self.data.rows()
        self.assertEqual(sorted((r["n"], r["dt"], r["transfers"]) for r in rows if math.isnan(r["min_ms"])),
                         [(32, 2.0 ** -10, "both"), (32, 2.0 ** -10, "none"), (128, 2.0 ** -10, "both"),
                          (128, 2.0 ** -10, "none")])
        self.assertEqual({r["reason"] for r in rows if math.isnan(r["min_ms"])},
                         {"abandoned: hard-exit at " + hung["trial_id"]})
        self.assertEqual([(t["n"], t["dt"], t["algorithm"]) for t in remaining],
                         [(8, 2.0 ** -10, "euler"), (128, 0.5, "tsit5")])

    def test_an_optimize_hard_exit_records_a_timeout_row_and_drops_the_optimize(self):
        trial_list = trials.build_trials([spec(8, {"n": 64}), spec(32, {"n": 64})])
        self.progress_for(trial_list[0], "optimize")
        remaining = abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev")
        self.assertEqual(self.data.rows(), [])
        self.assertEqual([(t["n"], t["optimize"]) for t in remaining], [(8, None), (32, 64)])
        import csv
        with open(os.path.join(self.data.root, "key=" + KEY, "package=cubie", "optimize.csv"),
                  newline="", encoding="utf-8") as handle:
            recorded = list(csv.DictReader(handle))
        self.assertEqual(len(recorded), 1)
        self.assertEqual((recorded[0]["label"], recorded[0]["n"], recorded[0]["setting"],
                          recorded[0]["mode"], recorded[0]["settings"]),
                         ("timeout", "64", "0.0009765625", "fixed", ""))

    def test_the_stores_failures_abandon_the_harder_runs_before_they_spawn(self):
        trial_list = trials.build_trials([spec(8), spec(32), spec(128)])
        fields = {f: trial_list[1][f] for f in store.TRIAL_FIELDS}
        self.data.record(dict(fields, transfers="none", key=KEY, states=3, min_ms=NAN,
                              reason="timeout: 130.0s over the 120s cap"))
        self.data.record(dict(fields, transfers="both", key=KEY, states=3, min_ms=1.0))
        kept = abandon.abandon_from_store(self.data, KEY, trial_list, "rev")
        self.assertEqual([(t["n"], t["transfers"]) for t in kept],
                         [(8, ["both", "none"]), (32, ["both", "none"]), (128, ["both"])])
        rows = self.data.rows(n=128)
        self.assertEqual([(r["transfers"], r["reason"]) for r in rows],
                         [("none", "abandoned: timeout at " + trial_list[1]["trial_id"])])
        self.assertEqual(rows[0]["suite_rev"], "rev")
        # A second pass records the same abandonment over the same row.
        self.assertEqual(abandon.abandon_from_store(self.data, KEY, trial_list, "rev"), kept)
        self.assertEqual(len(self.data.rows(n=128)), 1)
    def test_a_progress_file_naming_no_trial_returns_none(self):
        trial_list = trials.build_trials([spec(8)])
        with open(self.progress, "w", encoding="utf-8") as handle:
            json.dump({"trial_id": "none", "started_utc": "x"}, handle)
        self.assertIsNone(abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev"))
        os.remove(self.progress)
        self.assertIsNone(abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev"))


if __name__ == "__main__":
    unittest.main()
