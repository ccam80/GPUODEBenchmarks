"""abandon.py: the rows a hard exit implies for a solve, a warm and an optimize line, and the trials left to run."""

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


def spec(n, optimize=None):
    return dict(problem="lorenz", system_params="{}", duration=1.0, precision="float32",
                parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0, n=n,
                grid_dtype="float32", algorithm="tsit5", controller="fixed", dt=2.0 ** -10,
                dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains="{}", newton_atol=NAN,
                newton_rtol=NAN, package="cubie", transfers=["both", "none"], finals=False,
                axis="n", build="warm", optimize=optimize, set="test", stepping="fixed")


class AbandonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="abandon_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = store.Store(os.path.join(self.tmp, "data"))
        self.progress = os.path.join(self.tmp, "cubie.jsonl.progress")

    def progress_for(self, trial):
        with open(self.progress, "w", encoding="utf-8") as handle:
            json.dump({"trial_id": trial["trial_id"], "kind": trial["kind"], "started_utc": "x"}, handle)

    def record_ok(self, trial):
        fields = {f: trial[f] for f in store.TRIAL_FIELDS}
        for transfers in trial["transfers"]:
            self.data.record(dict(fields, transfers=transfers, key=KEY, states=3, min_ms=1.0))

    def test_a_solve_hard_exit_abandons_the_higher_ordinals_and_keeps_the_rest(self):
        trial_list = trials.build_trials([spec(8), spec(32), spec(128)])
        solves = [t for t in trial_list if t["kind"] == "solve"]
        self.record_ok(solves[0])
        self.progress_for(solves[1])
        remaining = abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev")
        rows = self.data.rows()
        self.assertEqual(sorted((r["n"], r["transfers"]) for r in rows if math.isnan(r["min_ms"])),
                         [(32, "both"), (32, "none"), (128, "both"), (128, "none")])
        self.assertEqual({r["reason"] for r in rows if math.isnan(r["min_ms"])},
                         {"abandoned: hard-exit at ordinal 1"})
        self.assertEqual(remaining, [])

    def test_an_optimize_hard_exit_records_nothing_and_drops_the_line(self):
        table = {"n": 64, "per": "leg"}
        trial_list = trials.build_trials([spec(8, table), spec(32, table)])
        optimize = [t for t in trial_list if t["kind"] == "optimize"]
        self.assertEqual(len(optimize), 1)
        self.progress_for(optimize[0])
        remaining = abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev")
        self.assertEqual(self.data.rows(), [])
        self.assertEqual([t["kind"] for t in remaining], ["warm", "solve", "solve"])
        self.assertEqual([t["n"] for t in remaining if t["kind"] == "solve"], [8, 32])
        import csv
        with open(os.path.join(self.data.root, "key=" + KEY, "package=cubie", "optimize.csv"),
                  newline="", encoding="utf-8") as handle:
            recorded = list(csv.DictReader(handle))
        self.assertEqual(len(recorded), 1)
        self.assertEqual((recorded[0]["label"], recorded[0]["n"], recorded[0]["setting"],
                          recorded[0]["mode"], recorded[0]["settings"]),
                         ("timeout", "64", "", "fixed", ""))

    def test_a_per_solve_optimize_hard_exit_keeps_its_solve(self):
        table = {"n": "solve", "per": "solve"}
        trial_list = trials.build_trials([spec(8, table), spec(32, table)])
        second = [t for t in trial_list if t["kind"] == "optimize"][1]
        self.progress_for(second)
        remaining = abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev")
        self.assertEqual(self.data.rows(), [])
        self.assertEqual([(t["kind"], t["n"]) for t in remaining],
                         [("warm", 8), ("optimize", 8), ("solve", 8), ("solve", 32)])

    def test_a_progress_file_naming_no_trial_returns_none(self):
        trial_list = trials.build_trials([spec(8)])
        with open(self.progress, "w", encoding="utf-8") as handle:
            json.dump({"trial_id": "none", "started_utc": "x"}, handle)
        self.assertIsNone(abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev"))
        os.remove(self.progress)
        self.assertIsNone(abandon.abandon_after_hard_exit(self.data, KEY, trial_list, self.progress, "rev"))


if __name__ == "__main__":
    unittest.main()
