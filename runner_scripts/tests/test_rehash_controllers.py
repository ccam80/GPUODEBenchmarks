"""The store rewrite to the declared cubie controllers: rows that ran the declared controller take its name and gains and new ids, their finals follow, other rows and their finals and optimize records are dropped, a cubie finals file no row names goes, duplicates keep the most complete row, and the controller exports go."""

import csv
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cubie_adapter  # noqa: E402
import grid  # noqa: E402
import rehash_controllers as rehash  # noqa: E402
import store  # noqa: E402

NAN = float("nan")
KEY = "windows_RTX-4070-SUPER"
TIER_TSIT5 = {"integral_gain": 0.36, "max_step_growth": 10.0, "min_step_shrink": 0.2,
              "proportional_gain": 0.4800000000000001, "safety": 0.9}


def spec(**overrides):
    fields = dict(problem="lorenz", system_params={}, duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0,
                  n=8, grid_dtype="float32", algorithm="tsit5", controller="pi",
                  dt=2.0 ** -10, dt_min=NAN, dt_max=NAN, atol=1e-5, rtol=1e-5, gains=TIER_TSIT5,
                  newton_atol=NAN, newton_rtol=NAN, transfers="both", package="cubie", key=KEY)
    fields.update(overrides)
    return fields


class RehashTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="rehash_")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.store = store.Store(self.root)

    def record(self, finals=False, **overrides):
        fields = spec(**overrides)
        relative = ""
        if finals:
            values = grid.grid(fields).astype(np.float64)
            relative = self.store.record_finals(fields, np.column_stack([values, values, values]),
                                                np.full(fields["n"], 1.0))
        return self.store.record(dict(fields, states=3, min_ms=overrides.get("min_ms", 1.0), finals=relative))

    def optimize_record(self, **overrides):
        trial = dict(spec(**overrides), transfers=["both"])
        trial["gains"] = store.canonical_json(trial["gains"])
        trial["system_params"] = store.canonical_json(trial["system_params"])
        cubie_adapter.record_optimize_timeout(trial, KEY, self.root)

    def test_rows_that_ran_the_declared_controller_are_rewritten_and_the_rest_dropped(self):
        tier = self.record(finals=True)
        self.record(transfers="none")
        self.record(controller="default", gains={})
        self.record(algorithm="kvaerno3", controller="default", gains={}, newton_atol=1e-5, newton_rtol=1e-5)
        radau_default = self.record(algorithm="radau_iia_5", controller="default", gains={}, newton_atol=1e-5,
                                    newton_rtol=1e-5, min_ms=NAN)
        radau_matched = self.record(algorithm="radau_iia_5", controller="gustafsson", gains={"safety": 0.9},
                                    newton_atol=1e-5, newton_rtol=1e-5, finals=True)
        self.record(algorithm="radau_iia_5", controller="pi", gains=TIER_TSIT5, newton_atol=1e-5, newton_rtol=1e-5,
                    finals=True)
        fixed = self.record(controller="fixed", atol=NAN, rtol=NAN, gains={})
        jax = self.record(package="jax", controller="default", gains={})
        rewritten, dropped = rehash.rehash(self.root, out=io.StringIO())
        self.assertEqual((rewritten, dropped), (4, 3))
        rows = {(r["algorithm"], r["controller"], r["transfers"]): r for r in self.store.rows()}
        tsit5 = rows[("tsit5", "pi", "both")]
        self.assertEqual(json.loads(tsit5["gains"]), {"integral_gain": 0.36, "max_step_growth": 10.0,
                                                       "min_step_shrink": 0.2, "proportional_gain": 0.48,
                                                       "safety": 0.9})
        self.assertEqual(tsit5["run_id"], store.run_id(tsit5))
        self.assertNotEqual(tsit5["run_id"], tier["run_id"])
        self.assertEqual(tsit5["finals"], store.finals_name(tsit5))
        self.assertTrue(self.store.finals_readable("cubie", KEY, tsit5["finals"]))
        self.assertEqual(rows[("tsit5", "pi", "none")]["trial_id"], tsit5["trial_id"])
        self.assertEqual(json.loads(rows[("kvaerno3", "pi", "both")]["gains"])["proportional_gain"], 0.533333333333)
        radau = rows[("radau_iia_5", "gustafsson", "both")]
        self.assertEqual(json.loads(radau["gains"]), {"max_step_growth": 8.0, "min_step_shrink": 0.2, "safety": 0.9})
        self.assertEqual(radau["min_ms"], 1.0)
        self.assertEqual(radau["finals"], store.finals_name(radau))
        self.assertEqual(rows[("tsit5", "fixed", "both")]["run_id"], fixed["run_id"])
        self.assertEqual(rows[("tsit5", "default", "both")]["run_id"], jax["run_id"])
        self.assertEqual(len(rows), 6)
        finals_dir = os.path.join(self.store.package_dir("cubie", KEY), "finals")
        self.assertEqual(sorted(os.listdir(finals_dir)),
                         sorted([os.path.basename(tsit5["finals"]), os.path.basename(radau["finals"])]))
        self.assertNotIn(os.path.basename(radau_default["finals"] or "x"), os.listdir(finals_dir))
        self.assertNotEqual(radau_matched["finals"], radau["finals"])

    def test_optimize_records_and_controller_exports_follow(self):
        self.optimize_record()
        self.optimize_record(controller="default", gains={})
        self.optimize_record(algorithm="radau_iia_5", controller="gustafsson", gains={"safety": 0.9})
        self.optimize_record(algorithm="radau_iia_5", controller="default", gains={})
        self.optimize_record(controller="fixed", atol=NAN, rtol=NAN, gains={})
        exports = os.path.join(self.root, "key=" + KEY, "package=julia_cpu", "controllers")
        os.makedirs(exports)
        with open(os.path.join(exports, "lorenz.csv"), "w") as handle:
            handle.write("algorithm,controller\n")
        stray = os.path.join(self.store.package_dir("cubie", KEY), "finals", "0" * 16 + ".parquet")
        os.makedirs(os.path.dirname(stray))
        open(stray, "wb").close()
        rehash.rehash(self.root, out=io.StringIO())
        self.assertFalse(os.path.exists(stray))
        with open(os.path.join(self.root, "key=" + KEY, "package=cubie", "optimize.csv"), newline="") as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(sorted((r["algorithm"], r["controller"]) for r in records),
                         [("radau_iia_5", "gustafsson"), ("tsit5", "fixed"), ("tsit5", "pi")])
        self.assertEqual(json.loads([r for r in records if r["controller"] == "pi"][0]["gains"])["proportional_gain"], 0.48)
        self.assertFalse(os.path.isdir(exports))

    def test_a_dry_run_changes_nothing(self):
        before = self.record(finals=True)
        self.record(controller="default", gains={})
        rehash.rehash(self.root, dry_run=True, out=io.StringIO())
        self.assertEqual(sorted(r["run_id"] for r in self.store.rows()),
                         sorted([before["run_id"], store.run_id(spec(controller="default", gains={}))]))
        self.assertTrue(self.store.finals_readable("cubie", KEY, before["finals"]))


if __name__ == "__main__":
    unittest.main()
