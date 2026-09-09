"""The parquet result store: spec columns, hashes against a hand fixture, upsert by run_id, floor, batches, the leg lock, finals in the run precision, DuckDB reads across keys, and the CLI."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import grid  # noqa: E402
import store  # noqa: E402

NAN = float("nan")
STORE_PY = os.path.join(os.path.dirname(HERE), "store.py")
KEY = "windows_RTX-4070-SUPER"

# The hand fixture: the canonical spec text written out in table order, hashed with sha1.
FIXTURE_HEAD = (
    '{"problem":"lorenz","system_params":"{}","duration":1,"precision":"float32",'
    '"parameter":"rho","grid_scale":"linear","grid_min":0,"grid_max":21,"n":8,'
    '"grid_dtype":"float32","algorithm":"tsit5","controller":"fixed","dt":0.0009765625,'
    '"dt_min":"nan","dt_max":"nan","atol":"nan","rtol":"nan","gains":"{}",'
    '"newton_atol":"nan","newton_rtol":"nan"')
FIXTURE_TRIAL_TEXT = FIXTURE_HEAD + ',"package":"cubie"}'
FIXTURE_RUN_TEXT = FIXTURE_HEAD + \
    ',"transfers":"both","package":"cubie","key":"windows_RTX-4070-SUPER"}'
FIXTURE_GROUP_TEXT = (
    '{"problem":"lorenz","system_params":"{}","duration":1,"precision":"float32",'
    '"algorithm":"tsit5","controller":"fixed","dt":0.0009765625,'
    '"dt_min":"nan","dt_max":"nan","atol":"nan","rtol":"nan","gains":"{}",'
    '"newton_atol":"nan","newton_rtol":"nan"}')
FIXTURE_TRIAL_ID = "9754cd221fddca05"
FIXTURE_RUN_ID = "eaa18a5f41beb42f"
FIXTURE_GROUP_ID = hashlib.sha1(FIXTURE_GROUP_TEXT.encode()).hexdigest()[:16]


def spec(**overrides):
    """The fixture spec: lorenz, fixed tsit5 at dt = 2^-10, n = 8, cubie on the 4070, both transfers."""
    fields = dict(problem="lorenz", system_params={}, duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0,
                  n=8, grid_dtype="float32", algorithm="tsit5", controller="fixed",
                  dt=2.0 ** -10, dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains={},
                  newton_atol=NAN, newton_rtol=NAN, transfers="both", package="cubie",
                  key=KEY)
    fields.update(overrides)
    return fields


def row(**overrides):
    """The fixture spec as a row with three states."""
    fields = spec(states=3)
    fields.update(overrides)
    return fields


def adaptive(**overrides):
    """The fixture stepping switched to the default controller at tol 1e-5 with dt0, dt_min."""
    fields = dict(controller="default", dt=2.0 ** -10, dt_min=1e-6, atol=1e-5, rtol=1e-5)
    fields.update(overrides)
    return fields


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="store_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = store.Store(self.tmp)

    def leg_file(self, package="cubie", key=KEY, problem="lorenz", algorithm="tsit5"):
        return os.path.join(self.tmp, "key=" + key, "package=" + package,
                            "results", "{0}__{1}.parquet".format(problem, algorithm))


class HashTests(unittest.TestCase):
    def test_the_identity_is_exactly_the_run_spec_in_table_order(self):
        self.assertEqual(store.SPEC_FIELDS, (
            "problem", "system_params", "duration", "precision",
            "parameter", "grid_scale", "grid_min", "grid_max", "n", "grid_dtype",
            "algorithm", "controller", "dt", "dt_min", "dt_max", "atol", "rtol", "gains",
            "newton_atol", "newton_rtol", "transfers", "package", "key"))
        self.assertEqual(store.TRIAL_FIELDS, store.SPEC_FIELDS[:-3] + ("package",))
        self.assertEqual(store.GROUP_FIELDS, store.SPEC_FIELDS[:4] + store.SPEC_FIELDS[10:20])
        self.assertEqual(list(store.COLUMNS), list(store.SPEC_FIELDS) + [
            "run_id", "trial_id", "group_id", "states", "min_ms", "samples_ms",
            "errored_pct", "build_s", "reason", "finals", "package_version",
            "suite_rev", "recorded_utc"])
        for absent in ("error", "reference"):
            self.assertNotIn(absent, store.COLUMNS)

    def test_hashes_match_the_hand_fixture(self):
        canonical = store.spec_of(spec())
        self.assertEqual(store.canonical_spec_text(canonical, store.TRIAL_FIELDS),
                         FIXTURE_TRIAL_TEXT)
        self.assertEqual(store.canonical_spec_text(canonical, store.SPEC_FIELDS),
                         FIXTURE_RUN_TEXT)
        self.assertEqual(store.canonical_spec_text(canonical, store.GROUP_FIELDS),
                         FIXTURE_GROUP_TEXT)
        self.assertEqual(hashlib.sha1(FIXTURE_TRIAL_TEXT.encode()).hexdigest()[:16],
                         FIXTURE_TRIAL_ID)
        self.assertEqual(store.trial_id(spec()), FIXTURE_TRIAL_ID)
        self.assertEqual(store.run_id(spec()), FIXTURE_RUN_ID)
        self.assertEqual(store.group_id(spec()), FIXTURE_GROUP_ID)
        self.assertEqual(store.ids(spec()), {"trial_id": FIXTURE_TRIAL_ID,
                                             "run_id": FIXTURE_RUN_ID,
                                             "group_id": FIXTURE_GROUP_ID})
        # A trial dict with extra fields hashes the same.
        self.assertEqual(store.trial_id(dict(spec(), kind="solve", finals=False, leg="x")),
                         FIXTURE_TRIAL_ID)

    def test_trial_id_ignores_transfers_and_key_and_run_id_does_not(self):
        base = spec()
        for change in (dict(transfers="none"), dict(key="linux_A100")):
            other = spec(**change)
            self.assertEqual(store.trial_id(other), store.trial_id(base))
            self.assertNotEqual(store.run_id(other), store.run_id(base))
        self.assertNotEqual(store.trial_id(spec(package="cubie_mlir")), store.trial_id(base))

    def test_group_id_ignores_the_ensemble_transfers_package_and_key(self):
        base = spec()
        for change in (dict(transfers="none"), dict(key="linux_A100"),
                       dict(package="julia_gpu"), dict(n=131072),
                       dict(n=1024, grid_max=0.16389), dict(parameter="sigma"),
                       dict(grid_scale="log", grid_min=1e-3)):
            self.assertEqual(store.group_id(spec(**change)), FIXTURE_GROUP_ID, change)
        for change in (dict(dt=2.0 ** -8), dict(precision="float64"), dict(duration=2.0),
                       dict(system_params={"states": 4}), dict(algorithm="vern7"),
                       adaptive(), dict(gains={"kp": 0.7}), dict(newton_atol=1e-6)):
            self.assertNotEqual(store.group_id(spec(**change)), FIXTURE_GROUP_ID, change)

    def test_floats_compare_exactly_and_nan_is_the_text_nan(self):
        base = store.run_id(spec())
        self.assertNotEqual(store.run_id(spec(dt=2.0 ** -10 * (1 + 1e-10))), base)
        self.assertEqual(store.run_id(spec(dt="0.0009765625")), base)
        self.assertEqual(store.run_id(spec(dt_min=None)), base)
        self.assertEqual(store.run_id(spec(dt_min="nan")), base)
        self.assertNotEqual(store.run_id(spec(dt_min=1e-6)), base)
        self.assertEqual(store.run_id(spec(duration=1)), base)
        self.assertEqual(store.run_id(spec(n="8")), base)
        self.assertEqual(store.run_id(spec(n=8.0)), base)
        for bad in (dict(n=8.5), dict(n="8.5"), dict(n=True), dict(dt="1e-5x")):
            with self.assertRaises(ValueError, msg=bad):
                store.run_id(spec(**bad))
        self.assertEqual(store.run_id(spec(dt_min="")), base)

    def test_json_fields_are_canonical(self):
        self.assertEqual(store.canonical_json({"states": 32}), '{"states":32}')
        self.assertEqual(store.canonical_json('{ "b": 1, "a": 2 }'), '{"a":2,"b":1}')
        self.assertEqual(store.canonical_json(""), "{}")
        self.assertEqual(store.canonical_json(None), "{}")
        self.assertEqual(store.run_id(spec(system_params='{"states": 32}')),
                         store.run_id(spec(system_params={"states": 32})))
        self.assertNotEqual(store.run_id(spec(system_params={"states": 32})),
                            store.run_id(spec()))
        self.assertEqual(store.run_id(spec(gains={"kp": 0.7, "ki": 0.4})),
                         store.run_id(spec(gains='{"ki":0.4,"kp":0.7}')))
        with self.assertRaises(ValueError):
            store.canonical_json([1, 2])

    def test_incomplete_or_invalid_specs_are_refused(self):
        for missing in ("problem", "gains", "newton_rtol", "key", "transfers"):
            with self.assertRaises(ValueError):
                store.run_id({k: v for k, v in spec().items() if k != missing})
        self.assertEqual(store.trial_id({k: v for k, v in spec().items()
                                         if k not in ("key", "transfers")}),
                         FIXTURE_TRIAL_ID)
        for bad in (dict(package="julia"), dict(precision="float16"),
                    dict(grid_scale="cubic"), dict(grid_dtype="float64"),
                    dict(transfers="d2h"), dict(problem=""), dict(problem="a/b"),
                    dict(key="a\\b"), dict(controller=""), dict(n=0),
                    dict(dt=float("inf"))):
            with self.assertRaises(ValueError):
                store.run_id(spec(**bad))


class SchemaTests(StoreCase):
    def test_a_leg_file_carries_every_column_with_its_arrow_type(self):
        self.store.record(row(min_ms=1.5, samples_ms=[9.0, 2.0, 1.5], errored_pct=0.0,
                              build_s=2.5, package_version="cubie 0.12.0+numba-cuda",
                              suite_rev="abc1234"))
        table = pq.read_table(self.leg_file())
        self.assertEqual(table.schema.names, list(store.COLUMNS))
        self.assertEqual(table.schema, store.SCHEMA)
        stored = table.to_pylist()[0]
        self.assertEqual(stored["run_id"], FIXTURE_RUN_ID)
        self.assertEqual(stored["trial_id"], FIXTURE_TRIAL_ID)
        self.assertEqual(stored["group_id"], FIXTURE_GROUP_ID)
        self.assertEqual(stored["system_params"], "{}")
        self.assertEqual(stored["gains"], "{}")
        self.assertEqual(stored["samples_ms"], [9.0, 2.0, 1.5])
        self.assertEqual(stored["states"], 3)
        self.assertEqual(stored["n"], 8)
        self.assertTrue(math.isnan(stored["atol"]))
        self.assertEqual(stored["reason"], "")
        self.assertEqual(stored["finals"], "")
        self.assertEqual(stored["recorded_utc"].tzinfo.utcoffset(None).total_seconds(), 0)

    def test_value_columns_default_and_the_spec_and_states_are_required(self):
        standing = self.store.record(row())
        self.assertTrue(math.isnan(standing["min_ms"]))
        self.assertEqual(standing["samples_ms"], [])
        for field in ("errored_pct", "build_s"):
            self.assertTrue(math.isnan(standing[field]))
        with self.assertRaises(ValueError):
            self.store.record({k: v for k, v in row().items() if k != "controller"})
        with self.assertRaises(ValueError):
            self.store.record(spec())
        for unknown in (dict(analysis="times"), dict(error=1e-4), dict(reference="x")):
            with self.assertRaises(ValueError):
                self.store.record(row(**unknown))
        with self.assertRaises(ValueError):
            self.store.record(row(package="julia"))
        with self.assertRaises(ValueError):
            self.store.record(row(samples_ms="9;2;1.5"))

    def test_given_ids_must_hash_the_spec(self):
        standing = self.store.record(row(run_id=FIXTURE_RUN_ID, trial_id=FIXTURE_TRIAL_ID,
                                         group_id=FIXTURE_GROUP_ID))
        self.assertEqual(standing["run_id"], FIXTURE_RUN_ID)
        for name in ("run_id", "trial_id", "group_id"):
            with self.assertRaises(ValueError):
                self.store.record(row(**{name: "0123456789abcdef"}))

    def test_null_floats_and_iso_timestamps_come_through_json(self):
        standing = self.store.record(row(min_ms=None, build_s=None,
                                         recorded_utc="2026-09-09T01:02:03Z"))
        self.assertTrue(math.isnan(standing["min_ms"]))
        self.assertEqual(standing["recorded_utc"],
                         datetime(2026, 9, 9, 1, 2, 3, tzinfo=timezone.utc))


class UpsertTests(StoreCase):
    def test_the_same_run_id_replaces_and_every_stepping_shares_the_leg_file(self):
        self.store.record(row(min_ms=1.5))
        self.store.record(row(min_ms=2.5, errored_pct=12.5))
        self.store.record(row(n=32, min_ms=4.0))
        self.store.record(row(transfers="none", min_ms=0.5))
        self.store.record(row(dt=2.0 ** -8, min_ms=0.9))
        self.store.record(row(**adaptive(), min_ms=9.0))
        self.store.record(row(system_params={"states": 4}, min_ms=9.5))
        rows = pq.read_table(self.leg_file()).to_pylist()
        self.assertEqual(len(rows), 6)
        first = [r for r in rows if r["run_id"] == FIXTURE_RUN_ID][0]
        self.assertEqual((first["min_ms"], first["errored_pct"]), (2.5, 12.5))
        controllers = sorted(r["controller"] for r in rows)
        self.assertEqual(controllers, ["default"] + ["fixed"] * 5)
        self.assertEqual(len({r["trial_id"] for r in rows}), 5)
        self.assertEqual(len({r["run_id"] for r in rows}), 6)

    def test_floats_that_differ_by_rounding_are_different_rows(self):
        self.store.record(row(dt=0.0625, min_ms=1.0))
        self.store.record(row(dt=0.0625 * (1 + 1e-10), min_ms=2.0))
        rows = pq.read_table(self.leg_file()).to_pylist()
        self.assertEqual(sorted(r["min_ms"] for r in rows), [1.0, 2.0])

    def test_floor_keeps_the_lower_finite_time_and_nan_never_wins(self):
        self.store.record(row(min_ms=1.5))
        self.store.record(row(min_ms=2.5), floor=True)
        self.store.record(row(min_ms=NAN, reason="error: x"), floor=True)
        self.assertEqual(self.store.status(spec()), "finite")
        stored = pq.read_table(self.leg_file()).to_pylist()[0]
        self.assertEqual(stored["min_ms"], 1.5)
        self.assertEqual(stored["reason"], "")
        self.store.record(row(min_ms=1.4), floor=True)
        self.assertEqual(pq.read_table(self.leg_file()).to_pylist()[0]["min_ms"], 1.4)
        # A NaN row is replaced by a finite one under floor, and a NaN by a later NaN.
        self.store.record(row(n=32, min_ms=NAN, reason="abandoned: oom at ordinal 3"))
        self.store.record(row(n=32, min_ms=9.0), floor=True)
        self.assertEqual(self.store.status(spec(n=32)), "finite")
        self.store.record(row(n=128, min_ms=NAN, reason="first"))
        self.store.record(row(n=128, min_ms=NAN, reason="second"), floor=True)
        rows = {r["n"]: r for r in pq.read_table(self.leg_file()).to_pylist()}
        self.assertEqual(rows[128]["reason"], "second")
        # Without floor a NaN replaces a finite time outright.
        self.store.record(row(n=32, min_ms=NAN))
        self.assertEqual(self.store.status(spec(n=32)), "nan")

    def test_status_answers_by_run_id_or_spec(self):
        self.assertEqual(self.store.status(FIXTURE_RUN_ID), "absent")
        self.store.record(row(min_ms=NAN, reason="error: boom"))
        self.store.record(row(transfers="none", min_ms=0.5))
        self.assertEqual(self.store.status(FIXTURE_RUN_ID), "nan")
        self.assertEqual(self.store.status(spec()), "nan")
        self.assertEqual(self.store.status(spec(transfers="none")), "finite")
        self.assertEqual(self.store.status(store.run_id(spec(transfers="none"))), "finite")
        self.assertEqual(self.store.status(spec(controller="pi")), "absent")

    def test_record_batch_locks_and_rewrites_each_leg_once_in_order(self):
        writes = []
        original = store.Store._write_leg

        def spy(path, rows):
            writes.append(os.path.basename(path))
            return original(path, rows)

        store.Store._write_leg = staticmethod(spy)
        self.addCleanup(setattr, store.Store, "_write_leg", staticmethod(original))
        standing = self.store.record_batch([
            row(min_ms=3.0), row(n=32, min_ms=4.0), row(min_ms=2.0),
            row(problem="pollu", parameter="k1", grid_scale="log", grid_min=3.5e-2,
                grid_max=3.5, duration=60.0, states=20, min_ms=7.0),
            row(algorithm="vern7", min_ms=8.0)])
        self.assertEqual(writes, ["lorenz__tsit5.parquet", "pollu__tsit5.parquet",
                                  "lorenz__vern7.parquet"])
        self.assertEqual([r["min_ms"] for r in standing], [3.0, 4.0, 2.0, 7.0, 8.0])
        rows = pq.read_table(self.leg_file()).to_pylist()
        self.assertEqual(sorted(r["min_ms"] for r in rows), [2.0, 4.0])
        self.assertEqual(len(self.store.rows()), 4)
        floored = self.store.record_batch([row(min_ms=5.0), row(n=32, min_ms=1.0)], floor=True)
        self.assertEqual([r["min_ms"] for r in floored], [2.0, 1.0])
        with self.assertRaises(ValueError):
            self.store.record_batch([row(), row(package="julia")])

    def test_clear_drops_matching_rows_and_removes_an_emptied_file(self):
        self.store.record(row(min_ms=1.0))
        self.store.record(row(n=32, min_ms=2.0))
        self.store.record(row(problem="pollu", min_ms=3.0))
        self.store.record(row(**adaptive(), min_ms=4.0))
        self.assertEqual(self.store.clear(problem="lorenz", n=32), 1)
        self.assertEqual(self.store.clear(problem="pollu"), 1)
        self.assertFalse(os.path.exists(self.leg_file(problem="pollu")))
        self.assertEqual(self.store.clear(problem="pollu"), 0)
        self.assertEqual(self.store.clear(controller="default", atol=1e-5), 1)
        self.assertEqual(self.store.clear(dt_min=NAN, run_id=FIXTURE_RUN_ID), 1)
        self.assertFalse(os.path.exists(self.leg_file()))
        with self.assertRaises(ValueError):
            self.store.clear(analysis="times")


class LockTests(StoreCase):
    def test_a_fresh_lock_blocks_until_the_timeout(self):
        path = self.leg_file()
        os.makedirs(os.path.dirname(path))
        os.mkdir(path + ".lock")
        with self.assertRaises(TimeoutError):
            with store._Lock(path, timeout=0.2):
                pass
        self.assertTrue(os.path.isdir(path + ".lock"))

    def test_a_stale_lock_is_taken_over_and_released(self):
        path = self.leg_file()
        os.makedirs(os.path.dirname(path))
        os.mkdir(path + ".lock")
        old = time.time() - 2 * store.LOCK_STALE_S
        os.utime(path + ".lock", (old, old))
        self.store.record(row(min_ms=1.0))
        self.assertFalse(os.path.exists(path + ".lock"))
        self.assertFalse(os.path.exists(path + ".partial"))
        self.assertEqual(self.store.status(spec()), "finite")

    def test_the_lock_is_held_while_the_file_is_rewritten(self):
        path = self.leg_file()
        seen = []
        original = store.Store._write_leg

        def spy(p, rows):
            seen.append(os.path.isdir(path + ".lock"))
            return original(p, rows)

        store.Store._write_leg = staticmethod(spy)
        self.addCleanup(setattr, store.Store, "_write_leg", staticmethod(original))
        self.store.record(row(min_ms=1.0))
        self.assertEqual(seen, [True])
        self.assertFalse(os.path.exists(path + ".lock"))


class FinalsTests(StoreCase):
    def test_finals_round_trip_named_by_trial_id_in_float32(self):
        trial = spec(**adaptive(), n=4)
        finals = np.arange(12, dtype=np.float64).reshape(4, 3) / 7
        converged = [True, True, False, True]
        relative = self.store.record_finals(trial, finals, converged)
        self.assertEqual(relative, "finals/" + store.trial_id(trial) + ".parquet")
        path = os.path.join(self.tmp, "key=" + KEY, "package=cubie", "finals",
                            relative.split("/")[1])
        table = pq.read_table(path)
        self.assertEqual(table.column_names, ["traj", "s1", "s2", "s3", "converged"])
        self.assertEqual(str(table.schema.field("traj").type), "int32")
        self.assertEqual(str(table.schema.field("s1").type), "float")
        self.assertEqual(str(table.schema.field("converged").type), "bool")
        traj, states, ok = self.store.load_finals("cubie", KEY, relative)
        self.assertEqual(list(traj), [0, 1, 2, 3])
        self.assertEqual(states.dtype, np.float32)
        np.testing.assert_array_equal(states, finals.astype(np.float32))
        self.assertEqual(list(ok), converged)
        # The row points at the file by that relative path; both transfer legs share it.
        for transfers in ("both", "none"):
            standing = self.store.record(dict(trial, transfers=transfers, states=3,
                                              min_ms=3.0, finals=relative))
            self.assertEqual(standing["finals"], relative)
        self.assertEqual({r["finals"] for r in self.store.rows(controller="default")},
                         {relative})

    def test_float64_runs_keep_float64_finals(self):
        trial = spec(precision="float64", n=2, package="julia_cpu")
        finals = [[1.0 + 1e-12, 2.0, 3.0], [0.1, 0.2, 0.3]]
        relative = self.store.record_finals(trial, finals, [True, False])
        table = pq.read_table(os.path.join(self.tmp, "key=" + KEY, "package=julia_cpu",
                                           *relative.split("/")))
        self.assertEqual(str(table.schema.field("s1").type), "double")
        traj, states, ok = self.store.load_finals("julia_cpu", KEY, relative)
        self.assertEqual(states.dtype, np.float64)
        self.assertEqual(states[0, 0], 1.0 + 1e-12)
        self.assertNotEqual(store.trial_id(trial), store.trial_id(spec(n=2, package="julia_cpu")))

    def test_finals_carry_all_n_rows_and_one_flag_each(self):
        trial = spec(n=2)
        with self.assertRaises(ValueError):
            self.store.record_finals(trial, [[1.0, 2.0, 3.0]], [True])
        with self.assertRaises(ValueError):
            self.store.record_finals(trial, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [True])
        with self.assertRaises(ValueError):
            self.store.record_finals({k: v for k, v in trial.items() if k != "key"},
                                     [[1.0, 2.0, 3.0]] * 2, [True, False])
        loose = {k: v for k, v in trial.items() if k != "transfers"}
        self.assertEqual(self.store.record_finals(loose, [[1.0, 2.0, 3.0]] * 2, [True, False]),
                         "finals/" + store.trial_id(trial) + ".parquet")


    def test_a_golden_row_and_a_prefix_grid_row_share_a_group_id_across_packages(self):
        # The analyses pair rows by group_id and rebuild each grid from its own spec.
        golden = spec(**adaptive(), n=131072, package="julia_cpu", precision="float64",
                      transfers="none")
        point = grid.grid_point("linear", 0.0, 21.0, 131072, 1023)
        prefix = spec(**adaptive(), n=1024, grid_max=float(format(point, ".17g")),
                      package="cubie", precision="float64", transfers="both")
        relative = self.store.record_finals(golden, np.zeros((131072, 3)), np.ones(131072, bool))
        self.store.record(dict(golden, states=3, finals=relative, reason="untimed"))
        short = self.store.record_finals(prefix, np.zeros((1024, 3)), np.ones(1024, bool))
        self.store.record(dict(prefix, states=3, min_ms=1.0, finals=short))
        rows = sorted(self.store.rows(group_id=store.group_id(golden)), key=lambda r: r["n"])
        self.assertEqual([(r["package"], r["n"]) for r in rows],
                         [("cubie", 1024), ("julia_cpu", 131072)])
        self.assertNotEqual(rows[0]["trial_id"], rows[1]["trial_id"])
        np.testing.assert_array_equal(grid.grid(rows[0]), grid.grid(rows[1])[:1024])


class DuckDBTests(StoreCase):
    def test_rows_read_across_keys_and_packages_with_hive_partitions(self):
        self.store.record(row(min_ms=1.0, samples_ms=[3.0, 1.0, 1.2]))
        self.store.record(row(key="linux_A100", package="julia_gpu", min_ms=2.0))
        self.store.record(row(key="linux_A100", package="julia_gpu", problem="pollu",
                              states=20, n=1024, transfers="none", min_ms=NAN,
                              reason="abandoned: oom at ordinal 2"))
        rows = self.store.rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(sorted((r["key"], r["package"]) for r in rows),
                         [("linux_A100", "julia_gpu"), ("linux_A100", "julia_gpu"),
                          (KEY, "cubie")])
        self.assertEqual(set(rows[0]), set(store.COLUMNS))
        by_package = self.store.rows(package="julia_gpu", problem="lorenz")
        self.assertEqual([r["min_ms"] for r in by_package], [2.0])
        self.assertEqual(self.store.rows(dt=2.0 ** -10, package="cubie")[0]["samples_ms"],
                         [3.0, 1.0, 1.2])
        self.assertEqual(self.store.rows(dt=2.0 ** -10 * (1 + 1e-10)), [])
        self.assertEqual(len(self.store.rows(atol=NAN, controller="fixed")), 3)
        self.assertEqual(self.store.rows(run_id=FIXTURE_RUN_ID)[0]["min_ms"], 1.0)
        nan_rows = self.store.rows("isnan(min_ms)")
        self.assertEqual([r["reason"] for r in nan_rows],
                         ["abandoned: oom at ordinal 2"])
        self.assertEqual(self.store.rows("n >= 1024", transfers="none")[0]["states"], 20)
        stamp = rows[0]["recorded_utc"]
        self.assertEqual(stamp.utcoffset().total_seconds(), 0)
        with self.assertRaises(ValueError):
            self.store.rows(analysis="times")

    def test_an_empty_tree_reads_as_no_rows_and_query_still_answers(self):
        self.assertEqual(self.store.rows(), [])
        self.assertEqual(self.store.status(FIXTURE_RUN_ID), "absent")
        table = self.store.query("SELECT count(*) AS c FROM results")
        self.assertEqual(table.to_pylist(), [{"c": 0}])
        self.assertEqual(self.store.query("SELECT * FROM results").column_names,
                         list(store.COLUMNS))

    def test_query_runs_sql_over_the_results_view(self):
        for n, ms in ((8, 1.0), (32, 2.0), (128, NAN)):
            self.store.record(row(n=n, min_ms=ms))
        table = self.store.query(
            "SELECT n, min_ms, trial_id FROM results WHERE package = 'cubie' ORDER BY n")
        rows = table.to_pylist()
        self.assertEqual([r["n"] for r in rows], [8, 32, 128])
        self.assertTrue(math.isnan(rows[2]["min_ms"]))
        self.assertEqual(rows[0]["trial_id"], FIXTURE_TRIAL_ID)


class CliTests(StoreCase):
    def run_cli(self, *args, stdin=None):
        proc = subprocess.run([sys.executable, STORE_PY, "--root", self.tmp] + list(args),
                              input=stdin, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_record_takes_a_json_array_with_samples_as_a_list(self):
        rows = [dict(row(), min_ms=1.5, samples_ms=[9.0, 2.0, 1.5], errored_pct=0.0,
                     package_version="cubie 0.12.0", suite_rev="abc1234"),
                dict(row(transfers="none"), min_ms=None, samples_ms=None,
                     reason="error: RuntimeError: boom")]
        self.run_cli("record", "-", stdin=json.dumps(rows))
        stored = self.store.rows()
        self.assertEqual(len(stored), 2)
        both = [r for r in stored if r["transfers"] == "both"][0]
        self.assertEqual(both["samples_ms"], [9.0, 2.0, 1.5])
        self.assertEqual(both["run_id"], FIXTURE_RUN_ID)
        none = [r for r in stored if r["transfers"] == "none"][0]
        self.assertTrue(math.isnan(none["min_ms"]))
        self.assertEqual(none["samples_ms"], [])
        self.assertEqual(none["trial_id"], FIXTURE_TRIAL_ID)
        self.assertEqual(none["reason"], "error: RuntimeError: boom")
        # NaN spelled as JSON's NaN literal is accepted too, and --floor keeps the lower time.
        path = os.path.join(self.tmp, "rows.json")
        with open(path, "w") as handle:
            handle.write('[{"problem": "lorenz", "system_params": "{}", "duration": 1.0, '
                         '"precision": "float32", "parameter": "rho", "grid_scale": "linear", '
                         '"grid_min": 0.0, "grid_max": 21.0, "n": 8, "grid_dtype": "float32", '
                         '"algorithm": "tsit5", "controller": "fixed", "dt": 0.0009765625, '
                         '"dt_min": NaN, "dt_max": NaN, "atol": NaN, "rtol": NaN, "gains": {}, '
                         '"newton_atol": NaN, "newton_rtol": NaN, "transfers": "both", '
                         '"package": "cubie", "key": "windows_RTX-4070-SUPER", "states": 3, '
                         '"min_ms": NaN, "samples_ms": []}]')
        self.run_cli("record", path, "--floor")
        self.assertEqual(self.store.rows(transfers="both")[0]["min_ms"], 1.5)
        self.run_cli("record", path)
        self.assertTrue(math.isnan(self.store.rows(transfers="both")[0]["min_ms"]))

    def test_hash_finals_status_query_and_clear(self):
        trial = spec(n=2, transfers="none")
        spec_path = os.path.join(self.tmp, "spec.json")
        with open(spec_path, "w") as handle:
            json.dump(trial, handle)
        ids = json.loads(self.run_cli("hash", spec_path))
        self.assertEqual(ids, {"trial_id": store.trial_id(trial), "run_id": store.run_id(trial),
                               "group_id": store.group_id(trial)})
        csv_path = os.path.join(self.tmp, "finals.csv")
        with open(csv_path, "w") as handle:
            handle.write("traj,s1,s2,s3,converged\n0,1.5,2.5,3.5,1\n1,0.1,0.2,0.3,0\n")
        relative = self.run_cli("finals", spec_path, csv_path).strip()
        self.assertEqual(relative, "finals/" + ids["trial_id"] + ".parquet")
        traj, states, ok = self.store.load_finals("cubie", KEY, relative)
        np.testing.assert_allclose(states, [[1.5, 2.5, 3.5], [0.1, 0.2, 0.3]], rtol=1e-6)
        self.assertEqual(list(ok), [True, False])
        self.assertEqual(self.run_cli("status", ids["run_id"]).strip(), "absent")
        self.run_cli("record", "-", stdin=json.dumps(
            [dict(trial, states=3, min_ms=2.0, finals=relative)]))
        self.assertEqual(self.run_cli("status", ids["run_id"]).strip(), "finite")
        out = self.run_cli("query", "SELECT n, min_ms, finals, samples_ms, run_id FROM results")
        lines = out.strip().splitlines()
        self.assertEqual(lines[0], "n,min_ms,finals,samples_ms,run_id")
        self.assertEqual(lines[1], "2,2,{0},,{1}".format(relative, ids["run_id"]))
        filter_path = os.path.join(self.tmp, "filter.json")
        with open(filter_path, "w") as handle:
            json.dump({"problem": "lorenz", "n": 2}, handle)
        self.assertEqual(self.run_cli("clear", filter_path).strip(), "1")
        self.assertEqual(self.run_cli("status", ids["run_id"]).strip(), "absent")


class JuliaShimTests(unittest.TestCase):
    """The results.jl shim round-trips through store.py; runs the Julia test script under the repo project."""

    def test_julia_round_trip(self):
        from launch import julia_command, suite_python
        if shutil.which(julia_command()[0]) is None:
            self.skipTest("julia is not on PATH")
        script = os.path.join(HERE, "test_results_shim.jl")
        root = os.path.dirname(os.path.dirname(HERE))
        proc = subprocess.run(julia_command() + ["--project=" + root, script,
                                                 suite_python()],
                              cwd=root, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("results.jl shim", proc.stdout)


class SuiteRevTests(unittest.TestCase):
    def test_suite_rev_is_a_short_hash_here_and_unknown_outside_git(self):
        rev = store.suite_rev()
        base = rev.replace("-dirty", "")
        self.assertTrue(7 <= len(base) <= 12 and all(c in "0123456789abcdef" for c in base), rev)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        self.assertEqual(store.suite_rev(tmp), "unknown")


if __name__ == "__main__":
    unittest.main()
