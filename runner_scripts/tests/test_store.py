"""The parquet result store: schema, upsert by identity, floor, setting tolerance, the leg lock, finals round-trip, DuckDB reads across keys, and the CLI."""

import io
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

import store  # noqa: E402

NAN = float("nan")
STORE_PY = os.path.join(os.path.dirname(HERE), "store.py")


def identity(**overrides):
    ident = dict(package="cubie", key="windows_RTX-4070-SUPER", problem="lorenz",
                 algorithm="tsit5", mode="fixed", setting_kind="dt",
                 setting=2.0 ** -10, n=8, states=3, tier="default",
                 transfers="both")
    ident.update(overrides)
    return ident


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="store_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = store.Store(self.tmp)

    def leg_file(self, package="cubie", key="windows_RTX-4070-SUPER",
                 problem="lorenz", algorithm="tsit5", mode="fixed"):
        return os.path.join(self.tmp, "key=" + key, "package=" + package,
                            "results", "{0}__{1}__{2}.parquet".format(
                                problem, algorithm, mode))


class SchemaTests(StoreCase):
    def test_a_leg_file_carries_every_column_with_its_arrow_type(self):
        self.store.record(identity(min_ms=1.5, samples_ms=[9.0, 2.0, 1.5],
                                   errored_pct=0.0, build_s=2.5,
                                   package_version="cubie 0.12.0 numba-cuda",
                                   suite_rev="abc1234"))
        table = pq.read_table(self.leg_file())
        self.assertEqual(table.schema.names, list(store.COLUMNS))
        self.assertEqual(table.schema, store.SCHEMA)
        self.assertNotIn("analysis", table.schema.names)
        row = table.to_pylist()[0]
        self.assertEqual(row["samples_ms"], [9.0, 2.0, 1.5])
        self.assertEqual(row["states"], 3)
        self.assertTrue(math.isnan(row["error"]))
        self.assertEqual(row["reason"], "")
        self.assertEqual(row["finals"], "")
        self.assertEqual(row["recorded_utc"].tzinfo.utcoffset(None).total_seconds(), 0)

    def test_value_columns_default_and_the_identity_is_required(self):
        row = self.store.record(identity())
        self.assertTrue(math.isnan(row["min_ms"]))
        self.assertEqual(row["samples_ms"], [])
        for field in ("errored_pct", "error", "build_s"):
            self.assertTrue(math.isnan(row[field]))
        with self.assertRaises(ValueError):
            self.store.record({k: v for k, v in identity().items() if k != "tier"})
        with self.assertRaises(ValueError):
            self.store.record(identity(analysis="times"))
        with self.assertRaises(ValueError):
            self.store.record(identity(package="julia"))
        with self.assertRaises(ValueError):
            self.store.record(identity(samples_ms="9;2;1.5"))

    def test_null_floats_and_iso_timestamps_come_through_json(self):
        row = self.store.record(identity(min_ms=None, error=None,
                                         recorded_utc="2026-09-09T01:02:03Z"))
        self.assertTrue(math.isnan(row["min_ms"]))
        self.assertEqual(row["recorded_utc"],
                         datetime(2026, 9, 9, 1, 2, 3, tzinfo=timezone.utc))


class UpsertTests(StoreCase):
    def test_the_same_identity_replaces_and_every_axis_shares_the_leg_file(self):
        self.store.record(identity(min_ms=1.5))
        self.store.record(identity(min_ms=2.5, errored_pct=12.5))
        self.store.record(identity(n=32, min_ms=4.0))
        self.store.record(identity(transfers="none", min_ms=0.5))
        self.store.record(identity(setting=2.0 ** -8, min_ms=0.9))
        self.store.record(identity(states=16, min_ms=9.0))
        rows = pq.read_table(self.leg_file()).to_pylist()
        self.assertEqual(len(rows), 5)
        first = [r for r in rows if r["n"] == 8 and r["transfers"] == "both"
                 and r["states"] == 3 and r["setting"] == 2.0 ** -10][0]
        self.assertEqual((first["min_ms"], first["errored_pct"]), (2.5, 12.5))

    def test_settings_match_within_a_relative_1e_8(self):
        self.store.record(identity(setting=0.0625, min_ms=1.0))
        self.store.record(identity(setting=0.0625 * (1 + 1e-10), min_ms=2.0))
        self.store.record(identity(setting=0.0625 * (1 + 1e-6), min_ms=3.0))
        rows = pq.read_table(self.leg_file()).to_pylist()
        self.assertEqual(sorted(r["min_ms"] for r in rows), [2.0, 3.0])
        self.assertEqual(self.store.status(identity(setting=0.0625 * (1 - 1e-9))),
                         "finite")

    def test_floor_keeps_the_lower_finite_time_and_nan_never_wins(self):
        self.store.record(identity(min_ms=1.5))
        self.store.record(identity(min_ms=2.5), floor=True)
        self.store.record(identity(min_ms=NAN, reason="error: x"), floor=True)
        self.assertEqual(self.store.status(identity()), "finite")
        row = pq.read_table(self.leg_file()).to_pylist()[0]
        self.assertEqual(row["min_ms"], 1.5)
        self.assertEqual(row["reason"], "")
        self.store.record(identity(min_ms=1.4), floor=True)
        self.assertEqual(pq.read_table(self.leg_file()).to_pylist()[0]["min_ms"], 1.4)
        # A NaN row is replaced by a finite one under floor, and a NaN by a later NaN.
        self.store.record(identity(n=32, min_ms=NAN, reason="abandoned: oom at ordinal 3"))
        self.store.record(identity(n=32, min_ms=9.0), floor=True)
        self.assertEqual(self.store.status(identity(n=32)), "finite")
        self.store.record(identity(n=128, min_ms=NAN, reason="first"))
        self.store.record(identity(n=128, min_ms=NAN, reason="second"), floor=True)
        rows = {r["n"]: r for r in pq.read_table(self.leg_file()).to_pylist()}
        self.assertEqual(rows[128]["reason"], "second")
        # Without floor a NaN replaces a finite time outright.
        self.store.record(identity(n=32, min_ms=NAN))
        self.assertEqual(self.store.status(identity(n=32)), "nan")

    def test_status_matches_the_identity_columns_given(self):
        self.assertEqual(self.store.status(identity()), "absent")
        self.store.record(identity(min_ms=NAN, reason="error: boom"))
        self.store.record(identity(transfers="none", min_ms=0.5))
        self.assertEqual(self.store.status(identity()), "nan")
        self.assertEqual(self.store.status(identity(transfers="none")), "finite")
        loose = {k: v for k, v in identity().items() if k != "transfers"}
        self.assertEqual(self.store.status(loose), "finite")
        self.assertEqual(self.store.status(identity(tier="pi")), "absent")

    def test_clear_drops_matching_rows_and_removes_an_emptied_file(self):
        self.store.record(identity(min_ms=1.0))
        self.store.record(identity(n=32, min_ms=2.0))
        self.store.record(identity(problem="pollu", min_ms=3.0))
        self.assertEqual(self.store.clear(problem="lorenz", n=32), 1)
        self.assertEqual(self.store.clear(problem="pollu"), 1)
        self.assertFalse(os.path.exists(self.leg_file(problem="pollu")))
        self.assertEqual(self.store.clear(problem="pollu"), 0)
        self.assertEqual(len(pq.read_table(self.leg_file()).to_pylist()), 1)


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
        self.store.record(identity(min_ms=1.0))
        self.assertFalse(os.path.exists(path + ".lock"))
        self.assertFalse(os.path.exists(path + ".partial"))
        self.assertEqual(self.store.status(identity()), "finite")

    def test_the_lock_is_held_while_the_file_is_rewritten(self):
        path = self.leg_file()
        seen = []
        original = store.Store._write_leg

        def spy(p, rows):
            seen.append(os.path.isdir(path + ".lock"))
            return original(p, rows)

        store.Store._write_leg = staticmethod(spy)
        self.addCleanup(setattr, store.Store, "_write_leg", staticmethod(original))
        self.store.record(identity(min_ms=1.0))
        self.assertEqual(seen, [True])
        self.assertFalse(os.path.exists(path + ".lock"))


class FinalsTests(StoreCase):
    def test_finals_round_trip_with_the_spec_schema_and_name(self):
        ident = identity(mode="adaptive", setting_kind="tol", setting=1e-5,
                         n=32768, tier="matched")
        finals = np.arange(12, dtype=np.float64).reshape(4, 3) / 7
        converged = [True, True, False, True]
        relative = self.store.record_finals(ident, finals, converged)
        self.assertEqual(relative, "finals/lorenz__tsit5__adaptive__tol-1e-05"
                                   "__n32768__s3__matched.parquet")
        path = os.path.join(self.tmp, "key=windows_RTX-4070-SUPER",
                            "package=cubie", "finals",
                            relative.split("/")[1])
        table = pq.read_table(path)
        self.assertEqual(table.column_names, ["traj", "s1", "s2", "s3", "converged"])
        self.assertEqual(str(table.schema.field("traj").type), "int32")
        self.assertEqual(str(table.schema.field("s1").type), "float")
        self.assertEqual(str(table.schema.field("converged").type), "bool")
        traj, states, ok = self.store.load_finals("cubie", "windows_RTX-4070-SUPER",
                                                  relative)
        self.assertEqual(list(traj), [0, 1, 2, 3])
        self.assertEqual(states.dtype, np.float32)
        np.testing.assert_array_equal(states, finals.astype(np.float32))
        self.assertEqual(list(ok), converged)
        # The row points at the file by that relative path.
        row = self.store.record(dict(ident, transfers="none", min_ms=3.0,
                                     finals=relative))
        self.assertEqual(row["finals"], relative)
        self.assertEqual(self.store.rows(tier="matched")[0]["finals"], relative)

    def test_a_dt_setting_is_named_with_ten_significant_digits(self):
        ident = identity(setting=0.0009765625, n=1024)
        relative = self.store.record_finals(ident, [[1.0, 2.0, 3.0]], [True])
        self.assertEqual(relative, "finals/lorenz__tsit5__fixed__dt-0.0009765625"
                                   "__n1024__s3__default.parquet")
        with self.assertRaises(ValueError):
            self.store.record_finals(ident, [[1.0, 2.0, 3.0]], [True, False])


class DuckDBTests(StoreCase):
    def test_rows_read_across_keys_and_packages_with_hive_partitions(self):
        self.store.record(identity(min_ms=1.0, samples_ms=[3.0, 1.0, 1.2]))
        self.store.record(identity(key="linux_A100", package="julia_gpu",
                                   min_ms=2.0))
        self.store.record(identity(key="linux_A100", package="julia_gpu",
                                   problem="pollu", states=20, n=1024,
                                   transfers="none", min_ms=NAN,
                                   reason="abandoned: oom at ordinal 2"))
        rows = self.store.rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(sorted((r["key"], r["package"]) for r in rows),
                         [("linux_A100", "julia_gpu"), ("linux_A100", "julia_gpu"),
                          ("windows_RTX-4070-SUPER", "cubie")])
        self.assertEqual(set(rows[0]), set(store.COLUMNS))
        by_package = self.store.rows(package="julia_gpu", problem="lorenz")
        self.assertEqual([r["min_ms"] for r in by_package], [2.0])
        self.assertEqual(self.store.rows(setting=2.0 ** -10 * (1 + 1e-10),
                                         package="cubie")[0]["samples_ms"],
                         [3.0, 1.0, 1.2])
        nan_rows = self.store.rows("isnan(min_ms)")
        self.assertEqual([r["reason"] for r in nan_rows],
                         ["abandoned: oom at ordinal 2"])
        self.assertEqual(self.store.rows("n > ?" if False else "n >= 1024",
                                         transfers="none")[0]["states"], 20)
        stamp = rows[0]["recorded_utc"]
        self.assertEqual(stamp.utcoffset().total_seconds(), 0)
        with self.assertRaises(ValueError):
            self.store.rows(analysis="times")

    def test_an_empty_tree_reads_as_no_rows_and_query_still_answers(self):
        self.assertEqual(self.store.rows(), [])
        table = self.store.query("SELECT count(*) AS c FROM results")
        self.assertEqual(table.to_pylist(), [{"c": 0}])
        self.assertEqual(self.store.query("SELECT * FROM results").column_names,
                         list(store.COLUMNS))

    def test_query_runs_sql_over_the_results_view(self):
        for n, ms in ((8, 1.0), (32, 2.0), (128, NAN)):
            self.store.record(identity(n=n, min_ms=ms))
        table = self.store.query(
            "SELECT n, min_ms FROM results WHERE package = 'cubie' ORDER BY n")
        rows = table.to_pylist()
        self.assertEqual([r["n"] for r in rows], [8, 32, 128])
        self.assertTrue(math.isnan(rows[2]["min_ms"]))


class CliTests(StoreCase):
    def run_cli(self, *args, stdin=None):
        proc = subprocess.run([sys.executable, STORE_PY, "--root", self.tmp] + list(args),
                              input=stdin, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_record_takes_a_json_array_with_samples_as_a_list(self):
        rows = [dict(identity(), min_ms=1.5, samples_ms=[9.0, 2.0, 1.5],
                     errored_pct=0.0, package_version="cubie 0.12.0",
                     suite_rev="abc1234"),
                dict(identity(transfers="none"), min_ms=None, samples_ms=None,
                     reason="error: RuntimeError: boom")]
        self.run_cli("record", "-", stdin=json.dumps(rows))
        stored = self.store.rows()
        self.assertEqual(len(stored), 2)
        both = [r for r in stored if r["transfers"] == "both"][0]
        self.assertEqual(both["samples_ms"], [9.0, 2.0, 1.5])
        none = [r for r in stored if r["transfers"] == "none"][0]
        self.assertTrue(math.isnan(none["min_ms"]))
        self.assertEqual(none["samples_ms"], [])
        self.assertEqual(none["reason"], "error: RuntimeError: boom")
        # NaN spelled as JSON's NaN literal is accepted too, and --floor keeps the lower time.
        path = os.path.join(self.tmp, "rows.json")
        with open(path, "w") as handle:
            handle.write('[{"package": "cubie", "key": "windows_RTX-4070-SUPER", '
                         '"problem": "lorenz", "algorithm": "tsit5", "mode": "fixed", '
                         '"setting_kind": "dt", "setting": 0.0009765625, "n": 8, '
                         '"states": 3, "tier": "default", "transfers": "both", '
                         '"min_ms": NaN, "samples_ms": []}]')
        self.run_cli("record", path, "--floor")
        self.assertEqual(self.store.rows(transfers="both")[0]["min_ms"], 1.5)
        self.run_cli("record", path)
        self.assertTrue(math.isnan(self.store.rows(transfers="both")[0]["min_ms"]))

    def test_finals_status_query_and_clear(self):
        ident_path = os.path.join(self.tmp, "identity.json")
        with open(ident_path, "w") as handle:
            json.dump(identity(n=1024, transfers="none"), handle)
        csv_path = os.path.join(self.tmp, "finals.csv")
        with open(csv_path, "w") as handle:
            handle.write("traj,s1,s2,s3,converged\n0,1.5,2.5,3.5,1\n1,0.1,0.2,0.3,0\n")
        relative = self.run_cli("finals", ident_path, csv_path).strip()
        self.assertEqual(relative, "finals/lorenz__tsit5__fixed__dt-0.0009765625"
                                   "__n1024__s3__default.parquet")
        traj, states, ok = self.store.load_finals("cubie", "windows_RTX-4070-SUPER",
                                                  relative)
        np.testing.assert_allclose(states, [[1.5, 2.5, 3.5], [0.1, 0.2, 0.3]],
                                   rtol=1e-6)
        self.assertEqual(list(ok), [True, False])
        self.assertEqual(self.run_cli("status", ident_path).strip(), "absent")
        self.run_cli("record", "-", stdin=json.dumps(
            [dict(identity(n=1024, transfers="none"), min_ms=2.0, finals=relative)]))
        self.assertEqual(self.run_cli("status", ident_path).strip(), "finite")
        out = self.run_cli("query", "SELECT n, min_ms, finals, samples_ms FROM results")
        lines = out.strip().splitlines()
        self.assertEqual(lines[0], "n,min_ms,finals,samples_ms")
        self.assertEqual(lines[1], "1024,2,{0},".format(relative))
        filter_path = os.path.join(self.tmp, "filter.json")
        with open(filter_path, "w") as handle:
            json.dump({"problem": "lorenz", "n": 1024}, handle)
        self.assertEqual(self.run_cli("clear", filter_path).strip(), "1")
        self.assertEqual(self.run_cli("status", ident_path).strip(), "absent")


class JuliaShimTests(unittest.TestCase):
    """The results.jl shim round-trips through store.py; runs the Julia test script under the repo project."""

    def test_julia_round_trip(self):
        import shutil as _shutil
        from launch import julia_command, suite_python
        if _shutil.which(julia_command()[0]) is None:
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
