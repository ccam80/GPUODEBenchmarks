"""The analysis scripts: the store read in its current form (a PI row within Float32 rounding of the DIRK tier, the most complete row of a run_id), selection by set or predicate with the ensemble fields ignored, the errored and untimed filters, the completeness report with the store's compile timeouts marked, CSVs without the columns no row captured, timing figures and CSVs on the n, states and error axes, and the agreement CSVs and figures."""

import csv
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "analyses"))

import abandon  # noqa: E402
import cubie_adapter  # noqa: E402
import grid  # noqa: E402
import sets  # noqa: E402
import store  # noqa: E402
import agreement  # noqa: E402
import shared  # noqa: E402
import timing  # noqa: E402

NAN = float("nan")
KEY = "windows_RTX-4070-SUPER"
OTHER_KEY = "linux_RTX-2060-SUPER"


def spec(**overrides):
    """A perf spec: lorenz, fixed tsit5 at dt = 2^-10, cubie on the 4070."""
    fields = dict(problem="lorenz", system_params={}, duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0,
                  n=8, grid_dtype="float32", algorithm="tsit5", controller="fixed",
                  dt=2.0 ** -10, dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains={},
                  newton_atol=NAN, newton_rtol=NAN, transfers="both", package="cubie", key=KEY)
    fields.update(overrides)
    return fields


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class AnalysesCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="analyses_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "data")
        self.out = os.path.join(self.tmp, "plots")
        self.store = store.Store(self.root)

    def row(self, min_ms=1.0, states=3, **overrides):
        return self.store.record(dict(spec(**overrides), states=states, min_ms=min_ms))

    def finals_row(self, offset=0.0, **overrides):
        """A row with finals [v + offset, 2v, -v] over its grid, all trajectories good."""
        fields = spec(**overrides)
        values = grid.grid(fields).astype(np.float64)
        finals = np.column_stack([values + offset, 2.0 * values, -values])
        relative = self.store.record_finals(fields, finals, np.full(fields["n"], 1.0))
        return self.store.record(dict(fields, states=3, finals=relative,
                                      errored_pct=overrides.get("errored_pct", 0.0),
                                      min_ms=overrides.get("min_ms", 1.0)))

    def golden(self):
        return self.finals_row(precision="float64", n=64, algorithm="Vern9", controller="default",
                               dt=NAN, atol=1e-13, rtol=1e-13, package="julia_cpu",
                               transfers="none", min_ms=NAN)

    def written(self, key=KEY, problem="lorenz"):
        directory = os.path.join(self.out, key, problem)
        return sorted(os.listdir(directory)) if os.path.isdir(directory) else []


class SelectionTests(AnalysesCase):
    def test_rows_are_selected_by_set_with_the_ensemble_fields_ignored_under_every_key(self):
        perf = [self.row(n=n) for n in (8, 32)]
        other_key = self.row(n=8, key=OTHER_KEY)
        golden_only = self.row(n=131072, dt=2.0 ** -3, transfers="none")
        alien = self.row(algorithm="euler", dt=2.0 ** -20)
        ids = lambda rows: sorted(r["run_id"] for r in rows)  # noqa: E731
        self.assertEqual(ids(shared.select_rows(self.store, ["perf"])), ids(perf + [other_key]))
        self.assertEqual(ids(shared.select_rows(self.store, ["golden_grid"])),
                         ids(perf + [other_key, golden_only]))
        self.assertEqual(ids(shared.select_rows(self.store, ["perf", "golden_grid"])),
                         ids(perf + [other_key, golden_only]))
        self.assertNotIn(alien["run_id"], ids(shared.select_rows(self.store, ["perf", "golden_grid"])))
        # A predicate names rows; every row sharing their (group_id, package) comes along.
        self.assertEqual(ids(shared.select_rows(self.store, where="n = 32")), ids(perf + [other_key]))
        self.assertEqual(ids(shared.select_rows(self.store, where="algorithm = 'euler'")), ids([alien]))
        self.assertEqual(shared.store_keys(self.store), [OTHER_KEY, KEY])

    def test_errored_and_untimed_filters(self):
        self.assertTrue(shared.within_errored_limit({"errored_pct": NAN}))
        self.assertTrue(shared.within_errored_limit({"errored_pct": None}))
        self.assertTrue(shared.within_errored_limit({}))
        self.assertTrue(shared.within_errored_limit({"errored_pct": 10.0}))
        self.assertFalse(shared.within_errored_limit({"errored_pct": 10.5}))
        self.assertFalse(timing.timed({"min_ms": NAN}))
        self.assertTrue(timing.timed({"min_ms": 0.5}))

    def pi_spec(self, package="cubie", n=8, **overrides):
        """The perf set's cubie PI point of lorenz tsit5 at n on both transfers, as the set declares it (the tier's exact gains)."""
        specs = [s for s in sets.expand(["perf"], KEY, self.root, packages=[package], algorithms=["tsit5"], n=[n])
                 if s["controller"] == "pi" and s["problem"] == "lorenz"]
        self.assertEqual(len(specs), 1)
        fields = {f: specs[0][f] for f in store.SPEC_FIELDS if f in specs[0]}
        return dict(fields, transfers="both", key=KEY, **overrides)

    @staticmethod
    def rounded(gains):
        """The gains as a Float32 table prints them."""
        return {k: float(np.float32(v)) for k, v in json.loads(gains).items()}

    def test_a_pi_row_within_float32_rounding_of_the_tier_reads_as_the_tier(self):
        exact = self.pi_spec()
        self.assertEqual(json.loads(exact["gains"])["proportional_gain"], 0.4800000000000001)
        rounded = self.store.record(dict(exact, gains=self.rounded(exact["gains"]), states=3, min_ms=2.0))
        self.assertNotEqual(rounded["gains"], exact["gains"])
        self.assertNotEqual(rounded["run_id"], store.run_id(exact))
        # Other gains, and a package outside cubie, read as written.
        other = self.store.record(dict(exact, gains={"integral_gain": 0.5, "proportional_gain": 0.1}, states=3))
        jax = self.store.record(dict(exact, package="jax", gains=self.rounded(exact["gains"]), states=3))
        data = shared.AnalysisStore(self.root)
        rows = {r["run_id"]: r for r in data.rows()}
        self.assertEqual(set(rows), {store.run_id(exact), other["run_id"], jax["run_id"]})
        read = rows[store.run_id(exact)]
        self.assertEqual((read["gains"], read["trial_id"], read["group_id"], read["min_ms"], read["finals"]),
                         (exact["gains"], store.trial_id(exact), store.group_id(exact), 2.0, ""))
        self.assertEqual(rows[jax["run_id"]]["gains"], jax["gains"])
        self.assertEqual(sorted(r["run_id"] for r in data.rows(key=KEY, package="cubie")),
                         sorted([store.run_id(exact), other["run_id"]]))
        self.assertIs(shared.normalise_gains(read), read)
        # The set selects it; the audit counts its row and finds its optimize record written under the rounded gains.
        self.assertIn(store.run_id(exact), [r["run_id"] for r in shared.select_rows(data, ["perf"])])
        trial = dict(exact, gains=rounded["gains"], transfers=["both"])
        cubie_adapter.record_optimize_timeout(trial, KEY, self.root)
        self.assertEqual(cubie_adapter.optimize_rows("cubie", KEY, self.root)[0]["gains"], rounded["gains"])
        self.assertEqual(data.optimize_rows("cubie", KEY)[0]["gains"], exact["gains"])
        lacking = {m.trial["trial_id"]: m.reasons() for _, m in shared.incomplete(data, ["perf"])}
        self.assertEqual(lacking.get(store.trial_id(exact)), ["row:none"])

    def test_the_most_complete_row_of_a_run_id_stands(self):
        exact = self.pi_spec()
        cold = self.store.record(dict(exact, states=3, min_ms=1.0, build_s=3.0,
                                      recorded_utc="2026-09-19T06:23:42Z"))
        self.store.record(dict(exact, gains=self.rounded(exact["gains"]), states=3, min_ms=1.1,
                               recorded_utc="2026-09-19T06:26:46Z"))
        rows = shared.AnalysisStore(self.root).rows()
        self.assertEqual([(r["run_id"], r["min_ms"], r["build_s"]) for r in rows], [(cold["run_id"], 1.0, 3.0)])
        timed = dict(cold, run_id="y", min_ms=1.0, build_s=NAN)
        untimed = dict(cold, run_id="y", min_ms=NAN, build_s=3.0)
        self.assertEqual([r["min_ms"] for r in shared.most_complete([untimed, timed])], [1.0])
        earlier = dict(timed, recorded_utc=cold["recorded_utc"].replace(year=2025))
        self.assertEqual([r["recorded_utc"] for r in shared.most_complete([earlier, timed])], [timed["recorded_utc"]])
        self.assertEqual([r["recorded_utc"] for r in shared.most_complete([timed, earlier])], [timed["recorded_utc"]])

    def test_a_csv_leaves_out_the_columns_no_row_captured(self):
        rows = [{"a": 1.0, "b": NAN, "c": "", "d": None, "e": [], "f": "x"},
                {"a": NAN, "b": NAN, "c": "", "d": None, "e": [1.0], "f": ""}]
        self.assertEqual(shared.captured_columns(("a", "b", "c", "d", "e", "f", "g"), rows), ["a", "e", "f"])
        self.assertEqual(shared.captured_columns(("a", "b"), []), ["a", "b"])
        path = shared.write_csv(os.path.join(self.tmp, "t.csv"), ("a", "b", "c", "d", "e", "f", "g"), rows)
        self.assertEqual(read_csv(path), [{"a": "1.0", "e": "", "f": "x"}, {"a": "nan", "e": "1.0", "f": ""}])

    def test_flags_need_a_set_or_a_predicate(self):
        with self.assertRaises(SystemExit):
            timing.main(["--no-sync", "--x", "n", "--root", self.root, "--out", self.out])
        with self.assertRaises(SystemExit):
            agreement.main(["--no-sync", "--set", "perf", "--where", "n = 8", "--root", self.root, "--out", self.out])


class CompletenessTests(AnalysesCase):
    """A named set is expanded and canonicalized under every key and what the store lacks of it is reported before anything is compared."""

    SET = ('[set]\npackages = ["cpp"]\nproblems = ["lorenz"]\nalgorithms = ["classical-rk4"]\nfinals = {finals}\n'
           'transfers = ["none"]\n[[grid]]\nn = [8, 32]\n[[stepping]]\ncontroller = "fixed"\ndt = [0.5]\n')

    def setUp(self):
        super().setUp()
        self.sets_dir = os.path.join(self.tmp, "sets")
        os.makedirs(self.sets_dir)
        with open(os.path.join(self.sets_dir, "tiny.toml"), "w", encoding="utf-8") as handle:
            handle.write(self.SET.format(finals="false"))
        with open(os.path.join(self.sets_dir, "keep.toml"), "w", encoding="utf-8") as handle:
            handle.write(self.SET.format(finals="true").replace("n = [8, 32]", "n = [32]"))

    def point(self, n, **overrides):
        return spec(package="cpp", algorithm="classical-rk4", dt=0.5, transfers="none", n=n, **overrides)

    def test_the_report_names_every_lacking_row_and_artifact_under_each_key(self):
        trial_list = shared.canonical_trials(self.store, ["tiny"], KEY, self.sets_dir)
        self.assertEqual([(t["n"], t["finals"], t["sets"]) for t in trial_list],
                         [(8, False, ["tiny"]), (32, True, ["keep", "tiny"])])
        self.store.record(dict(self.point(8), states=3, min_ms=1.0))
        self.store.record(dict(self.point(32), states=3, min_ms=1.0))
        self.store.record(dict(self.point(8, key=OTHER_KEY), states=3, min_ms=NAN))
        lacking = shared.incomplete(self.store, ["tiny"], self.sets_dir)
        self.assertEqual([(key, m.trial["n"], m.reasons()) for key, m in lacking],
                         [(OTHER_KEY, 32, ["row:none", "finals"]), (KEY, 32, ["finals"])])
        count = shared.report_incomplete(self.store, ["tiny"], self.out, self.sets_dir)
        self.assertEqual(count, 2)
        for key in (KEY, OTHER_KEY):
            table = read_csv(os.path.join(self.out, key, "incomplete.csv"))
            # A fixed stepping captures no atol.
            self.assertEqual(list(table[0]), [c for c in shared.INCOMPLETE_COLUMNS if c != "atol"])
            self.assertEqual([(r["key"], r["package"], r["n"], r["sets"], r["missing"]) for r in table],
                             [(key, "cpp", "32", "keep tiny", "row:none finals" if key == OTHER_KEY else "finals")])
        # The 32-point's finals file completes this key; the other key still lacks its row.
        fields = self.point(32)
        relative = self.store.record_finals(fields, np.zeros((32, 3)), np.full(32, 1.0))
        self.store.record(dict(fields, states=3, min_ms=1.0, finals=relative))
        self.assertEqual([(key, m.trial["n"]) for key, m in shared.incomplete(self.store, ["tiny"], self.sets_dir)],
                         [(OTHER_KEY, 32)])
        other = self.point(32, key=OTHER_KEY)
        self.store.record(dict(other, states=3, min_ms=NAN, finals=relative))
        self.assertEqual([(key, m.reasons()) for key, m in shared.incomplete(self.store, ["tiny"], self.sets_dir)],
                         [(OTHER_KEY, ["finals"])])
        self.store.record_finals(other, np.zeros((32, 3)), np.full(32, 1.0))
        self.assertEqual(shared.incomplete(self.store, ["tiny"], self.sets_dir), [])
        self.assertEqual(shared.report_incomplete(self.store, ["tiny"], self.out, self.sets_dir), 0)

    def test_a_compile_timeout_the_store_records_marks_its_lines_as_a_plan_does(self):
        def lines():
            return [t for t in shared.canonical_trials(self.store, ["perf"], KEY)
                    if (t["package"], t["problem"], t["algorithm"], t["controller"]) ==
                    ("cubie", "lorenz", "tsit5", "fixed")]
        perf = lines()
        self.assertTrue(perf)
        self.assertEqual({(t["optimize"], t["compile"]) for t in perf}, {(True, "")})
        abandon.abandon_compile(self.store, KEY, perf, perf[0])
        marked = lines()
        self.assertEqual({(t["optimize"], t["compile"]) for t in marked}, {(False, "timeout")})
        # The rows abandon_compile wrote are no rows, and no line of the group wants an optimize record.
        ids = {t["trial_id"] for t in marked}
        lacking = {m.trial["trial_id"]: m.reasons() for _, m in shared.incomplete(self.store, ["perf"])
                   if m.trial["trial_id"] in ids}
        self.assertEqual(set(lacking), ids)
        self.assertEqual({tuple(v[:2]) for v in lacking.values()}, {("row:both", "row:none")})
        self.assertEqual({tuple(v[2:]) for v in lacking.values()}, {(), ("finals",)})

    def test_the_scripts_report_a_shipped_set_the_store_lacks_and_exit_1(self):
        self.row(n=8)
        self.assertEqual(timing.main(["--no-sync", "--set", "perf", "--x", "n", "--root", self.root, "--out", self.out]), 1)
        self.assertTrue(os.path.isfile(os.path.join(self.out, KEY, "incomplete.csv")))
        self.assertEqual(agreement.main(["--no-sync", "--set", "golden", "--root", self.root, "--out", self.out]), 1)
        table = read_csv(os.path.join(self.out, KEY, "incomplete.csv"))
        self.assertEqual({r["package"] for r in table}, {"julia_cpu"})
        self.assertEqual(len(table), 8)
        self.assertEqual(agreement.main(["--no-sync", "--where", "n = 8", "--root", self.root, "--out", self.out]), 0)


class TimingTests(AnalysesCase):
    def test_n_axis_one_figure_per_stepping_and_transfers_with_a_series_per_package(self):
        for n, ms in ((8, 1.0), (32, 2.0), (128, 4.0)):
            self.row(n=n, min_ms=ms)
            self.row(n=n, min_ms=ms / 2, transfers="none")
        self.row(n=8, min_ms=3.0, package="julia_gpu")
        self.row(n=32, min_ms=6.0, package="julia_gpu")
        dropped = self.row(n=128, min_ms=9.0, package="julia_gpu", errored_pct=50.0)
        untimed = self.row(n=512, min_ms=NAN, package="julia_gpu")
        # A stepping with one point is not a curve.
        self.row(n=8, min_ms=5.0, dt=2.0 ** -8)
        written, skipped = timing.run(self.store, "n", where="problem = 'lorenz'", out=self.out)
        self.assertEqual(skipped, 1)
        names = self.written()
        self.assertEqual(len(names), 4)
        self.assertEqual(sorted(os.path.basename(p) for p in written), names)
        self.assertEqual({n.rsplit(".", 1)[1] for n in names}, {"csv", "png"})
        both = [n for n in names if n.startswith("timing_n_both_tsit5_fixed_") and n.endswith(".csv")]
        self.assertEqual(len(both), 1)
        big = read_csv(os.path.join(self.out, KEY, "lorenz", both[0]))
        self.assertEqual([(r["package"], r["x"], r["min_ms"]) for r in big],
                         [("cubie", "8.0", "1.0"), ("cubie", "32.0", "2.0"), ("cubie", "128.0", "4.0"),
                          ("julia_gpu", "8.0", "3.0"), ("julia_gpu", "32.0", "6.0")])
        self.assertNotIn(dropped["run_id"], [r["run_id"] for r in big])
        self.assertNotIn(untimed["run_id"], [r["run_id"] for r in big])
        # No build time, error, errored share, finals or reason on these rows: their columns are left out.
        self.assertEqual(list(big[0]), [c for c in timing.CSV_COLUMNS if c not in (
            "build_s", "error", "errored_pct", "dt_min", "dt_max", "atol", "rtol", "newton_atol", "newton_rtol",
            "reason", "finals")])
        none = [n for n in names if "_none_" in n and n.endswith(".csv")]
        self.assertEqual([r["min_ms"] for r in read_csv(os.path.join(self.out, KEY, "lorenz", none[0]))],
                         ["0.5", "1.0", "2.0"])

    def test_states_axis_varies_system_params_and_adds_build_seconds(self):
        for states, build in ((4, 10.0), (8, 20.0), (16, 40.0)):
            fields = dict(problem="lorenz96", system_params={"states": states}, n=131072, transfers="none")
            self.store.record(dict(spec(**fields), states=states, min_ms=float(states), build_s=build))
            self.store.record(dict(spec(package="jax", **fields), states=states, min_ms=2.0 * states))
        # The same stepping at another n is another figure, here a lone point.
        self.store.record(dict(spec(problem="lorenz96", system_params={"states": 32}, n=8, transfers="none"),
                               states=32, min_ms=0.1, build_s=5.0))
        # A build without a time stays on the build panel.
        self.store.record(dict(spec(problem="lorenz96", system_params={"states": 64}, n=131072,
                                    transfers="none"), states=64, min_ms=NAN, build_s=80.0))
        written, skipped = timing.run(self.store, "states", set_names=["states"], out=self.out)
        self.assertEqual(skipped, 1)
        names = self.written(problem="lorenz96")
        self.assertEqual(len(names), 2)
        self.assertEqual(len(written), 2)
        table = read_csv(os.path.join(self.out, KEY, "lorenz96", names[0]))
        self.assertEqual([(r["package"], r["x"], r["build_s"]) for r in table],
                         [("cubie", "4.0", "10.0"), ("cubie", "8.0", "20.0"), ("cubie", "16.0", "40.0"),
                          ("jax", "4.0", "nan"), ("jax", "8.0", "nan"), ("jax", "16.0", "nan")])
        _, builds = timing.collect(shared.select_rows(self.store, ["states"]), "states", None)
        self.assertEqual(sorted(builds.values(), key=lambda b: len(b["cubie"])),
                         [{"cubie": [(32.0, 5.0)]},
                          {"cubie": [(4.0, 10.0), (8.0, 20.0), (16.0, 40.0), (64.0, 80.0)]}])

    def test_error_axis_sweeps_the_stepping_and_scores_each_row_against_the_golden(self):
        self.golden()
        for k, offset in ((3, 0.125), (4, 0.0625), (5, 0.03125)):
            self.finals_row(offset=offset, n=64, dt=2.0 ** -k, transfers="none", min_ms=float(k))
            self.finals_row(offset=offset / 2, n=64, dt=2.0 ** -k, transfers="none", min_ms=2.0 * k,
                            package="julia_gpu")
        # No finals: no error, so not on the axis.
        self.row(n=64, dt=2.0 ** -6, transfers="none", min_ms=6.0)
        written, skipped = timing.run(self.store, "error", where="problem = 'lorenz'", out=self.out)
        self.assertEqual((len(written), skipped), (2, 0))
        table = read_csv([p for p in written if p.endswith(".csv")][0])
        self.assertEqual([(r["package"], r["dt"], r["min_ms"]) for r in table],
                         [("cubie", "0.125", "3.0"), ("cubie", "0.0625", "4.0"), ("cubie", "0.03125", "5.0"),
                          ("julia_gpu", "0.125", "6.0"), ("julia_gpu", "0.0625", "8.0"),
                          ("julia_gpu", "0.03125", "10.0")])
        errors = [float(r["x"]) for r in table]
        self.assertEqual(errors, [float(r["error"]) for r in table])
        for value, expected in zip(errors, (0.125, 0.0625, 0.03125, 0.0625, 0.03125, 0.015625)):
            self.assertAlmostEqual(value / (expected / math.sqrt(3)), 1.0, places=3)
        self.assertTrue(os.path.basename(written[0]).startswith("timing_error_none_tsit5_fixed_"))


class AgreementTests(AnalysesCase):
    def test_errors_and_pairwise_differences_per_stepping(self):
        golden = self.golden()
        rows = {}
        for k, offset in ((3, 0.125), (4, 0.0625)):
            rows[("cubie", k)] = self.finals_row(offset=offset, n=64, dt=2.0 ** -k, transfers="none")
            # The both row of the same trial shares the finals and is counted once.
            both = {c: v for c, v in rows[("cubie", k)].items() if c not in store.ID_FIELDS}
            self.store.record(dict(both, transfers="both"))
            rows[("julia_gpu", k)] = self.finals_row(offset=-offset, n=64, dt=2.0 ** -k, transfers="none",
                                                     package="julia_gpu")
        rows[("jax", 3)] = self.finals_row(offset=0.0, n=64, dt=2.0 ** -3, transfers="none", package="jax",
                                           errored_pct=50.0)
        written = agreement.run(self.store, set_names=["golden_grid"], out=self.out)
        names = self.written()
        self.assertEqual(names[:2], ["agreement.csv", "agreement_pairs.csv"])
        self.assertEqual(len(names), 3)
        self.assertTrue(names[2].startswith("agreement_tsit5_fixed_") and names[2].endswith(".png"))
        self.assertEqual(sorted(os.path.basename(p) for p in written), names)
        table = read_csv(os.path.join(self.out, KEY, "lorenz", "agreement.csv"))
        self.assertEqual(list(table[0]), [c for c in agreement.ROW_COLUMNS if c not in (
            "dt_min", "dt_max", "atol", "rtol", "newton_atol", "newton_rtol")])
        self.assertEqual([(r["package"], r["dt"]) for r in table],
                         [("cubie", "0.125"), ("julia_gpu", "0.125"), ("cubie", "0.0625"), ("julia_gpu", "0.0625")])
        for record, expected in zip(table, (0.125, 0.125, 0.0625, 0.0625)):
            self.assertAlmostEqual(float(record["error"]) / (expected / math.sqrt(3)), 1.0, places=3)
        pairs = read_csv(os.path.join(self.out, KEY, "lorenz", "agreement_pairs.csv"))
        self.assertEqual(list(pairs[0]), [c for c in agreement.PAIR_COLUMNS if c not in (
            "dt_min", "dt_max", "atol", "rtol", "newton_atol", "newton_rtol")])
        self.assertEqual([(p["package_a"], p["package_b"], p["dt"]) for p in pairs],
                         [("cubie", "julia_gpu", "0.125"), ("cubie", "julia_gpu", "0.0625")])
        for record, expected in zip(pairs, (0.25, 0.125)):
            self.assertAlmostEqual(float(record["difference"]) / (expected / math.sqrt(3)), 1.0, places=3)
        self.assertNotIn(golden["package"], {r["package"] for r in table})
        self.assertNotIn("jax", {r["package"] for r in table})

    def test_no_finals_writes_nothing(self):
        self.row(n=8)
        self.assertEqual(agreement.run(self.store, where="n = 8", out=self.out), [])
        self.assertEqual(agreement.main(["--no-sync", "--where", "n = 8", "--root", self.root, "--out", self.out]), 0)


if __name__ == "__main__":
    unittest.main()
