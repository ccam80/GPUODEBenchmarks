"""The analysis script: the store read with two rows of one run_id refused, selection by predicate over the rows (every row without one) and by figure kind, the errored and untimed filters, CSVs without the columns no row captured, and the base figures of a (key, problem, algorithm): runtime against n, error against runtime, error against dt, error against tolerance, and runtime with cold build time against the state count."""

import csv
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

import grid  # noqa: E402
import store  # noqa: E402
import errors  # noqa: E402
import plots  # noqa: E402
import shared  # noqa: E402

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


def adaptive(tol, **overrides):
    """The default controller at a tolerance, dt0 = 2^-10."""
    return dict(controller="default", atol=tol, rtol=tol, **overrides)


def sid(row):
    """'<package> <controller kind>[ +]' of a CSV row: the series without its display text."""
    split = row["kind"] in ("runtime_vs_n", "error_vs_runtime", "states")
    return "{0} {1}{2}".format(row["package"], row["controller"], " +" if split and row["transfers"] == "both" else "")


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

    def written(self, key=KEY):
        """'<kind>/[limited_data/]<name>' of every file under a key's kind directories, sorted."""
        directory = os.path.join(self.out, key)
        if not os.path.isdir(directory):
            return []
        return sorted(os.path.relpath(os.path.join(base, name), directory).replace(os.sep, "/")
                      for base, _, names in os.walk(directory) for name in names if base != directory)

    def analysis_store(self):
        return shared.AnalysisStore(self.root)

    def table(self, name, key=KEY, problem="lorenz"):
        return read_csv(os.path.join(self.out, key, name, problem + ".csv"))


class SelectionTests(AnalysesCase):
    def test_rows_are_selected_by_predicate_over_the_rows_or_every_row(self):
        perf = [self.row(n=n) for n in (8, 32)]
        other_key = self.row(n=8, key=OTHER_KEY)
        golden_only = self.row(n=131072, dt=2.0 ** -3, transfers="none")
        alien = self.row(algorithm="euler", dt=2.0 ** -20)
        ids = lambda rows: sorted(r["run_id"] for r in rows)  # noqa: E731
        self.assertEqual(ids(shared.select_rows(self.store)), ids(perf + [other_key, golden_only, alien]))
        # A predicate names rows and nothing else comes along: not the other n of a sweep, not the other key.
        self.assertEqual(ids(shared.select_rows(self.store, where="n = 32")), ids(perf[1:]))
        self.assertEqual(ids(shared.select_rows(self.store, where="algorithm = 'euler'")), ids([alien]))
        self.assertEqual(ids(shared.select_rows(self.store, where="key = '{0}'".format(OTHER_KEY))), ids([other_key]))
        self.assertEqual(ids(shared.select_rows(self.store, where="problem = 'lorenz' AND algorithm = 'tsit5' AND n = 8")),
                         ids([perf[0], other_key]))

    def test_errored_and_untimed_filters(self):
        self.assertTrue(shared.within_errored_limit({"errored_pct": NAN}))
        self.assertTrue(shared.within_errored_limit({"errored_pct": None}))
        self.assertTrue(shared.within_errored_limit({}))
        self.assertTrue(shared.within_errored_limit({"errored_pct": 10.0}))
        self.assertFalse(shared.within_errored_limit({"errored_pct": 10.5}))
        self.assertFalse(shared.timed({"min_ms": NAN}))
        self.assertTrue(shared.timed({"min_ms": 0.5}))

    def test_a_csv_leaves_out_the_columns_no_row_captured(self):
        rows = [{"a": 1.0, "b": NAN, "c": "", "d": None, "e": [], "f": "x"},
                {"a": NAN, "b": NAN, "c": "", "d": None, "e": [1.0], "f": ""}]
        self.assertEqual(shared.captured_columns(("a", "b", "c", "d", "e", "f", "g"), rows), ["a", "e", "f"])
        self.assertEqual(shared.captured_columns(("a", "b"), []), ["a", "b"])
        path = shared.write_csv(os.path.join(self.tmp, "t.csv"), ("a", "b", "c", "d", "e", "f", "g"), rows)
        self.assertEqual(read_csv(path), [{"a": "1.0", "e": "", "f": "x"}, {"a": "nan", "e": "1.0", "f": ""}])

    def test_the_flags_take_a_predicate_and_kinds_and_refuse_an_unknown_kind(self):
        args = shared.parser("", plots.KIND_NAMES).parse_args(["--no-sync"])
        self.assertEqual((args.where, args.kind), ("", []))
        args = shared.parser("", plots.KIND_NAMES).parse_args(
            ["--where", "n = 8", "--kind", "error_vs_runtime", "--kind", "states"])
        self.assertEqual((args.where, args.kind), ("n = 8", ["error_vs_runtime", "states"]))
        with self.assertRaises(SystemExit):
            shared.parser("", plots.KIND_NAMES).parse_args(["--kind", "work_precision"])
        for gone in ("check_selection", "report_incomplete", "incomplete", "canonical_trials", "selection_pairs"):
            self.assertFalse(hasattr(shared, gone), gone)


class PlotTests(AnalysesCase):
    def test_runtime_against_n_is_one_figure_per_algorithm_with_a_series_per_package_controller_and_transfers(self):
        for n, ms in ((8, 1.0), (32, 2.0), (128, 4.0), (512, 8.0)):
            self.row(n=n, min_ms=ms)
            self.row(n=n, min_ms=ms / 2, transfers="none")
            self.row(n=n, min_ms=3.0 * ms, package="jax", **adaptive(1e-5))
            self.row(n=n, min_ms=9.0 * ms, **adaptive(1e-5))
        self.row(n=8, min_ms=3.0, package="julia_gpu")
        self.row(n=32, min_ms=6.0, package="julia_gpu")
        crossed = self.row(n=128, min_ms=9.0, package="julia_gpu", errored_pct=50.0)
        untimed = self.row(n=512, min_ms=NAN, package="julia_gpu")
        cpu = self.row(n=8, min_ms=100.0, package="julia_cpu", transfers="none")
        self.row(n=32, min_ms=200.0, package="julia_cpu", transfers="none")
        # A stepping with one point is not a curve; another algorithm is another figure.
        self.row(n=8, min_ms=5.0, dt=2.0 ** -8)
        self.row(n=8, min_ms=5.0, algorithm="euler")
        self.row(n=32, min_ms=6.0, algorithm="euler")
        written = plots.run(self.analysis_store(), where="problem = 'lorenz'", out=self.out)
        # euler has one package: limited data. The grids hold every algorithm and every problem.
        self.assertEqual(self.written(), ["runtime_vs_n/euler_problems.png", "runtime_vs_n/limited_data/lorenz_euler.png",
                                          "runtime_vs_n/lorenz.csv", "runtime_vs_n/lorenz.png",
                                          "runtime_vs_n/lorenz_algorithms.png", "runtime_vs_n/lorenz_tsit5.png",
                                          "runtime_vs_n/tsit5_problems.png"])
        self.assertEqual(sorted(os.path.relpath(p, os.path.join(self.out, KEY)).replace(os.sep, "/") for p in written),
                         self.written())
        table = self.table("runtime_vs_n")
        tsit5 = [r for r in table if r["algorithm"] == "tsit5"]
        self.assertEqual([(sid(r), r["x"], r["y"]) for r in tsit5],
                         [("cubie fixed +", "8.0", "0.001"),
                          ("cubie fixed +", "32.0", "0.002"),
                          ("cubie fixed +", "128.0", "0.004"),
                          ("cubie fixed +", "512.0", "0.008"),
                          ("cubie fixed", "8.0", "0.0005"),
                          ("cubie fixed", "32.0", "0.001"),
                          ("cubie fixed", "128.0", "0.002"),
                          ("cubie fixed", "512.0", "0.004"),
                          ("cubie adaptive +", "8.0", "0.009"), ("cubie adaptive +", "32.0", "0.018"),
                          ("cubie adaptive +", "128.0", "0.036"), ("cubie adaptive +", "512.0", "0.072"),
                          ("jax adaptive +", "8.0", "0.003"), ("jax adaptive +", "32.0", "0.006"),
                          ("jax adaptive +", "128.0", "0.012"), ("jax adaptive +", "512.0", "0.024"),
                          ("julia_gpu fixed +", "8.0", "0.003"),
                          ("julia_gpu fixed +", "32.0", "0.006"),
                          ("julia_gpu fixed +", "128.0", "0.009")])
        self.assertEqual([sid(r) for r in table if r["algorithm"] == "euler"],
                         ["cubie fixed +"] * 2)
        self.assertFalse(os.path.isdir(os.path.join(self.out, plots.ALL_CARDS)))
        # The row over the errored limit stays (drawn crossed out); the untimed and CPU rows do not.
        self.assertIn(crossed["run_id"], [r["run_id"] for r in table])
        for absent in (untimed, cpu):
            self.assertNotIn(absent["run_id"], [r["run_id"] for r in table])
        # No build time, error, finals or reason on these rows: their columns are left out.
        self.assertEqual(list(table[0]), [c for c in plots.CSV_COLUMNS if c not in (
            "build_s", "error", "interval_error", "trace_errored_pct", "dt_min", "dt_max", "newton_atol", "newton_rtol", "reason",
            "finals", "traces")])

    def test_a_series_keeps_the_stepping_with_the_most_points(self):
        for n in (8, 32, 128):
            self.row(n=n, dt=2.0 ** -10)
            self.row(n=n, package="jax", **adaptive(1e-3))
        for n in (8, 32):
            self.row(n=n, dt=2.0 ** -8)
            self.row(n=n, package="jax", **adaptive(1e-5))
        plots.run(self.analysis_store(), where="problem = 'lorenz'", out=self.out)
        table = self.table("runtime_vs_n")
        self.assertEqual(sorted({(sid(r), r["dt"], r["atol"]) for r in table}),
                         [("cubie fixed +", "0.0009765625", "nan"),
                          ("jax adaptive +", "0.0009765625", "0.001")])
        self.assertEqual(len(table), 6)

    def test_states_figure_has_runtime_and_cold_build_panels(self):
        for states, build in ((4, 10.0), (8, 20.0), (16, 40.0)):
            fields = dict(problem="lorenz96", system_params={"states": states}, n=131072, **adaptive(1e-5))
            self.store.record(dict(spec(package="julia_gpu", transfers="none", **fields), states=states,
                                   min_ms=float(states), build_s=build))
            # The both row of the trial records the build again; the panel counts it once.
            self.store.record(dict(spec(package="julia_gpu", **fields), states=states, min_ms=2.0 * states, build_s=build))
            self.store.record(dict(spec(package="jax", transfers="none", **fields), states=states, min_ms=3.0 * states))
        # The same stepping at another n is a lone point.
        self.store.record(dict(spec(package="julia_gpu", problem="lorenz96", system_params={"states": 32}, n=8,
                                    transfers="none", **adaptive(1e-5)), states=32, min_ms=0.1, build_s=5.0))
        # A build without a time stays on the build panel.
        self.store.record(dict(spec(package="julia_gpu", problem="lorenz96", system_params={"states": 64}, n=131072,
                                    transfers="none", **adaptive(1e-5)), states=64, min_ms=NAN, build_s=80.0))
        written = plots.run(self.analysis_store(), where="problem = 'lorenz96'", out=self.out)
        # Three timed points per package, but the builds panel has four: not limited data.
        self.assertEqual(self.written(), ["states/lorenz96.csv", "states/lorenz96_algorithms.png",
                                          "states/lorenz96_tsit5.png", "states/tsit5_problems.png"])
        self.assertEqual(len(written), 4)
        table = self.table("states", problem="lorenz96")
        self.assertEqual([(sid(r), r["x"], r["y"]) for r in table if r["kind"] == "states"],
                         [("jax adaptive", "4.0", "0.012"), ("jax adaptive", "8.0", "0.024"),
                          ("jax adaptive", "16.0", "0.048"),
                          ("julia_gpu adaptive +", "4.0", "0.008"),
                          ("julia_gpu adaptive +", "8.0", "0.016"),
                          ("julia_gpu adaptive +", "16.0", "0.032"),
                          ("julia_gpu adaptive", "4.0", "0.004"),
                          ("julia_gpu adaptive", "8.0", "0.008"),
                          ("julia_gpu adaptive", "16.0", "0.016")])
        self.assertEqual([(sid(r), r["x"], r["y"]) for r in table if r["kind"] == "builds"],
                         [("julia_gpu adaptive", "4.0", "10.0"), ("julia_gpu adaptive", "8.0", "20.0"),
                          ("julia_gpu adaptive", "16.0", "40.0"), ("julia_gpu adaptive", "64.0", "80.0")])

    def error_sweep(self):
        """A golden and, at n = 64, cubie and julia_gpu fixed steppings at dt = 2^-3..2^-5 and default steppings at three tolerances, plus julia_cpu at the tolerances; cubie's both rows share the none rows' finals."""
        self.golden()
        for k, offset in ((3, 0.125), (4, 0.0625), (5, 0.03125)):
            row = self.finals_row(offset=offset, n=64, dt=2.0 ** -k, transfers="none", min_ms=float(k))
            self.store.record(dict({c: v for c, v in row.items() if c not in store.ID_FIELDS}, transfers="both",
                                   min_ms=2.0 * k))
            self.finals_row(offset=offset / 2, n=64, dt=2.0 ** -k, transfers="none", min_ms=2.0 * k,
                            package="julia_gpu")
        for tol, offset, ms in ((1e-3, 0.1, 1000.0), (1e-4, 0.01, 10000.0), (1e-5, 0.001, 100000.0)):
            self.finals_row(offset=offset, n=64, transfers="none", min_ms=ms, package="julia_gpu", **adaptive(tol))
            self.finals_row(offset=offset, n=64, transfers="none", min_ms=NAN, package="julia_cpu", **adaptive(tol))
        # No finals: no error, so on no error axis.
        self.row(n=64, dt=2.0 ** -6, transfers="none", min_ms=6.0)

    def test_error_against_dt_and_tolerance_take_every_package_once_per_trial(self):
        self.error_sweep()
        plots.run(self.analysis_store(), where="problem = 'lorenz'", out=self.out)
        self.assertEqual([n for n in self.written() if n.endswith(".png")],
                         ["error_vs_dt/limited_data/lorenz_tsit5.png", "error_vs_runtime/limited_data/lorenz_tsit5.png",
                          "error_vs_runtime/lorenz.png", "error_vs_runtime/lorenz_algorithms.png",
                          "error_vs_runtime/tsit5_problems.png", "error_vs_tol/limited_data/lorenz_tsit5.png"])
        by_dt = self.table("error_vs_dt")
        self.assertEqual([(sid(r), r["x"], r["min_ms"]) for r in by_dt],
                         [("cubie fixed", "0.03125", "5.0"), ("cubie fixed", "0.0625", "4.0"),
                          ("cubie fixed", "0.125", "3.0"),
                          ("julia_gpu fixed", "0.03125", "10.0"), ("julia_gpu fixed", "0.0625", "8.0"),
                          ("julia_gpu fixed", "0.125", "6.0")])
        for record, expected in zip(by_dt, (0.03125, 0.0625, 0.125, 0.015625, 0.03125, 0.0625)):
            self.assertAlmostEqual(float(record["y"]) / (expected / math.sqrt(3)), 1.0, places=3)
            self.assertEqual(record["y"], record["error"])
        by_tol = self.table("error_vs_tol")
        self.assertEqual([(sid(r), r["x"]) for r in by_tol],
                         [("julia_gpu adaptive", "1e-05"), ("julia_gpu adaptive", "0.0001"),
                          ("julia_gpu adaptive", "0.001"),
                          ("julia_cpu adaptive", "1e-05"),
                          ("julia_cpu adaptive", "0.0001"),
                          ("julia_cpu adaptive", "0.001")])
        self.assertNotIn("julia_cpu", {r["package"] for r in self.table("error_vs_runtime")})

    def test_error_against_runtime_follows_each_sweep_from_loose_to_tight(self):
        self.error_sweep()
        plots.run(self.analysis_store(), where="problem = 'lorenz'", out=self.out)
        table = self.table("error_vs_runtime")
        swept = lambda r: r["atol"] if r["controller"] == "adaptive" else r["dt"]  # noqa: E731
        self.assertEqual([(sid(r), r["x"], swept(r)) for r in table],
                         [("cubie fixed +", "0.006", "0.125"),
                          ("cubie fixed +", "0.008", "0.0625"),
                          ("cubie fixed +", "0.01", "0.03125"),
                          ("cubie fixed", "0.003", "0.125"),
                          ("cubie fixed", "0.004", "0.0625"),
                          ("cubie fixed", "0.005", "0.03125"),
                          ("julia_gpu fixed", "0.006", "0.125"),
                          ("julia_gpu fixed", "0.008", "0.0625"),
                          ("julia_gpu fixed", "0.01", "0.03125"),
                          ("julia_gpu adaptive", "1.0", "0.001"),
                          ("julia_gpu adaptive", "10.0", "0.0001"),
                          ("julia_gpu adaptive", "100.0", "1e-05")])
        for record in table:
            self.assertTrue(errors.is_finite_positive(float(record["y"])))

    def test_a_figure_comparing_nothing_is_limited_data(self):
        four = {(KEY, package, "fixed", "both"): [(float(n), 1.0, {}) for n in (8, 32, 128, 512)]
                for package in ("cubie", "jax")}
        self.assertFalse(plots.limited(four))
        self.assertTrue(plots.limited({k: v[:3] for k, v in four.items()}))
        cubies = {(KEY, "cubie", "fixed", transfers): four[(KEY, "cubie", "fixed", "both")]
                  for transfers in ("both", "none")}
        self.assertTrue(plots.limited(cubies))
        self.assertTrue(plots.limited({}))
        # The builds panel counts.
        self.assertFalse(plots.limited(cubies, {(KEY, "jax", "fixed", ""): four[(KEY, "jax", "fixed", "both")]}))

    def test_two_keys_share_an_all_cards_tree_with_a_marker_set_per_key(self):
        for key in (KEY, OTHER_KEY):
            for n, ms in ((8, 1.0), (32, 2.0), (128, 4.0), (512, 8.0)):
                self.row(n=n, min_ms=ms, key=key)
                self.row(n=n, min_ms=3.0 * ms, key=key, package="jax", **adaptive(1e-5))
        written = plots.run(self.analysis_store(), where="problem = 'lorenz'", out=self.out)
        self.assertEqual(self.written(), ["runtime_vs_n/lorenz.csv", "runtime_vs_n/lorenz.png",
                                          "runtime_vs_n/lorenz_algorithms.png", "runtime_vs_n/lorenz_tsit5.png",
                                          "runtime_vs_n/tsit5_problems.png"])
        self.assertEqual(self.written(OTHER_KEY), self.written())
        self.assertEqual(self.written(plots.ALL_CARDS), self.written())
        self.assertEqual(len(written), 15)
        table = self.table("runtime_vs_n", key=plots.ALL_CARDS)
        self.assertEqual([(r["key"], sid(r)) for r in table][::4],
                         [(OTHER_KEY, "cubie fixed +"), (OTHER_KEY, "jax adaptive +"),
                          (KEY, "cubie fixed +"), (KEY, "jax adaptive +")])
        series = plots.series_of(plots.KINDS[0], plots.with_errors(self.analysis_store().rows(), errors.Errors(self.store)))
        self.assertEqual(plots.cards_of(series), [OTHER_KEY, KEY])

    def test_the_kinds_named_are_written_and_every_kind_without_a_name(self):
        self.error_sweep()
        for n, ms in ((8, 1.0), (32, 2.0), (128, 4.0), (512, 8.0)):
            self.row(n=n, min_ms=ms, transfers="none")
            self.row(n=n, min_ms=3.0 * ms, package="jax", transfers="none", **adaptive(1e-5))
        written = plots.run(self.analysis_store(), kinds=["error_vs_runtime"], out=self.out)
        self.assertEqual(self.written(), ["error_vs_runtime/limited_data/lorenz_tsit5.png",
                                          "error_vs_runtime/lorenz.csv", "error_vs_runtime/lorenz.png",
                                          "error_vs_runtime/lorenz_algorithms.png", "error_vs_runtime/tsit5_problems.png"])
        self.assertEqual(len(written), 5)
        plots.run(self.analysis_store(), where="problem = 'lorenz' AND algorithm = 'tsit5'",
                  kinds=["runtime_vs_n", "error_vs_dt"], out=self.out)
        self.assertEqual({n.partition("/")[0] for n in self.written()},
                         {"error_vs_runtime", "runtime_vs_n", "error_vs_dt"})
        shutil.rmtree(self.out)
        self.assertEqual(plots.main(["--no-sync", "--root", self.root, "--out", self.out]), 0)
        self.assertEqual({n.partition("/")[0] for n in self.written()},
                         {"error_vs_runtime", "runtime_vs_n", "error_vs_dt", "error_vs_tol"})
        shutil.rmtree(self.out)
        self.assertEqual(plots.main(["--no-sync", "--kind", "error_vs_tol", "--root", self.root, "--out", self.out]), 0)
        self.assertEqual({n.partition("/")[0] for n in self.written()}, {"error_vs_tol"})

    def test_no_curve_writes_nothing(self):
        self.row(n=8)
        self.assertEqual(plots.run(self.analysis_store(), where="n = 8", out=self.out), [])
        self.assertEqual(plots.main(["--no-sync", "--where", "n = 8", "--root", self.root, "--out", self.out]), 0)


if __name__ == "__main__":
    unittest.main()
