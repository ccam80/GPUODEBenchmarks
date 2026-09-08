"""convert_legacy.py on a synthetic legacy tree: every rule's counts, the merged rows, the finals, the controllers table, the deletions, and the DuckDB check."""

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
sys.path.insert(0, os.path.dirname(HERE))

import convert_legacy  # noqa: E402
import store  # noqa: E402
from problems import get_problem  # noqa: E402
from protocol import N_NE, N_WP, TIMING_TOL  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
LORENZ = get_problem("lorenz")
DT = LORENZ.timing_dt
RESULTS_HEADER = ("package,key,analysis,problem,algorithm,mode,setting_kind,setting,n,states,"
                  "tier,transfers,min_ms,samples_ms,errored_pct,error,build_s,recorded_utc")


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


class LegacyTree:
    """A data/ root holding one of every legacy file kind."""

    def __init__(self, root):
        self.root = root
        self.write_results()
        self.write_overlap()
        self.write_ne_julia()
        self.write_numerical()

    def results_row(self, package, analysis, algorithm, mode, kind, setting, n, transfers,
                    min_ms, samples="", error="nan", stamp="2026-09-08T05:44:47Z", states=3,
                    problem="lorenz"):
        return ",".join(str(v) for v in (
            package, KEY, analysis, problem, algorithm, mode, kind, format(setting, ".10g"), n,
            states, "default", transfers, min_ms, samples, "nan", error, "nan", stamp))

    def write_results(self):
        julia = [RESULTS_HEADER,
                 self.results_row("julia", "times", "tsit5", "fixed", "dt", DT, 8, "both", 1.5, "9;2;1.5"),
                 self.results_row("julia", "times", "tsit5", "fixed", "dt", DT, 8, "none", 0.5, "3;0.5;0.6"),
                 self.results_row("julia", "times", "tsit5", "fixed", "dt", DT, 32768, "both", 2.5, "9;2.5"),
                 self.results_row("julia", "times", "tsit5", "fixed", "dt", DT, 32768, "none", 2.0, "9;2"),
                 self.results_row("julia", "times", "tsit5", "fixed", "dt", DT, N_WP, "none", 4.0, "9;4"),
                 self.results_row("julia", "wp", "tsit5", "fixed", "dt", DT, N_WP, "d2h", 4.5, "9;4.5", error="1e-3"),
                 self.results_row("julia", "wp", "tsit5", "fixed", "dt", DT / 2, N_WP, "d2h", 6.5, "", error="2e-4"),
                 self.results_row("julia", "states", "tsit5", "fixed", "dt", DT, N_WP, "both", 7.0, "9;7",
                                  states=32, problem="lorenz96")]
        write_lines(os.path.join(self.root, "Julia", KEY, "results.csv"), julia)
        jax = [RESULTS_HEADER,
               self.results_row("jax", "times", "tsit5", "fixed", "dt", DT, 32768, "both", 3.0, "9;3"),
               self.results_row("jax", "times", "tsit5", "fixed", "dt", DT, 32768, "none", 2.9, "9;2.9"),
               self.results_row("jax", "times", "tsit5", "fixed", "dt", DT, N_WP, "none", 5.0, "9;5",
                                stamp="2026-09-08T05:44:40Z"),
               self.results_row("jax", "wp", "tsit5", "fixed", "dt", DT, N_WP, "none", 5.5, "", error="3e-4",
                                stamp="2026-09-08T05:44:50Z"),
               self.results_row("jax", "wp", "kvaerno3", "adaptive", "tol", 1e-3, N_WP, "none", 8.0, "9;8", error="1e-2"),
               self.results_row("jax", "times", "kvaerno3", "adaptive", "tol", TIMING_TOL, 8, "both", 8.0, "9;8")]
        write_lines(os.path.join(self.root, "JAX", KEY, "results.csv"), jax)
        cpp = [RESULTS_HEADER,
               self.results_row("cpp", "times", "classical-rk4", "fixed", "dt", DT, 8, "both", 0.1, "1;0.1")]
        write_lines(os.path.join(self.root, "CPP", KEY, "results.csv"), cpp)

    def write_overlap(self):
        base = os.path.join(self.root, "cubie_julia_overlap", KEY, "lorenz")
        header = "framework,algorithm,phase,mode,tier,transfers,n,setting_kind,setting,samples,min_ms,p05_ms,median_ms,p95_ms,max_ms"
        rows = [header,
                "julia,tsit5,performance,fixed,fixed,both,8,dt,{0},20,1.7,1.8,1.9,2.0,2.1".format(DT),
                "julia,tsit5,performance,fixed,fixed,none,8,dt,{0},20,0.7,0.8,0.9,1.0,1.1".format(DT),
                "julia,tsit5,performance,adaptive,julia,both,8,tol,1.0e-8,20,1.6,1.7,1.8,1.9,2.0",
                "julia,kvaerno3,performance,adaptive,pi,none,8,tol,1.0e-8,20,1.1,1.2,1.3,1.4,1.5",
                "julia,tsit5,work_precision,adaptive,julia,none,{0},tol,0.001,20,1.3,1.4,1.5,1.6,1.7".format(N_WP),
                "julia,tsit5,work_precision,adaptive,julia,none,2048,tol,0.001,20,1.3,1.4,1.5,1.6,1.7",
                "julia,tsit5,numerical,adaptive,julia,none,{0},tol,0.001,20,0.3,0.4,0.5,0.6,0.7".format(N_NE)]
        write_lines(os.path.join(base, "julia_timings.csv"), rows)
        write_lines(os.path.join(base, "timing_summary.csv"), rows)
        write_lines(os.path.join(base, "work_precision.csv"), [
            "framework,algorithm,mode,tier,transfers,n,setting_kind,setting,samples,min_ms,p05_ms,median_ms,p95_ms,max_ms,golden_rmse,finite_trajectories,failed_trajectories",
            "julia,tsit5,adaptive,julia,none,{0},tol,0.001,20,1.3,1.4,1.5,1.6,1.7,0.0049,{1},{2}".format(
                N_WP, N_WP - 8, 8)])
        write_lines(os.path.join(base, "julia_metrics.csv"), [
            "framework,algorithm,phase,mode,tier,n,setting_kind,setting,golden_rmse,finite_trajectories,failed_trajectories,finals_path",
            "julia,tsit5,performance,fixed,fixed,8,dt,{0},,8,0,".format(DT),
            "julia,tsit5,performance,adaptive,julia,8,tol,1.0e-8,,6,2,",
            "julia,tsit5,numerical,adaptive,julia,{0},tol,0.001,0.001,{0},0,finals/julia/tsit5/x.csv".format(N_NE)])
        write_lines(os.path.join(base, "julia_failures.csv"), [
            "framework,algorithm,phase,mode,tier,n,setting_kind,setting,error_type,message",
            "julia,vern7,performance,adaptive,julia,8,tol,1.0e-8,CuError,boom"])
        write_lines(os.path.join(base, "finals", "julia", "tsit5", "adaptive_julia_tol_0p001.csv"),
                    ["traj,s1,s2,s3", "0,1,2,3"])
        with open(os.path.join(base, "julia_metadata.json"), "w") as handle:
            json.dump({"framework": "DiffEqGPU", "diffeqgpu_version": "3.17.0"}, handle)
        write_lines(os.path.join(self.root, "cubie_julia_overlap", KEY, "speedups.csv"), ["a,b"])

    def write_ne_julia(self):
        base = os.path.join(self.root, "numerical_equivalence", "julia", KEY, "lorenz")
        fixed = ["dt,traj,s1,s2,s3,converged"]
        for dt in (0.5, 0.25):
            fixed += ["{0},{1},{2},{3},{4},{5}".format(dt, traj, traj * 0.5, dt, 1.0, 1 if traj % 2 else 0)
                      for traj in range(N_NE)]
        write_lines(os.path.join(base, "kvaerno3.csv"), fixed)
        adaptive = ["tol,traj,s1,s2,s3,naccept,nreject,converged"]
        adaptive += ["0.01,{0},{1},2,3,9,0,1".format(traj, traj) for traj in range(N_NE)]
        adaptive += ["0.001,{0},{1},2,3,9,0,1".format(traj, traj) for traj in range(N_NE - 1)]
        write_lines(os.path.join(base, "kvaerno3_adaptive.csv"), adaptive)
        write_lines(os.path.join(base, "controller_constants.csv"), [
            "cubie_alias,controller,beta1,beta2,qmin,qmax,gamma,order",
            "kvaerno3,PIController,0.23333333,0.13333334,0.2,10.0,0.9,3"])

    def write_numerical(self):
        base = os.path.join(self.root, "numerical", KEY, "lorenz")
        os.makedirs(base, exist_ok=True)
        finals = np.arange(32768 * 3, dtype=np.float64).reshape(32768, 3) / 7.0
        finals[5, 1] = np.nan
        np.savetxt(os.path.join(base, "jax.csv"), finals, delimiter=",")
        np.savetxt(os.path.join(base, "pytorch.csv"), finals, delimiter=",")
        np.savetxt(os.path.join(base, "julia_fixed.csv"), finals, delimiter=",")
        np.savetxt(os.path.join(base, "mpgos.csv"), finals, delimiter=",")
        np.savetxt(os.path.join(base, "myokit_cuda.csv"), finals[:10], delimiter=",")
        golden = os.path.join(self.root, "numerical", "golden_lorenz_131072.csv")
        write_lines(golden, ["1,2,3"])
        self.golden = golden


class ConvertLegacyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="convert_legacy_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tree = LegacyTree(self.tmp)
        self.conversion = convert_legacy.Conversion(self.tmp).run()
        self.expected, self.found = self.conversion.verify()
        self.conversion.delete_old()
        self.store = store.Store(self.tmp)
        self.counts = self.conversion.counts

    def rows(self, **filters):
        return self.store.rows(**filters)

    def test_results_rows_convert_with_the_package_renamed_and_the_drops_applied(self):
        self.assertEqual(self.counts["results.csv cpp rows dropped"], 1)
        self.assertEqual(self.counts["results.csv jax kvaerno3 rows dropped"], 2)
        self.assertEqual(self.counts["results.csv wp rows with transfers != none dropped"], 2)
        self.assertEqual(self.rows(package="cpp"), [])
        self.assertEqual(self.rows(algorithm="kvaerno3", package="jax"), [])
        julia = self.rows(package="julia_gpu", n=8, transfers="both", mode="fixed")
        self.assertEqual(len(julia), 1)
        self.assertEqual(julia[0]["samples_ms"], [9.0, 2.0, 1.5])
        self.assertNotIn("analysis", julia[0])
        self.assertEqual(julia[0]["min_ms"], 1.5)
        states = self.rows(problem="lorenz96")
        self.assertEqual((states[0]["states"], states[0]["package"]), (32, "julia_gpu"))

    def test_identity_collisions_keep_the_sampled_row_and_fill_its_gaps(self):
        # jax times (samples, earlier) vs jax wp (no samples, later, error): the sampled row wins, the error fills in.
        row = self.rows(package="jax", n=N_WP, transfers="none")[0]
        self.assertEqual(row["min_ms"], 5.0)
        self.assertEqual(row["samples_ms"], [9.0, 5.0])
        self.assertAlmostEqual(row["error"], 3e-4)
        self.assertEqual(self.counts["identity collisions merged (results.csv times vs results.csv wp)"], 1)
        # overlap performance fixed at n = 8 meets the results.csv julia times row: the sampled row stands.
        self.assertEqual(self.counts["identity collisions merged (overlap performance vs results.csv times)"], 2)
        both = self.rows(package="julia_gpu", n=8, transfers="both", mode="fixed")[0]
        self.assertEqual(both["min_ms"], 1.5)
        self.assertEqual(both["package_version"], "3.17.0")

    def test_overlap_rows_take_the_tier_map_the_error_and_no_samples(self):
        self.assertEqual(self.counts["overlap performance rows converted"], 4)
        self.assertEqual(self.counts["overlap work_precision rows converted"], 1)
        self.assertEqual(self.counts["overlap numerical phase rows dropped"], 1)
        self.assertEqual(self.counts["overlap work_precision rows off n = {0} dropped".format(N_WP)], 1)
        self.assertEqual(self.counts["overlap failure records dropped"], 1)
        self.assertEqual(self.counts["overlap finals files dropped"], 1)
        self.assertEqual(self.counts["overlap derived tables dropped"], 2)
        adaptive = self.rows(package="julia_gpu", mode="adaptive", n=8)
        self.assertEqual(sorted((r["algorithm"], r["tier"]) for r in adaptive),
                         [("kvaerno3", "pi"), ("tsit5", "default")])
        tsit5 = [r for r in adaptive if r["algorithm"] == "tsit5"][0]
        self.assertEqual(tsit5["samples_ms"], [])
        self.assertEqual(tsit5["min_ms"], 1.6)
        self.assertEqual(tsit5["errored_pct"], 25.0)
        self.assertTrue(math.isnan(tsit5["error"]))
        wp = self.rows(package="julia_gpu", mode="adaptive", n=N_WP)[0]
        self.assertAlmostEqual(wp["error"], 0.0049)
        self.assertAlmostEqual(wp["errored_pct"], 100.0 * 8 / N_WP)
        self.assertEqual(wp["transfers"], "none")
        self.assertEqual(wp["package_version"], "3.17.0")
        self.assertEqual(wp["suite_rev"], "")

    def test_ne_julia_trees_become_julia_cpu_finals_with_untimed_rows(self):
        self.assertEqual(self.counts["NE julia_cpu finals converted"], 3)
        self.assertEqual(self.counts["NE settings without {0} trajectories dropped".format(N_NE)], 1)
        self.assertEqual(self.counts["NE controller tables copied"], 1)
        rows = self.rows(package="julia_cpu")
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["reason"], "untimed")
            self.assertTrue(math.isnan(row["min_ms"]))
            self.assertEqual((row["n"], row["transfers"], row["tier"]), (N_NE, "none", "default"))
        fixed = [r for r in rows if r["mode"] == "fixed" and r["setting"] == 0.25][0]
        self.assertEqual(fixed["finals"], "finals/lorenz__kvaerno3__fixed__dt-0.25__n1024__s3__default.parquet")
        traj, states, converged = self.store.load_finals("julia_cpu", KEY, fixed["finals"])
        self.assertEqual(list(traj[:3]), [0, 1, 2])
        np.testing.assert_allclose(states[3], [1.5, 0.25, 1.0])
        self.assertEqual(list(converged[:2]), [False, True])
        controllers = os.path.join(self.tmp, "key=" + KEY, "package=julia_cpu",
                                   "controllers", "lorenz.csv")
        with open(controllers, newline="") as handle:
            self.assertEqual(next(csv.DictReader(handle))["cubie_alias"], "kvaerno3")

    def test_numerical_finals_attach_to_the_32768_times_row_and_the_rest_drop(self):
        self.assertEqual(self.counts["numerical finals attached"], 1)
        self.assertEqual(self.counts["numerical finals without a times row dropped"], 1)
        self.assertEqual(self.counts["numerical julia_fixed.csv dropped"], 1)
        self.assertEqual(self.counts["numerical mpgos.csv dropped"], 1)
        self.assertEqual(self.counts["numerical files of the wrong shape dropped"], 1)
        rows = self.rows(package="jax", n=32768)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["finals"], "finals/lorenz__tsit5__fixed__dt-0.0009765625__n32768__s3__default.parquet")
        traj, states, converged = self.store.load_finals("jax", KEY, rows[0]["finals"])
        self.assertEqual(states.shape, (32768, 3))
        self.assertEqual(list(converged[4:7]), [True, False, True])
        self.assertEqual(self.rows(package="pytorch"), [])

    def test_the_store_counts_match_and_the_old_trees_are_gone_but_the_goldens_stay(self):
        self.assertEqual(self.found, self.expected)
        self.assertEqual(sum(self.expected.values()), len(self.conversion.rows))
        for gone in ("Julia", "JAX", "CPP", "cubie_julia_overlap",
                     os.path.join("numerical_equivalence", "julia"),
                     os.path.join("numerical", KEY)):
            self.assertFalse(os.path.exists(os.path.join(self.tmp, gone)), gone)
        self.assertTrue(os.path.isfile(self.tree.golden))
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "key=" + KEY, "package=jax", "results")))
        report = self.conversion.report(self.expected)
        self.assertIn("| store rows | {0} |".format(len(self.conversion.rows)), report)
        self.assertIn("| {0} | julia_cpu | 3 |".format(KEY), report)


class DryRunTests(unittest.TestCase):
    def test_a_dry_run_writes_and_deletes_nothing(self):
        tmp = tempfile.mkdtemp(prefix="convert_legacy_dry_")
        self.addCleanup(shutil.rmtree, tmp, True)
        LegacyTree(tmp)
        conversion = convert_legacy.Conversion(tmp, dry_run=True).run()
        conversion.verify()
        conversion.delete_old()
        self.assertTrue(os.path.isdir(os.path.join(tmp, "Julia")))
        self.assertFalse(os.path.isdir(os.path.join(tmp, "key=" + KEY)))
        self.assertGreater(len(conversion.rows), 0)


if __name__ == "__main__":
    unittest.main()
