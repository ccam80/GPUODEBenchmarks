"""convert_legacy.py on a synthetic legacy tree: the spec mapping of every legacy kind, the Newton column of the catalogue, the printed-value snapping, the run_id merge, the golden finals, the controllers table, the drops, the shipped-set match, the DuckDB check, and a dry run."""

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

NAN = float("nan")
KEY = "windows_RTX-4070-SUPER"
DT = 2.0 ** -10
N_WP = 131072
N_NE = 1024
RESULTS_HEADER = ("package,key,analysis,problem,algorithm,mode,setting_kind,setting,n,states,"
                  "tier,transfers,min_ms,samples_ms,errored_pct,error,build_s,recorded_utc")


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def lorenz_spec(**overrides):
    """The spec a set expands to for lorenz on the 4070: fixed tsit5 at dt = 2^-10 unless overridden."""
    fields = dict(problem="lorenz", system_params={}, duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0,
                  n=8, grid_dtype="float32", algorithm="tsit5", controller="fixed",
                  dt=DT, dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains={},
                  newton_atol=NAN, newton_rtol=NAN, transfers="both", package="julia_gpu",
                  key=KEY)
    fields.update(overrides)
    return fields


def results_row(package, analysis, algorithm, mode, kind, setting, n, transfers, min_ms,
                samples="", error="nan", build_s="nan", stamp="2026-09-08T05:44:47Z",
                states=3, problem="lorenz"):
    return ",".join(str(v) for v in (
        package, KEY, analysis, problem, algorithm, mode, kind, setting, n, states,
        "default", transfers, min_ms, samples, "nan", error, build_s, stamp))


class LegacyTree:
    """A data/ root holding one of every legacy file kind."""

    def __init__(self, root):
        self.root = root
        self.write_results()
        self.write_overlap()
        self.write_ne_julia()
        self.write_numerical()
        write_lines(os.path.join(root, "clocks", "calibration_" + KEY + ".csv"), ["a,b"])

    def write_results(self):
        r = results_row
        julia = [RESULTS_HEADER,
                 r("julia", "times", "tsit5", "fixed", "dt", "0.0009765625", 8, "both", 1.5, "9;2;1.5"),
                 r("julia", "times", "tsit5", "fixed", "dt", "0.0009765625", 8, "none", 0.5, "3;0.5;0.6"),
                 r("julia", "times", "tsit5", "fixed", "dt", "0.0009765625", 32768, "both", 2.5, "9;2.5"),
                 r("julia", "times", "tsit5", "fixed", "dt", "0.0009765625", 32768, "none", 2.0, "9;2"),
                 r("julia", "times", "tsit5", "fixed", "dt", "0.0009765625", N_WP, "none", 4.0, "9;4"),
                 r("julia", "times", "kvaerno3", "fixed", "dt", "0.0009765625", 8, "both", 1.6, "9;1.6"),
                 r("julia", "times", "kvaerno3", "adaptive", "tol", "1e-05", 8, "both", 1.7, "9;1.7"),
                 r("julia", "wp", "tsit5", "fixed", "dt", "0.0009765625", N_WP, "d2h", 4.5, "9;4.5", error="1e-3"),
                 r("julia", "wp", "tsit5", "fixed", "dt", "0.00048828125", N_WP, "d2h", 6.5, "", error="2e-4"),
                 r("julia", "states", "tsit5", "fixed", "dt", "0.0009765625", N_WP, "both", 7.0, "9;7",
                   build_s="12.5", states=32, problem="lorenz96"),
                 r("julia", "states", "tsit5", "fixed", "dt", "0.0009765625", N_WP, "both", 9.0, "9;9",
                   build_s="14.5", states=4, problem="lorenz96"),
                 r("julia", "times", "tsit5", "fixed", "dt", "0.0009765625", N_WP, "both", 6.5, "9;6.5",
                   states=32, problem="lorenz96", stamp="2026-09-08T05:44:48Z"),
                 r("julia", "times", "tsit5", "adaptive", "tol", "1e-05", 8, "both", 1.0, "9;1",
                   states=14, problem="nand_gate")]
        write_lines(os.path.join(self.root, "Julia", KEY, "results.csv"), julia)
        jax = [RESULTS_HEADER,
               r("jax", "times", "tsit5", "fixed", "dt", "0.0009765625", 32768, "both", 3.0, "9;3"),
               r("jax", "times", "tsit5", "fixed", "dt", "0.0009765625", 32768, "none", 2.9, "9;2.9"),
               r("jax", "times", "tsit5", "fixed", "dt", "0.0009765625", N_WP, "none", 5.0, "9;5",
                 stamp="2026-09-08T05:44:40Z"),
               r("jax", "wp", "tsit5", "fixed", "dt", "0.0009765625", N_WP, "none", 5.5, "", error="3e-4",
                 stamp="2026-09-08T05:44:50Z"),
               r("jax", "wp", "tsit5", "fixed", "dt", "3.051757812e-05", N_WP, "none", 8.5, "", error="1e-6"),
               r("jax", "wp", "kvaerno3", "adaptive", "tol", "0.001", N_WP, "none", 8.0, "9;8", error="1e-2"),
               r("jax", "times", "kvaerno3", "adaptive", "tol", "1e-05", 8, "both", 8.0, "9;8"),
               r("jax", "times", "tsit5", "adaptive", "tol", "1e-05", 8, "both", 1.1, "9;1.1")]
        write_lines(os.path.join(self.root, "JAX", KEY, "results.csv"), jax)
        cpp = [RESULTS_HEADER,
               r("cpp", "times", "classical-rk4", "fixed", "dt", "0.0009765625", 8, "both", 0.1, "1;0.1")]
        write_lines(os.path.join(self.root, "CPP", KEY, "results.csv"), cpp)

    def write_overlap(self):
        base = os.path.join(self.root, "cubie_julia_overlap", KEY, "lorenz")
        header = ("framework,algorithm,phase,mode,tier,transfers,n,setting_kind,setting,samples,"
                  "min_ms,p05_ms,median_ms,p95_ms,max_ms")
        rows = [header,
                "julia,tsit5,performance,fixed,fixed,both,8,dt,0.0009765625,20,1.7,1.8,1.9,2.0,2.1",
                "julia,tsit5,performance,fixed,fixed,none,8,dt,0.0009765625,20,0.7,0.8,0.9,1.0,1.1",
                "julia,tsit5,performance,adaptive,julia,both,8,tol,1.0e-8,20,1.6,1.7,1.8,1.9,2.0",
                "julia,kvaerno3,performance,adaptive,julia,none,8,tol,1.0e-8,20,1.1,1.2,1.3,1.4,1.5",
                "julia,tsit5,work_precision,adaptive,julia,none,{0},tol,0.010000000000000002,20,1.3,1.4,1.5,1.6,1.7".format(N_WP),
                "julia,tsit5,work_precision,adaptive,julia,both,{0},tol,0.010000000000000002,20,1.4,1.4,1.5,1.6,1.7".format(N_WP),
                "julia,tsit5,numerical,adaptive,julia,none,{0},tol,0.001,1,0.3,0.4,0.5,0.6,0.7".format(N_NE)]
        write_lines(os.path.join(base, "julia_timings.csv"), rows)
        write_lines(os.path.join(base, "timing_summary.csv"), rows)
        write_lines(os.path.join(base, "work_precision.csv"), [
            "framework,algorithm,mode,tier,transfers,n,setting_kind,setting,samples,min_ms,p05_ms,"
            "median_ms,p95_ms,max_ms,golden_rmse,finite_trajectories,failed_trajectories",
            "julia,tsit5,adaptive,julia,none,{0},tol,0.010000000000000002,20,1.3,1.4,1.5,1.6,1.7,0.0049,{1},{2}".format(
                N_WP, N_WP - 8, 8)])
        write_lines(os.path.join(base, "julia_metrics.csv"), [
            "framework,algorithm,phase,mode,tier,n,setting_kind,setting,golden_rmse,"
            "finite_trajectories,failed_trajectories,finals_path",
            "julia,tsit5,performance,fixed,fixed,8,dt,0.0009765625,,8,0,",
            "julia,tsit5,performance,adaptive,julia,8,tol,1.0e-8,,6,2,",
            "julia,tsit5,work_precision,adaptive,julia,{0},tol,0.010000000000000002,0.0049,{1},8,".format(
                N_WP, N_WP - 8),
            "julia,tsit5,numerical,adaptive,julia,{0},tol,0.001,0.001,{0},0,finals/julia/tsit5/x.csv".format(N_NE)])
        write_lines(os.path.join(base, "julia_failures.csv"), [
            "framework,algorithm,phase,mode,tier,n,setting_kind,setting,error_type,message",
            "julia,vern7,performance,adaptive,julia,8,tol,1.0e-8,CuError,boom"])
        write_lines(os.path.join(base, "finals", "julia", "tsit5", "adaptive_julia_tol_0p001.csv"),
                    ["traj,s1,s2,s3", "0,1,2,3"])
        with open(os.path.join(base, "julia_metadata.json"), "w") as handle:
            json.dump({"framework": "DiffEqGPU", "diffeqgpu_version": "3.17.0"}, handle)
        write_lines(os.path.join(self.root, "cubie_julia_overlap", KEY, "speedups.csv"), ["a,b"])
        nand = os.path.join(self.root, "cubie_julia_overlap", KEY, "nand_gate")
        write_lines(os.path.join(nand, "julia_timings.csv"), [
            header, "julia,tsit5,performance,adaptive,julia,both,8,tol,1.0e-8,20,1.6,1.7,1.8,1.9,2.0"])

    def write_ne_julia(self):
        base = os.path.join(self.root, "numerical_equivalence", "julia", KEY, "lorenz")
        fixed = ["dt,traj,s1,s2,s3,converged"]
        for dt in (0.5, 0.25):
            fixed += ["{0},{1},{2},{3},{4},{5}".format(dt, traj, traj * 0.5, dt, 1.0, 1 if traj % 2 else 0)
                      for traj in range(N_NE)]
        write_lines(os.path.join(base, "kvaerno3.csv"), fixed)
        adaptive = ["tol,traj,s1,s2,s3,naccept,nreject,converged"]
        adaptive += ["0.01,{0},{1},2,3,9,0,1".format(traj, "nan" if traj < 4 else traj)
                     for traj in range(N_NE)]
        adaptive += ["0.001,{0},{1},2,3,9,0,1".format(traj, traj) for traj in range(N_NE - 1)]
        write_lines(os.path.join(base, "kvaerno3_adaptive.csv"), adaptive)
        write_lines(os.path.join(base, "tsit5_adaptive.csv"),
                    ["tol,traj,s1,s2,s3,naccept,nreject,converged"]
                    + ["1.0e-5,{0},{0},2,3,9,0,1".format(traj) for traj in range(N_NE)])
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
        golden = np.arange(N_WP * 3, dtype=np.float64).reshape(N_WP, 3) / 3.0
        np.savetxt(os.path.join(self.root, "numerical", "golden_lorenz_131072.csv"), golden,
                   delimiter=",", fmt="%.17g")
        write_lines(os.path.join(self.root, "numerical", "golden_lorenz_131072_retcodes.csv"),
                    ["row,retcode", "1,Unstable", "4177,Unstable"])
        write_lines(os.path.join(self.root, "numerical", "README.md"), ["# gone"])


class ConvertLegacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="convert_legacy_")
        cls.tree = LegacyTree(cls.tmp)
        cls.conversion = convert_legacy.Conversion(cls.tmp).run()
        cls.expected, cls.found = cls.conversion.verify()
        cls.conversion.delete_old()
        cls.store = store.Store(cls.tmp)
        cls.counts = cls.conversion.counts

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, True)

    def rows(self, **filters):
        return self.store.rows(**filters)

    def one(self, **filters):
        rows = self.rows(**filters)
        self.assertEqual(len(rows), 1, filters)
        return rows[0]

    # ------------------------------------------------------- results.csv
    def test_fixed_rows_map_to_the_fixed_controller_with_the_step_as_dt(self):
        row = self.one(package="julia_gpu", algorithm="tsit5", controller="fixed", n=8, transfers="both")
        for field, value in lorenz_spec().items():
            if isinstance(value, float) and math.isnan(value):
                self.assertTrue(math.isnan(row[field]), field)
            elif isinstance(value, dict):
                self.assertEqual(row[field], store.canonical_json(value), field)
            else:
                self.assertEqual(row[field], value, field)
        self.assertEqual(row["samples_ms"], [9.0, 2.0, 1.5])
        self.assertEqual((row["min_ms"], row["states"], row["reason"]), (1.5, 3, ""))
        self.assertNotIn("analysis", row)
        self.assertNotIn("error", row)

    def test_the_converted_run_id_is_the_hash_of_the_spec_a_set_expands_to(self):
        fixed = self.one(package="julia_gpu", algorithm="tsit5", controller="fixed", n=8, transfers="both")
        self.assertEqual(fixed["run_id"], store.run_id(lorenz_spec()))
        adaptive = self.one(package="julia_gpu", algorithm="kvaerno3", controller="default", atol=1e-5)
        self.assertEqual(adaptive["run_id"], store.run_id(lorenz_spec(
            algorithm="kvaerno3", controller="default", dt=DT, atol=1e-5, rtol=1e-5)))
        jax = self.one(package="jax", algorithm="tsit5", controller="default")
        self.assertEqual(jax["run_id"], store.run_id(lorenz_spec(
            package="jax", controller="default", dt=DT, atol=1e-5, rtol=1e-5)))
        states = self.one(package="julia_gpu", problem="lorenz96", states=4)
        self.assertEqual(states["run_id"], store.run_id(lorenz_spec(
            problem="lorenz96", system_params={"states": 4}, parameter="F", grid_max=16.0,
            n=N_WP)))

    def test_adaptive_rows_leave_dt_min_and_dt_max_to_the_package(self):
        adaptive = self.rows(controller="default")
        self.assertGreater(len(adaptive), 0)
        for row in adaptive:
            self.assertTrue(math.isnan(row["dt_min"]), row["run_id"])
            self.assertTrue(math.isnan(row["dt_max"]), row["run_id"])
            if row["precision"] == "float32":
                self.assertEqual(row["dt"], DT)
        for row in self.rows(controller="fixed"):
            self.assertTrue(math.isnan(row["dt_min"]) and math.isnan(row["dt_max"]))
            self.assertTrue(math.isnan(row["atol"]) and math.isnan(row["rtol"]))

    def test_newton_follows_the_catalogue_row_of_the_package_and_algorithm(self):
        # julia_gpu kvaerno3 has newton = false: NaN on the fixed and the adaptive row.
        fixed = self.one(package="julia_gpu", algorithm="kvaerno3", controller="fixed")
        self.assertTrue(math.isnan(fixed["newton_atol"]))
        self.assertTrue(math.isnan(fixed["newton_rtol"]))
        adaptive = self.one(package="julia_gpu", algorithm="kvaerno3", controller="default", atol=1e-5)
        self.assertTrue(math.isnan(adaptive["newton_atol"]))
        self.assertTrue(math.isnan(adaptive["newton_rtol"]))
        self.assertEqual((adaptive["atol"], adaptive["rtol"], adaptive["dt"]), (1e-5, 1e-5, DT))
        explicit = self.one(package="jax", algorithm="tsit5", controller="default")
        self.assertTrue(math.isnan(explicit["newton_atol"]))
        self.assertEqual(self.rows(package="jax", algorithm="kvaerno3"), [])

    def test_a_catalogue_row_with_newton_true_scales_the_stage_solve(self):
        tmp = tempfile.mkdtemp(prefix="convert_legacy_newton_")
        self.addCleanup(shutil.rmtree, tmp, True)
        write_lines(os.path.join(tmp, "Julia", KEY, "results.csv"), [
            RESULTS_HEADER,
            results_row("julia", "times", "kvaerno3", "fixed", "dt", "0.0009765625", 8, "both", 1.6, "9;1.6"),
            results_row("julia", "times", "kvaerno3", "adaptive", "tol", "1e-05", 8, "both", 1.7, "9;1.7"),
            results_row("julia", "times", "tsit5", "adaptive", "tol", "1e-05", 8, "both", 1.0, "9;1")])
        newton = dict(convert_legacy.load_newton())
        newton[("julia_gpu", "kvaerno3")] = True
        conversion = convert_legacy.Conversion(tmp, newton=newton).run()
        conversion.verify()
        rows = {(r["algorithm"], r["controller"]): r for r in store.Store(tmp).rows()}
        self.assertEqual((rows[("kvaerno3", "fixed")]["newton_atol"],
                          rows[("kvaerno3", "fixed")]["newton_rtol"]), (1e-6, 1e-6))
        self.assertEqual((rows[("kvaerno3", "default")]["newton_atol"],
                          rows[("kvaerno3", "default")]["newton_rtol"]), (1e-5, 1e-5))
        self.assertTrue(math.isnan(rows[("tsit5", "default")]["newton_atol"]))
        self.assertEqual(conversion.counts.get("rows of (julia_gpu, kvaerno3) without a catalogue row", 0), 0)

    def test_the_drops_are_counted_and_absent(self):
        self.assertEqual(self.counts["results.csv cpp rows dropped"], 1)
        self.assertEqual(self.counts["results.csv jax kvaerno3 rows dropped"], 2)
        self.assertEqual(self.counts["results.csv rows with transfers = d2h dropped"], 2)
        self.assertEqual(self.counts["results.csv nand_gate rows dropped"], 1)
        self.assertEqual(self.counts["overlap nand_gate rows dropped"], 1)
        self.assertEqual(self.rows(package="cpp"), [])
        self.assertEqual(self.rows(algorithm="kvaerno3", package="jax"), [])
        self.assertEqual(self.rows(problem="nand_gate"), [])
        self.assertTrue(all(r["transfers"] in ("both", "none") for r in self.rows()))
        self.assertEqual(self.rows(reason="untimed"), [])

    def test_printed_settings_snap_to_the_exact_step_and_tolerance(self):
        self.assertEqual(convert_legacy.snap_dt(3.051757812e-05, 1.0), 2.0 ** -15)
        self.assertEqual(convert_legacy.snap_dt(0.0001831054688, 3.0), 3.0 * 2.0 ** -14)
        self.assertEqual(convert_legacy.snap_dt(0.3, 1.0), 0.3)
        self.assertEqual(convert_legacy.snap_tol(0.010000000000000002), 1e-2)
        self.assertEqual(convert_legacy.snap_tol(1.0000000000000001e-7), 1e-7)
        self.assertEqual(convert_legacy.snap_tol(3e-4), 3e-4)
        row = self.one(package="jax", algorithm="tsit5", controller="fixed", n=N_WP, dt=2.0 ** -15)
        self.assertEqual(row["min_ms"], 8.5)
        wp = self.one(package="julia_gpu", n=N_WP, controller="default", transfers="none")
        self.assertEqual(wp["atol"], 1e-2)

    def test_rows_meeting_by_run_id_keep_the_sampled_row_and_fill_its_gaps(self):
        # jax times (samples, earlier) vs jax wp (no samples, later): the sampled row wins.
        row = self.one(package="jax", n=N_WP, dt=DT)
        self.assertEqual((row["min_ms"], row["samples_ms"]), (5.0, [9.0, 5.0]))
        self.assertEqual(self.counts["rows merged by run_id (results.csv times vs results.csv wp)"], 1)
        # overlap performance meets the results.csv julia times row: the sampled row stands, the version fills.
        self.assertEqual(self.counts["rows merged by run_id (overlap performance vs results.csv times)"], 2)
        both = self.one(package="julia_gpu", algorithm="tsit5", controller="fixed", n=8, transfers="both")
        self.assertEqual((both["min_ms"], both["package_version"]), (1.5, "3.17.0"))
        # the lorenz96 states row at 32 states meets the times row: build_s fills in.
        self.assertEqual(self.counts["rows merged by run_id (results.csv states vs results.csv times)"], 1)
        merged = self.one(package="julia_gpu", problem="lorenz96", states=32)
        self.assertEqual((merged["min_ms"], merged["build_s"]), (6.5, 12.5))
        self.assertEqual(merged["system_params"], '{"states":32}')

    # ----------------------------------------------------------- overlap
    def test_overlap_keeps_the_fixed_performance_rows_and_the_work_precision_rows(self):
        self.assertEqual(self.counts["overlap performance rows converted"], 2)
        self.assertEqual(self.counts["overlap performance rows of tier julia dropped"], 2)
        self.assertEqual(self.counts["overlap work_precision rows converted"], 2)
        self.assertEqual(self.counts["overlap numerical phase rows dropped"], 1)
        self.assertEqual(self.counts["overlap failure records dropped"], 1)
        self.assertEqual(self.counts["overlap finals files dropped"], 1)
        self.assertEqual(self.counts["overlap derived tables dropped"], 2)
        self.assertEqual(self.counts["overlap rows without a metrics row (errored_pct NaN)"], 0)
        self.assertEqual(self.rows(package="julia_gpu", atol=1e-8), [])
        wp = self.rows(package="julia_gpu", n=N_WP, controller="default")
        self.assertEqual(sorted((r["transfers"], r["min_ms"]) for r in wp),
                         [("both", 1.4), ("none", 1.3)])
        for row in wp:
            self.assertAlmostEqual(row["errored_pct"], 100.0 * 8 / N_WP)
            self.assertEqual((row["samples_ms"], row["package_version"], row["suite_rev"]),
                             ([], "3.17.0", ""))
            self.assertEqual((row["atol"], row["rtol"], row["dt"]), (1e-2, 1e-2, DT))
            self.assertTrue(math.isnan(row["dt_min"]))

    # ---------------------------------------------------------- NE julia
    def test_ne_trees_leave_only_the_controllers_table(self):
        self.assertEqual(self.counts["NE controller tables copied"], 1)
        self.assertEqual(self.counts["NE sweep files dropped"], 3)
        self.assertEqual(self.counts["NE sweep rows dropped"], 5)
        self.assertEqual(self.rows(package="julia_cpu", precision="float32"), [])
        self.assertEqual(self.rows(n=N_NE), [])
        controllers = os.path.join(self.tmp, "key=" + KEY, "package=julia_cpu",
                                   "controllers", "lorenz.csv")
        with open(controllers, newline="") as handle:
            self.assertEqual(next(csv.DictReader(handle))["cubie_alias"], "kvaerno3")
        finals_dir = os.path.join(self.tmp, "key=" + KEY, "package=julia_cpu", "finals")
        self.assertEqual(len(os.listdir(finals_dir)), 1)

    # ------------------------------------------------------------ golden
    def test_the_golden_is_a_float64_julia_cpu_row_with_the_sidecar_rows_unconverged(self):
        self.assertEqual(self.counts["golden rows converted"], 1)
        self.assertEqual(self.counts["golden retcode sidecar rows carried"], 2)
        row = self.one(package="julia_cpu", precision="float64")
        self.assertEqual(row["run_id"], store.run_id(lorenz_spec(
            package="julia_cpu", precision="float64", algorithm="Vern9", controller="default",
            dt=NAN, atol=1e-13, rtol=1e-13, n=N_WP, transfers="none")))
        self.assertEqual((row["reason"], row["states"], row["samples_ms"]), ("", 3, []))
        self.assertTrue(math.isnan(row["min_ms"]))
        self.assertAlmostEqual(row["errored_pct"], 200.0 / N_WP)
        self.assertTrue(math.isnan(row["newton_atol"]))
        self.assertTrue(math.isnan(row["dt_min"]) and math.isnan(row["dt_max"]))
        traj, states, t_final, retcode = self.store.load_finals("julia_cpu", KEY, row["finals"])
        self.assertEqual((states.dtype, states.shape), (np.float64, (N_WP, 3)))
        self.assertEqual(states[1, 2], 5.0 / 3.0)
        self.assertTrue(math.isnan(t_final[0]) and math.isnan(t_final[4176]))
        self.assertEqual(list(t_final[[1, 4177]]), [1.0, 1.0])
        self.assertEqual(list(retcode[[0, 1, 4176, 4177]]), ["Unstable", "", "Unstable", ""])

    def test_the_golden_takes_the_catalogue_algorithm_and_tolerance(self):
        tmp = tempfile.mkdtemp(prefix="convert_legacy_ring_")
        self.addCleanup(shutil.rmtree, tmp, True)
        golden = np.ones((N_WP, 3), dtype=np.float64)
        os.makedirs(os.path.join(tmp, "numerical"))
        np.savetxt(os.path.join(tmp, "numerical", "golden_ring_modulator_131072.csv"), golden,
                   delimiter=",")
        problems = dict(convert_legacy.load_problems())
        problems["ring_modulator"] = dict(problems["ring_modulator"], states=3)
        conversion = convert_legacy.Conversion(tmp, problems=problems).run()
        conversion.verify()
        row = store.Store(tmp).rows(package="julia_cpu")[0]
        self.assertEqual((row["algorithm"], row["atol"]), ("Rodas5P", 1e-10))
        self.assertTrue(math.isnan(row["newton_atol"]))
        self.assertEqual((row["controller"], row["grid_scale"], row["key"], row["reason"]),
                         ("default", "log", KEY, ""))
        self.assertTrue(math.isnan(row["dt"]))
        self.assertEqual(row["errored_pct"], 0.0)

    # --------------------------------------------------------- numerical
    def test_numerical_finals_files_drop_and_no_timing_row_carries_them(self):
        for name in ("jax.csv", "pytorch.csv", "myokit_cuda.csv", "julia_fixed.csv", "mpgos.csv"):
            self.assertEqual(self.counts["numerical {0} dropped".format(name)], 1, name)
        rows = self.rows(n=32768)
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row["finals"], "")
        self.assertFalse(os.path.isdir(os.path.join(self.tmp, "key=" + KEY, "package=jax", "finals")))
        self.assertEqual(self.rows(package="pytorch"), [])

    # -------------------------------------------------------------- sets
    def test_the_rows_a_shipped_set_produces_hash_to_its_run_id_and_the_rest_are_counted(self):
        produced, unproduced = self.conversion.produced, self.conversion.unproduced
        self.assertEqual(len(self.conversion.rows), 17)
        # perf: the dt 2^-10 and tol 1e-5 rows on a perf n at the default state count.
        self.assertEqual(produced["perf"], 12)
        self.assertEqual(produced["states"], 2)
        self.assertEqual(produced["golden"], 1)
        # golden_grid: the n = 131072, transfers = none rows at a listed step or tolerance.
        self.assertEqual(produced["golden_grid"], 3)
        # The overlap wp row with transfers = both and the jax step 2^-15 belong to no set.
        self.assertEqual(dict(unproduced), {("julia_gpu", "default"): 1, ("jax", "fixed"): 1})
        report = self.conversion.report(self.expected)
        self.assertIn("| perf | 12 |", report)
        self.assertIn("| any set | 15 |", report)
        self.assertIn("| jax | fixed | 1 |", report)
        self.assertIn("| all | all | 2 |", report)

    # ------------------------------------------------------------- store
    def test_the_store_counts_match_and_the_legacy_trees_are_gone(self):
        self.assertEqual(self.found, self.expected)
        self.assertEqual(sum(self.expected.values()), len(self.conversion.rows))
        for gone in ("Julia", "JAX", "CPP", "cubie_julia_overlap", "numerical_equivalence",
                     "numerical"):
            self.assertFalse(os.path.exists(os.path.join(self.tmp, gone)), gone)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "clocks")))
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "key=" + KEY, "package=jax", "results")))
        for row in self.rows():
            self.assertEqual(row["run_id"], store.run_id(row))
            self.assertEqual(row["trial_id"], store.trial_id(row))
        report = self.conversion.report(self.expected)
        self.assertIn("| store rows | {0} |".format(len(self.conversion.rows)), report)
        self.assertIn("| {0} | julia_cpu | 1 |".format(KEY), report)


class DryRunTests(unittest.TestCase):
    def test_a_dry_run_writes_and_deletes_nothing(self):
        tmp = tempfile.mkdtemp(prefix="convert_legacy_dry_")
        self.addCleanup(shutil.rmtree, tmp, True)
        LegacyTree(tmp)
        conversion = convert_legacy.Conversion(tmp, dry_run=True).run()
        conversion.verify()
        conversion.delete_old()
        self.assertTrue(os.path.isdir(os.path.join(tmp, "Julia")))
        self.assertTrue(os.path.isfile(os.path.join(tmp, "numerical", "golden_lorenz_131072.csv")))
        self.assertFalse(os.path.isdir(os.path.join(tmp, "key=" + KEY)))
        self.assertGreater(len(conversion.rows), 0)
        self.assertEqual(conversion.finals_written, 1)
        self.assertEqual(conversion.produced["golden"], 1)


if __name__ == "__main__":
    unittest.main()
