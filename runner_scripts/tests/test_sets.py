"""Set expansion: the shipped sets' counts against the catalogues, the julia_cpu prefix grid, matched and pi controller resolution, the trial merge, the narrowing flags, and the schema checks."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

import grid  # noqa: E402
import sets  # noqa: E402
import store  # noqa: E402
import trials  # noqa: E402
from algorithms import load_algorithms  # noqa: E402
from problems import load_problems  # noqa: E402
from protocol import OPTIMIZE_N, OPTIMIZE_PER_POINT_FAMILIES  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
PERF_N = [8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 16777216]
TOLS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

ALGORITHMS = {row.name: row for row in load_algorithms()}
PROBLEMS = {row.name: row for row in load_problems()}


def by_package_kind(specs_or_trials, field="package"):
    return Counter(item[field] for item in specs_or_trials)


def solve_counts(trial_list):
    return Counter(t["package"] for t in trial_list if t["kind"] == "solve")


def kind_counts(trial_list, package):
    return Counter(t["kind"] for t in trial_list if t["package"] == package)


def capable(package, kind):
    return [row for row in ALGORITHMS.values() if row.supports(package, kind)]


def pi_algorithms(package):
    """Adaptive algorithms whose DIRK PI defaults differ from cubie's shipped controller."""
    import cubie_adapter
    out = []
    for row in capable(package, "adaptive"):
        shipped = cubie_adapter.default_controller(row.name, row["family"], row["order"])
        if not cubie_adapter.controllers_equal(cubie_adapter.pi_tier_controller(row["order"]), shipped):
            out.append(row)
    return out


def problems_of(package):
    return [row for row in PROBLEMS.values() if row.supports(package)]


class ShippedSetTests(unittest.TestCase):
    """Every shipped set's counts, derived from the catalogues and pinned as literals."""

    @classmethod
    def setUpClass(cls):
        cls.expanded = {name: sets.expand([name], KEY, root=os.path.join(tempfile.gettempdir(), "no-data"))
                        for name in sets.set_names()}
        cls.trials = {name: trials.build_trials(specs) for name, specs in cls.expanded.items()}

    def test_the_four_sets_ship(self):
        self.assertEqual(sets.set_names(), ["golden", "golden_grid", "perf", "states"])

    def test_perf_counts(self):
        specs = self.expanded["perf"]
        self.assertEqual(sorted({s["n"] for s in specs}), PERF_N)
        self.assertEqual({s["axis"] for s in specs}, {"n"})
        self.assertEqual({tuple(s["transfers"]) for s in specs}, {("both", "none")})
        self.assertEqual({s["finals"] for s in specs}, {False})
        self.assertNotIn("julia_cpu", by_package_kind(specs))
        expected = {}
        for package in ("cubie", "cubie_mlir", "jax", "pytorch", "myokit_cuda", "cpp", "julia_gpu"):
            legs = len(capable(package, "fixed")) + len(capable(package, "adaptive"))
            if package in ("cubie", "cubie_mlir"):
                legs += len(pi_algorithms(package))
            expected[package] = legs * len(problems_of(package)) * len(PERF_N)
        self.assertEqual(dict(by_package_kind(specs)), expected)
        self.assertEqual(expected, {"cubie": 4608, "cubie_mlir": 4608, "jax": 360, "pytorch": 180,
                                    "myokit_cuda": 36, "cpp": 120, "julia_gpu": 960})
        built = self.trials["perf"]
        self.assertEqual(dict(solve_counts(built)), expected)
        self.assertEqual(kind_counts(built, "cubie"), {"solve": 4608, "warm": 384, "optimize": 384})
        self.assertEqual(kind_counts(built, "jax"), {"solve": 360, "warm": 30})
        self.assertEqual(kind_counts(built, "julia_gpu"), {"solve": 960, "warm": 80})

    def test_perf_stepping_values(self):
        specs = self.expanded["perf"]
        lorenz = [s for s in specs if s["problem"] == "lorenz" and s["package"] == "cubie" and s["n"] == 8]
        fixed = {s["algorithm"]: s for s in lorenz if s["controller"] == "fixed"}
        self.assertEqual(fixed["kvaerno3"]["dt"], 2.0 ** -10)
        self.assertEqual((fixed["kvaerno3"]["newton_atol"], fixed["kvaerno3"]["newton_rtol"]), (1e-6, 1e-6))
        self.assertTrue(np.isnan(fixed["tsit5"]["newton_atol"]))
        for field in ("dt_min", "dt_max", "atol", "rtol"):
            self.assertTrue(np.isnan(fixed["kvaerno3"][field]), field)
        self.assertEqual(fixed["kvaerno3"]["gains"], "{}")
        default = {s["algorithm"]: s for s in lorenz if s["controller"] == "default"}
        self.assertEqual((default["kvaerno3"]["atol"], default["kvaerno3"]["rtol"]), (1e-5, 1e-5))
        self.assertEqual(default["kvaerno3"]["dt"], 2.0 ** -10)
        self.assertEqual(default["kvaerno3"]["dt_min"], 1e-6)
        self.assertTrue(np.isnan(default["kvaerno3"]["dt_max"]))
        self.assertEqual(default["kvaerno3"]["newton_atol"], 1e-5)
        self.assertTrue(np.isnan(default["tsit5"]["newton_atol"]))
        pollu = [s for s in specs if s["problem"] == "pollu" and s["package"] == "cubie"
                 and s["controller"] == "default" and s["algorithm"] == "kvaerno3"][0]
        self.assertEqual(pollu["dt"], 60.0 * 2.0 ** -10)
        self.assertEqual(pollu["dt_min"], 60.0 * 1e-6)
        self.assertEqual((pollu["parameter"], pollu["grid_scale"], pollu["grid_min"], pollu["grid_max"]),
                         ("k1", "log", 3.5e-2, 3.5))
        self.assertEqual(pollu["system_params"], "{}")
        lorenz96 = [s for s in specs if s["problem"] == "lorenz96"][0]
        self.assertEqual(lorenz96["system_params"], '{"states":32}')
        # A package without the Newton setting carries NaN on its implicit rows.
        julia = [s for s in specs if s["package"] == "julia_gpu" and s["algorithm"] == "kvaerno3"
                 and s["problem"] == "lorenz" and s["n"] == 8]
        self.assertEqual(len(julia), 2)
        self.assertTrue(all(np.isnan(s["newton_atol"]) for s in julia))
        # jax has it.
        jax = [s for s in specs if s["package"] == "jax" and s["algorithm"] == "kvaerno3"
               and s["problem"] == "lorenz" and s["n"] == 8 and s["controller"] == "fixed"][0]
        self.assertEqual(jax["newton_atol"], 1e-6)

    def test_perf_pi_stepping_skips_the_shipped_dirk_controller(self):
        specs = [s for s in self.expanded["perf"] if s["stepping"] == "pi" and s["problem"] == "lorenz"
                 and s["package"] == "cubie" and s["n"] == 8]
        names = sorted(s["algorithm"] for s in specs)
        self.assertEqual(names, sorted(r.name for r in pi_algorithms("cubie")))
        for absent in ("kvaerno3", "kvaerno5", "l_stable_sdirk_4"):
            self.assertNotIn(absent, names)
        self.assertIn("tsit5", names)
        tsit5 = [s for s in specs if s["algorithm"] == "tsit5"][0]
        self.assertEqual(tsit5["controller"], "pi")
        gains = json.loads(tsit5["gains"])
        self.assertEqual(set(gains), {"integral_gain", "proportional_gain", "safety",
                                      "min_step_shrink", "max_step_growth"})
        self.assertEqual(gains["proportional_gain"], 0.4 * 6 / 5)
        self.assertEqual(tsit5["newton_atol"] == tsit5["newton_atol"], False)
        self.assertNotIn("julia_gpu", {s["package"] for s in self.expanded["perf"] if s["stepping"] == "pi"})

    def test_states_counts(self):
        specs = self.expanded["states"]
        self.assertEqual({s["problem"] for s in specs}, {"lorenz96"})
        self.assertEqual({s["n"] for s in specs}, {131072})
        self.assertEqual({s["axis"] for s in specs}, {"states"})
        self.assertEqual(sorted({json.loads(s["system_params"])["states"] for s in specs}),
                         [4, 8, 16, 32, 64, 128])
        expected = {}
        for package in store.PACKAGES:
            legs = len(capable(package, "fixed")) + len(capable(package, "adaptive"))
            if package in ("cubie", "cubie_mlir"):
                legs += len(pi_algorithms(package))
            expected[package] = legs * 6
        self.assertEqual(dict(by_package_kind(specs)), expected)
        self.assertEqual(expected["cubie"], 288)
        self.assertEqual(expected["julia_cpu"], 186)
        built = self.trials["states"]
        # Cold builds: no warm or optimize trials, one solve per leg.
        self.assertEqual({t["kind"] for t in built}, {"solve"})
        self.assertEqual(len({t["leg"] for t in built if t["package"] == "cubie"}), 288)
        self.assertEqual({t["ordinal"] for t in built}, {0})

    def test_golden_grid_counts(self):
        specs = self.expanded["golden_grid"]
        self.assertEqual({tuple(s["transfers"]) for s in specs}, {("none",)})
        self.assertEqual({s["finals"] for s in specs}, {True})
        self.assertEqual({s["axis"] for s in specs}, {"dt", "tol"})
        self.assertEqual({s["stepping"] for s in specs}, {"fixed", "default", "pi"})
        expected = {}
        for package in store.PACKAGES:
            fixed = capable(package, "fixed")
            dts = sum(10 if row.name == "euler" else 13 for row in fixed)
            tols = len(capable(package, "adaptive")) * len(TOLS)
            if package in ("cubie", "cubie_mlir"):
                tols += len(pi_algorithms(package)) * len(TOLS)
            expected[package] = (dts + tols) * len(problems_of(package))
        self.assertEqual(dict(by_package_kind(specs)), expected)
        self.assertEqual(expected, {"cubie": 3480, "cubie_mlir": 3480, "jax": 315, "pytorch": 180,
                                    "myokit_cuda": 30, "cpp": 100, "julia_gpu": 800, "julia_cpu": 2408})
        euler = sorted({s["dt"] for s in specs if s["algorithm"] == "euler" and s["problem"] == "lorenz"})
        self.assertEqual(euler, [2.0 ** -k for k in range(17, 7, -1)])
        tsit5 = sorted({s["dt"] for s in specs if s["algorithm"] == "tsit5" and s["problem"] == "lorenz"
                        and s["controller"] == "fixed"})
        self.assertEqual(tsit5, [2.0 ** -k for k in range(13, 0, -1)])
        self.assertEqual(sorted({s["atol"] for s in specs if s["controller"] != "fixed"}), sorted(TOLS))
        built = self.trials["golden_grid"]
        self.assertEqual(kind_counts(built, "cubie"), {"solve": 3480, "warm": 384, "optimize": 2544})
        self.assertEqual(kind_counts(built, "julia_cpu"), {"solve": 2408, "warm": 248})

    def test_golden_grid_optimize_trials_follow_the_families(self):
        built = [t for t in self.trials["golden_grid"] if t["package"] == "cubie" and t["problem"] == "lorenz"]
        optimize = [t for t in built if t["kind"] == "optimize"]
        self.assertEqual({t["n"] for t in optimize}, {OPTIMIZE_N})
        per_leg = Counter(t["leg"] for t in optimize)
        for t in built:
            family = ALGORITHMS[t["algorithm"]]["family"]
            solves = sum(1 for s in built if s["kind"] == "solve" and s["leg"] == t["leg"])
            self.assertEqual(per_leg[t["leg"]], solves if family in OPTIMIZE_PER_POINT_FAMILIES else 1, t["leg"])

    def test_golden_counts_and_values(self):
        specs = self.expanded["golden"]
        self.assertEqual(len(specs), 8)
        self.assertEqual({s["package"] for s in specs}, {"julia_cpu"})
        for spec in specs:
            problem = PROBLEMS[spec["problem"]]
            self.assertEqual(spec["algorithm"], problem["golden_algorithm"])
            self.assertEqual((spec["atol"], spec["rtol"]), (problem["golden_tol"], problem["golden_tol"]))
            self.assertEqual((spec["precision"], spec["n"], spec["controller"], spec["gains"]),
                             ("float64", 131072, "default", "{}"))
            for field in ("dt", "dt_min", "dt_max", "newton_atol", "newton_rtol"):
                self.assertTrue(np.isnan(spec[field]), field)
            self.assertEqual((spec["grid_min"], spec["grid_max"]), (problem["sweep_min"], problem["sweep_max"]))
            self.assertTrue(spec["finals"])
            self.assertEqual(spec["transfers"], ["none"])
        built = self.trials["golden"]
        self.assertEqual(kind_counts(built, "julia_cpu"), {"solve": 8, "warm": 8})

    def test_julia_cpu_golden_grid_is_the_1024_prefix_of_the_131072_grid(self):
        julia = [s for s in self.expanded["golden_grid"] if s["package"] == "julia_cpu"]
        self.assertEqual({s["n"] for s in julia}, {1024})
        others = [s for s in self.expanded["golden_grid"] if s["package"] != "julia_cpu"]
        self.assertEqual({s["n"] for s in others}, {131072})
        stepping = ("algorithm", "controller", "dt", "atol", "newton_atol")

        def same(a, b):
            return all(a[f] == b[f] or (a[f] != a[f] and b[f] != b[f]) for f in stepping)

        for problem in PROBLEMS.values():
            short = [s for s in julia if s["problem"] == problem.name][0]
            full = [s for s in others if s["problem"] == problem.name and same(s, short)][0]
            np.testing.assert_array_equal(grid.grid(short), grid.grid(full)[:1024], problem.name)
            self.assertEqual(short["grid_max"], grid.grid_point(problem["sweep_scale"], problem["sweep_min"],
                                                                problem["sweep_max"], 131072, 1023))
            self.assertEqual(store.group_id(dict(short, transfers="none", key=KEY)),
                             store.group_id(dict(full, transfers="none", key=KEY)))

    def test_no_spec_field_outside_the_run_spec(self):
        for name, specs in self.expanded.items():
            for spec in specs:
                self.assertEqual(set(spec), set(sets.SPEC_KEYS) | set(sets.EXTRA_KEYS), name)
                store.spec_of(dict(spec, transfers=spec["transfers"][0], key=KEY))

    def test_set_module_cli_prints_counts(self):
        out = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "sets.py"), "golden", KEY],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "julia_cpu 8")


class FakeControllers:
    """cubie_adapter's controller functions without cubie: shipped i for erk, pi for dirk, gustafsson for firk."""

    @staticmethod
    def default_controller(alias, family, order):
        if family == "erk":
            return {"step_controller": "i", "integral_gain": 1.2, "safety": 0.9,
                    "min_step_shrink": 0.2, "max_step_growth": 10.0}
        if family == "dirk":
            return FakeControllers.pi_tier_controller(order)
        if family in ("firk", "rosenbrock", "implicit"):
            return {"step_controller": "gustafsson", "safety": 0.9, "min_step_shrink": 0.2,
                    "max_step_growth": 8.0}
        return None

    @staticmethod
    def pi_tier_controller(order):
        return {"step_controller": "pi", "integral_gain": 0.3 * 4 / order,
                "proportional_gain": 0.4 * 4 / order, "safety": 0.9,
                "min_step_shrink": 0.2, "max_step_growth": 10.0}


class ControllerResolutionTests(unittest.TestCase):
    """matched and pi steppings through cubie_adapter's mapping functions, with the shipped tables faked."""

    def setUp(self):
        import cubie_adapter
        self.tmp = tempfile.mkdtemp(prefix="sets_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.sets_dir = os.path.join(self.tmp, "sets")
        os.makedirs(self.sets_dir)
        self.root = os.path.join(self.tmp, "data")
        for name in ("default_controller", "pi_tier_controller"):
            original = getattr(cubie_adapter, name)
            setattr(cubie_adapter, name, getattr(FakeControllers, name))
            self.addCleanup(setattr, cubie_adapter, name, original)

    def write_set(self, name, text):
        with open(os.path.join(self.sets_dir, name + ".toml"), "w", encoding="utf-8") as handle:
            handle.write(text)

    def write_controllers(self, problem, rows):
        directory = os.path.join(self.root, "key=" + KEY, "package=julia_cpu", "controllers")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, problem + ".csv"), "w", encoding="utf-8", newline="") as handle:
            handle.write("cubie_alias,controller,beta1,beta2,qmin,qmax,gamma,order\n")
            for row in rows:
                handle.write(",".join(row) + "\n")

    ADAPTIVE = '''
[set]
packages = ["cubie"]
problems = ["lorenz"]
transfers = ["none"]
[[grid]]
n = [8]
[[stepping]]
controller = "{controller}"
tol = [1.0e-5]
dt0 = {{duration_times_2_pow = -10}}
dt_min = {{duration_times = 1.0e-6}}
newton = "tol"
{gains}
'''

    def test_matched_resolves_julias_row_and_skips_absent_and_shipped_ones(self):
        self.write_set("m", self.ADAPTIVE.format(controller="matched", gains=""))
        self.assertEqual(sets.expand(["m"], KEY, self.root, sets_dir=self.sets_dir), [])
        self.write_controllers("lorenz", [
            ("tsit5", "PIController", "0.28", "0.04", "0.2", "10.0", "0.9", "5"),
            ("radau_iia_5", "PredictiveController", "", "", "0.2", "8.0", "0.9", "5"),
            ("kvaerno3", "PIController", "0.1", "0.1", "0.2", "10.0", "0.9", "3"),
            ("ros3p", "Other", "", "", "", "", "", "3"),
        ])
        specs = {s["algorithm"]: s for s in sets.expand(["m"], KEY, self.root, sets_dir=self.sets_dir)}
        self.assertEqual(sorted(specs), ["kvaerno3", "radau_iia_5", "tsit5"])
        tsit5 = specs["tsit5"]
        self.assertEqual(tsit5["controller"], "pi")
        self.assertEqual(json.loads(tsit5["gains"]), {
            "integral_gain": 0.28 * 6 - 0.04 * 6, "proportional_gain": 0.04 * 6, "safety": 0.9,
            "min_step_shrink": 0.2, "max_step_growth": 10.0})
        self.assertEqual(tsit5["stepping"], "matched")
        radau = specs["radau_iia_5"]
        self.assertEqual(radau["controller"], "gustafsson")
        self.assertEqual(json.loads(radau["gains"]), {"safety": 0.9})
        self.assertEqual(json.loads(specs["kvaerno3"]["gains"])["proportional_gain"], 0.1 * 4)
        # Julia's row equal to cubie's shipped controller is skipped.
        # beta1 (order + 1) = I + P and beta2 (order + 1) = P of the shipped pi at order 3.
        self.write_controllers("lorenz", [
            ("kvaerno3", "PIController", "0.2333333333333333", "0.1333333333333333", "0.2", "10.0", "0.9", "3")])
        self.assertEqual(sets.expand(["m"], KEY, self.root, sets_dir=self.sets_dir), [])

    def test_pi_with_dirk_defaults_skips_dirk_algorithms(self):
        self.write_set("p", self.ADAPTIVE.format(controller="pi", gains='gains = "dirk_defaults"'))
        specs = sets.expand(["p"], KEY, self.root, sets_dir=self.sets_dir)
        names = {s["algorithm"] for s in specs}
        self.assertEqual(names, {r.name for r in capable("cubie", "adaptive") if r["family"] != "dirk"})
        for spec in specs:
            self.assertEqual(spec["controller"], "pi")
            gains = json.loads(spec["gains"])
            self.assertEqual(gains["integral_gain"], 0.3 * 4 / ALGORITHMS[spec["algorithm"]]["order"])
            self.assertEqual(gains["proportional_gain"], 0.4 * 4 / ALGORITHMS[spec["algorithm"]]["order"])
            self.assertEqual(spec["stepping"], "pi")

    def test_explicit_gains_and_a_named_controller_pass_through(self):
        self.write_set("g", self.ADAPTIVE.format(controller="pid", gains="gains = {kp = 0.7, ki = 0.4}"))
        specs = sets.expand(["g"], KEY, self.root, algorithms=["tsit5"], sets_dir=self.sets_dir)
        self.assertEqual(len(specs), 1)
        self.assertEqual((specs[0]["controller"], specs[0]["gains"]), ("pid", '{"ki":0.4,"kp":0.7}'))

    def test_matched_and_dirk_defaults_are_cubie_only(self):
        text = self.ADAPTIVE.format(controller="matched", gains="").replace('["cubie"]', '["jax"]')
        self.write_set("bad", text)
        self.write_controllers("lorenz", [("tsit5", "PIController", "0.28", "0.04", "0.2", "10.0", "0.9", "5")])
        with self.assertRaises(sets.SetError):
            sets.expand(["bad"], KEY, self.root, sets_dir=self.sets_dir)


class MergeAndNarrowTests(unittest.TestCase):
    def setUp(self):
        import cubie_adapter
        for name in ("default_controller", "pi_tier_controller"):
            original = getattr(cubie_adapter, name)
            setattr(cubie_adapter, name, getattr(FakeControllers, name))
            self.addCleanup(setattr, cubie_adapter, name, original)
        self.root = os.path.join(tempfile.gettempdir(), "no-data")

    def test_a_trial_shared_by_two_sets_merges_transfers_and_finals(self):
        specs = sets.expand(["perf", "golden_grid"], KEY, self.root, packages=["jax"], problems=["lorenz"])
        built = trials.build_trials(specs)
        shared = [t for t in built if t["kind"] == "solve" and t["n"] == 131072 and t["controller"] == "fixed"
                  and t["algorithm"] == "tsit5" and t["dt"] == 2.0 ** -10]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]["transfers"], ["both", "none"])
        self.assertTrue(shared[0]["finals"])
        # The first set's leg keeps the trial; the perf leg is on the n axis.
        self.assertEqual(shared[0]["leg"], "lorenz/{}/tsit5/fixed/float32/n")
        self.assertEqual(shared[0]["ordinal"], PERF_N.index(131072))
        dt_leg = [t for t in built if t["leg"] == "lorenz/{}/tsit5/fixed/float32/dt" and t["kind"] == "solve"]
        self.assertEqual(len(dt_leg), 12)
        self.assertNotIn(2.0 ** -10, [t["dt"] for t in dt_leg])
        ids = [t["trial_id"] for t in built if t["kind"] == "solve"]
        self.assertEqual(len(ids), len(set(ids)))
        # Reversed order: the golden_grid leg keeps it.
        reverse = trials.build_trials(sets.expand(["golden_grid", "perf"], KEY, self.root,
                                                  packages=["jax"], problems=["lorenz"]))
        shared = [t for t in reverse if t["trial_id"] == shared[0]["trial_id"] and t["kind"] == "solve"][0]
        self.assertEqual(shared["leg"], "lorenz/{}/tsit5/fixed/float32/dt")
        self.assertEqual(shared["transfers"], ["both", "none"])

    def test_ordinals_follow_the_cost_order(self):
        built = trials.build_trials(sets.expand(["golden_grid"], KEY, self.root, packages=["jax"],
                                                problems=["lorenz"], algorithms=["kvaerno3"]))
        fixed = sorted((t["ordinal"], t["dt"]) for t in built if t["kind"] == "solve" and t["controller"] == "fixed")
        self.assertEqual([o for o, _ in fixed], list(range(13)))
        self.assertEqual([dt for _, dt in fixed], [2.0 ** -k for k in range(1, 14)])
        adaptive = sorted((t["ordinal"], t["atol"]) for t in built if t["kind"] == "solve" and t["controller"] != "fixed")
        self.assertEqual([tol for _, tol in adaptive], TOLS)
        warm = [t for t in built if t["kind"] == "warm"]
        self.assertEqual(len(warm), 2)
        self.assertEqual({(t["ordinal"], t["dt"]) for t in warm if t["controller"] == "fixed"}, {(0, 0.5)})
        perf = trials.build_trials(sets.expand(["perf"], KEY, self.root, packages=["jax"], problems=["lorenz"],
                                               algorithms=["tsit5"]))
        self.assertEqual([t["n"] for t in perf if t["kind"] == "solve" and t["controller"] == "fixed"], PERF_N)
        self.assertEqual({t["axis"] for t in perf}, {"n"})

    def test_narrowing_by_package_problem_algorithm_and_n(self):
        specs = sets.expand(["perf"], KEY, self.root, packages=["pytorch"], problems=["pollu", "lorenz"],
                            algorithms=["euler", "tsit5"], n=[8, 32])
        self.assertEqual({s["package"] for s in specs}, {"pytorch"})
        self.assertEqual({s["problem"] for s in specs}, {"lorenz", "pollu"})
        self.assertEqual({s["algorithm"] for s in specs}, {"euler", "tsit5"})
        self.assertEqual(sorted({s["n"] for s in specs}), [8, 32])
        self.assertEqual(len(specs), 2 * 2 * 2)
        # -n replaces the n list of every set's grids.
        golden = sets.expand(["golden_grid"], KEY, self.root, packages=["jax"], problems=["lorenz"],
                             algorithms=["euler"], n=[16])
        self.assertEqual({s["n"] for s in golden}, {16})
        self.assertEqual({s["axis"] for s in golden}, {"dt"})

    def test_narrowing_by_mode_controller_tol_and_dt(self):
        specs = sets.expand(["golden_grid"], KEY, self.root, packages=["cubie"], problems=["lorenz"],
                            algorithms=["tsit5"])
        self.assertEqual({s["stepping"] for s in specs}, {"fixed", "default", "pi"})
        fixed = sets.narrow(specs, mode="fixed")
        self.assertEqual({s["controller"] for s in fixed}, {"fixed"})
        self.assertEqual(len(fixed), 13)
        adaptive = sets.narrow(specs, mode="adaptive")
        self.assertEqual(len(adaptive), 14)
        self.assertEqual({s["controller"] for s in sets.narrow(specs, controllers=["pi"])}, {"pi"})
        self.assertEqual(len(sets.narrow(specs, controllers=["default", "fixed"])), 20)
        tol = sets.narrow(specs, tols=[1e-5])
        self.assertEqual([(s["controller"], s["atol"]) for s in tol], [("default", 1e-5), ("pi", 1e-5)])
        dt = sets.narrow(specs, dts=[2.0 ** -10, 0.5])
        self.assertEqual(sorted(s["dt"] for s in dt), [2.0 ** -10, 0.5])
        both = sets.narrow(specs, tols=[1e-3], dts=[0.25])
        self.assertEqual(sorted((s["controller"], s["dt"] if s["controller"] == "fixed" else s["atol"])
                                for s in both), [("default", 1e-3), ("fixed", 0.25), ("pi", 1e-3)])
        self.assertEqual(sets.narrow(specs, mode="fixed", tols=[1e-3]), [])
        matched = sets.narrow(specs, controllers=["matched"])
        self.assertEqual(matched, [])


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sets_schema_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, text, name="s"):
        path = os.path.join(self.tmp, name + ".toml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    MINIMAL = '[set]\npackages = ["jax"]\nproblems = ["lorenz"]\n[[grid]]\nn = [8]\n[[stepping]]\ncontroller = "fixed"\ndt = [0.5]\n'

    def test_defaults_fill_the_optional_keys(self):
        loaded = sets.load_set("s", self.tmp) if self.write(self.MINIMAL) else None
        head = loaded["set"]
        self.assertEqual((head["algorithms"], head["precision"], head["finals"], head["transfers"], head["build"]),
                         ("all", "float32", False, ["both", "none"], "warm"))
        self.assertEqual(loaded["grid"][0]["parameter"], "default")
        self.assertEqual(loaded["stepping"][0]["newton"], "none")
        specs = sets.expand(["s"], KEY, sets_dir=self.tmp)
        self.assertEqual(len(specs), len(capable("jax", "fixed")))
        self.assertEqual({s["dt"] for s in specs}, {0.5})
        self.assertTrue(all(np.isnan(s["newton_atol"]) for s in specs))

    def test_bad_sets_are_refused(self):
        bad = {
            "unknown key": self.MINIMAL.replace("[[grid]]", "[[grid]]\nshape = 1"),
            "unknown package": self.MINIMAL.replace('["jax"]', '["fortran"]'),
            "unknown problem": self.MINIMAL.replace('["lorenz"]', '["lorenz1000"]'),
            "bad precision": self.MINIMAL.replace("[set]", "[set]\nprecision = \"float16\""),
            "bad transfers": self.MINIMAL.replace("[set]", "[set]\ntransfers = [\"d2h\"]"),
            "no stepping": self.MINIMAL.split("[[stepping]]")[0],
            "fixed with tol": self.MINIMAL + "tol = [1e-5]\n",
            "adaptive without tol": self.MINIMAL.replace('controller = "fixed"\ndt = [0.5]', 'controller = "default"'),
            "adaptive with dt": self.MINIMAL.replace('controller = "fixed"', 'controller = "default"\ntol = [1e-5]'),
            "bad n": self.MINIMAL.replace("n = [8]", "n = [1]"),
            "states on a fixed-size problem": self.MINIMAL.replace("n = [8]", "n = [8]\nsystem_params = {states = [4]}"),
            "bad newton": self.MINIMAL + 'newton = "tight"\n',
            "bad pin": self.MINIMAL.replace('dt = [0.5]', 'dt = {half_life = 2}'),
        }
        for label, text in bad.items():
            self.write(text)
            with self.assertRaises(sets.SetError, msg=label):
                sets.expand(["s"], KEY, sets_dir=self.tmp)
        with self.assertRaises(sets.SetError):
            sets.load_set("nosuchset", self.tmp)

    def test_grid_overrides_and_pins_resolve(self):
        text = ('[set]\npackages = ["cpp"]\nproblems = ["pollu", "lorenz"]\nalgorithms = ["cash-karp-54"]\n'
                '[[grid]]\nn = [8]\nmax = 2.0\n[grid.problems]\npollu = {min = 0.1, scale = "linear"}\n'
                '[[stepping]]\ncontroller = "default"\ntol = [1e-3, 1e-4]\ndt0 = {duration_times_2_pow = -2}\n'
                'dt_min = 0.25\ndt_max = {duration_times = 0.5}\n')
        self.write(text)
        specs = {s["problem"]: s for s in sets.expand(["s"], KEY, sets_dir=self.tmp) if s["atol"] == 1e-3}
        self.assertEqual((specs["lorenz"]["grid_min"], specs["lorenz"]["grid_max"], specs["lorenz"]["grid_scale"]),
                         (0.0, 2.0, "linear"))
        self.assertEqual((specs["pollu"]["grid_min"], specs["pollu"]["grid_max"], specs["pollu"]["grid_scale"]),
                         (0.1, 2.0, "linear"))
        self.assertEqual((specs["pollu"]["dt"], specs["pollu"]["dt_min"], specs["pollu"]["dt_max"]),
                         (15.0, 0.25, 30.0))
        self.assertEqual({s["axis"] for s in sets.expand(["s"], KEY, sets_dir=self.tmp)}, {"tol"})


class TrialFileTests(unittest.TestCase):
    def test_jsonl_round_trip_writes_null_for_nan_and_only_the_trial_fields(self):
        import cubie_adapter
        saved = (cubie_adapter.default_controller, cubie_adapter.pi_tier_controller)
        cubie_adapter.default_controller = FakeControllers.default_controller
        cubie_adapter.pi_tier_controller = FakeControllers.pi_tier_controller
        try:
            built = trials.build_trials(sets.expand(["perf"], KEY, packages=["cpp"], problems=["lorenz"], n=[8]))
        finally:
            cubie_adapter.default_controller, cubie_adapter.pi_tier_controller = saved
        tmp = tempfile.mkdtemp(prefix="trials_")
        self.addCleanup(shutil.rmtree, tmp, True)
        path = trials.write_jsonl(os.path.join(tmp, "cpp.jsonl"), built)
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(len(lines), len(built))
        first = json.loads(lines[0])
        self.assertEqual(list(first), list(trials.TRIAL_KEYS))
        self.assertNotIn("nan", lines[0].lower())
        self.assertIsNone(first["atol"])
        back = trials.read_jsonl(path)
        self.assertEqual(len(back), len(built))
        self.assertTrue(np.isnan(back[0]["atol"]))
        self.assertEqual(back[0]["trial_id"], store.trial_id(back[0]))
        kinds, legs = trials.counts(back)
        self.assertEqual(kinds, {"solve": 2, "warm": 2, "optimize": 0})
        self.assertEqual(legs, {"lorenz/{}/classical-rk4/fixed/float32/n": 1,
                                "lorenz/{}/cash-karp-54/default/float32/n": 1})


if __name__ == "__main__":
    unittest.main()
