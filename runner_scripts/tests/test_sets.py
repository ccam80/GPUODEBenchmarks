"""Set expansion: the shipped sets' counts against the catalogues, the julia_cpu prefix grid, matched and pi controller resolution, the canonical trial merge and file order, the narrowing flags, and the schema checks."""

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
import protocol  # noqa: E402
import sets  # noqa: E402
import store  # noqa: E402
import trials  # noqa: E402
from algorithms import algorithm_facts, load_algorithms  # noqa: E402
from problems import load_problems  # noqa: E402
KEY = "windows_RTX-4070-SUPER"
DATA = os.path.join(ROOT, "data")
NAN = float("nan")
OPTIMIZE_N = 262144
PERF_N = [8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 16777216]
TOLS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

# One catalogue row per (package, algorithm); the facts an algorithm's rows share.
ROWS = load_algorithms()
FACTS = {name: algorithm_facts(name) for name in {row.name for row in ROWS}}
PROBLEMS = {row.name: row for row in load_problems()}


def by_package_kind(specs_or_trials, field="package"):
    return Counter(item[field] for item in specs_or_trials)


def solve_counts(trial_list):
    return Counter(t["package"] for t in trial_list if t["transfers"])


def line_counts(trial_list, package):
    """(lines, lines that optimize, cold lines, builds) of a package's trials."""
    return trials.counts([t for t in trial_list if t["package"] == package])


def capable(package, kind):
    return [row for row in ROWS if row.supports(package, kind)]


def families(algorithm, kind):
    """The package families, cubie and cubie_mlir counted once and julia_cpu untimed, running an algorithm in a stepping kind."""
    return {"cubie" if r.package in ("cubie", "cubie_mlir") else r.package
            for r in ROWS if r.name == algorithm and r[kind] and r.package != "julia_cpu"}


def stepping_values(stepping):
    """How many dt or tolerance values one stepping spells out."""
    if stepping["controller"] == "fixed":
        return len(stepping["dt"]["duration_times_2_pow"])
    return len(stepping["tol"])


def pi_algorithms(package, names=None):
    """Adaptive algorithms (within names when given) whose DIRK PI defaults differ from cubie's shipped controller."""
    import cubie_adapter
    out = []
    for row in capable(package, "adaptive"):
        if names is not None and row.name not in names:
            continue
        shipped = cubie_adapter.default_controller(row.name, row["family"], row["order"])
        if not cubie_adapter.controllers_equal(cubie_adapter.pi_tier_controller(row["order"]), shipped):
            out.append(row)
    return out


def problems_of(package):
    return [row for row in PROBLEMS.values() if row.supports(package)]


def matched_algorithms(package, names, problem):
    """Adaptive algorithms whose julia_cpu controller row under data/ maps to a controller other than cubie's shipped one."""
    import cubie_adapter
    table = sets.controllers_table(KEY, problem, DATA)
    out = []
    for row in capable(package, "adaptive"):
        if row.name not in names or row.name not in table:
            continue
        settings, _ = cubie_adapter.matched_controller(table[row.name], row["order"])
        shipped = cubie_adapter.default_controller(row.name, row["family"], row["order"])
        if settings is not None and not cubie_adapter.controllers_equal(settings, shipped):
            out.append(row)
    return out


def stepping_algorithms(loaded, index, package, problem):
    """The catalogue rows one stepping of a loaded set yields for a package and problem: its list narrowed by capability, pi and matched minus the shipped matches."""
    stepping = loaded["stepping"][index]
    kind = "fixed" if stepping["controller"] == "fixed" else "adaptive"
    names = [r.name for r in capable(package, kind)]
    if stepping["algorithms"] != "all":
        names = [n for n in names if n in stepping["algorithms"]]
    if stepping["packages"] != "all" and package not in stepping["packages"]:
        return []
    if stepping["controller"] == "pi":
        return pi_algorithms(package, names)
    if stepping["controller"] == "matched":
        return matched_algorithms(package, names, problem)
    return [r for r in capable(package, kind) if r.name in names]


def leg_count(loaded, package, problem):
    return sum(len(stepping_algorithms(loaded, i, package, problem)) for i in range(len(loaded["stepping"])))


class ShippedSetTests(unittest.TestCase):
    """Every shipped set's counts, derived from the catalogues and pinned as literals."""

    @classmethod
    def setUpClass(cls):
        cls.expanded = {name: sets.expand([name], KEY, root=DATA) for name in sets.set_names()}
        cls.trials = {name: trials.build_trials(specs) for name, specs in cls.expanded.items()}

    def test_the_four_sets_ship(self):
        self.assertEqual(sets.set_names(), ["golden", "golden_grid", "perf", "states"])

    def test_perf_counts(self):
        specs = self.expanded["perf"]
        self.assertEqual(sorted({s["n"] for s in specs}), PERF_N)
        self.assertEqual({tuple(s["transfers"]) for s in specs}, {("both", "none")})
        self.assertEqual({s["finals"] for s in specs}, {False})
        self.assertNotIn("julia_cpu", by_package_kind(specs))
        loaded = sets.load_set("perf")
        expected = {package: sum(leg_count(loaded, package, p.name) for p in problems_of(package)) * len(PERF_N)
                    for package in loaded["set"]["packages"]}
        self.assertEqual(dict(by_package_kind(specs)), expected)
        self.assertEqual(expected, {"cubie": 1632, "cubie_mlir": 1632, "jax": 360, "pytorch": 180,
                                    "myokit_cuda": 36, "cpp": 120, "julia_gpu": 960})
        # The timed algorithms are exactly those two package families run in the stepping kind.
        timed = {"fixed": ["euler", "classical-rk4", "tsit5", "rosenbrock23_sciml", "kvaerno3", "vern7", "kvaerno5"],
                 "adaptive": ["tsit5", "cash-karp-54", "rosenbrock23_sciml", "kvaerno3", "vern7", "kvaerno5"]}
        for kind, names in timed.items():
            self.assertEqual(sorted(names), sorted(n for n in FACTS if len(families(n, kind)) >= 2), kind)
            self.assertEqual(sorted({s["algorithm"] for s in specs
                                     if (s["controller"] == "fixed") == (kind == "fixed")}), sorted(names))
        built = self.trials["perf"]
        self.assertEqual(dict(solve_counts(built)), expected)
        self.assertEqual(line_counts(built, "cubie"), (1632, 1632, 0, 136))
        self.assertEqual(line_counts(built, "jax"), (360, 0, 0, 30))
        self.assertEqual(line_counts(built, "julia_gpu"), (960, 0, 0, 80))
        self.assertEqual({t["cold"] for t in built}, {False})
        self.assertEqual({tuple(t["sets"]) for t in built}, {("perf",)})
        # Every cubie line optimizes on its own n.
        self.assertEqual({t["optimize"] for t in built if t["package"] == "cubie"}, {"solve"})
        self.assertEqual({t["optimize"] for t in built if t["package"] == "jax"}, {None})

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
        # No shipped set pins the step floor or cap.
        self.assertTrue(np.isnan(default["kvaerno3"]["dt_min"]))
        self.assertTrue(np.isnan(default["kvaerno3"]["dt_max"]))
        self.assertEqual(default["kvaerno3"]["newton_atol"], 1e-5)
        self.assertTrue(np.isnan(default["tsit5"]["newton_atol"]))
        pollu = [s for s in specs if s["problem"] == "pollu" and s["package"] == "cubie"
                 and s["controller"] == "default" and s["algorithm"] == "kvaerno3"][0]
        self.assertEqual(pollu["dt"], 60.0 * 2.0 ** -10)
        self.assertTrue(np.isnan(pollu["dt_min"]))
        self.assertEqual((pollu["parameter"], pollu["grid_scale"], pollu["grid_min"], pollu["grid_max"]),
                         ("k1", "log", 3.5e-2, 3.5))
        self.assertEqual(pollu["system_params"], "{}")
        lorenz96 = [s for s in specs if s["problem"] == "lorenz96"][0]
        self.assertEqual(lorenz96["system_params"], '{"states":32}')
        # Newton tolerances follow the catalogue's newton column: julia_gpu's kvaerno3 row is false, jax's true.
        julia = [s for s in specs if s["package"] == "julia_gpu" and s["algorithm"] == "kvaerno3"
                 and s["problem"] == "lorenz" and s["n"] == 8]
        self.assertEqual(len(julia), 2)
        self.assertTrue(all(np.isnan(s["newton_atol"]) for s in julia))
        jax = [s for s in specs if s["package"] == "jax" and s["algorithm"] == "kvaerno3"
               and s["problem"] == "lorenz" and s["n"] == 8 and s["controller"] == "fixed"][0]
        self.assertEqual(jax["newton_atol"], 1e-6)
        for spec in specs:
            row = [r for r in ROWS if r.package == spec["package"] and r.name == spec["algorithm"]][0]
            self.assertEqual(np.isnan(spec["newton_atol"]), not row["newton"], (spec["package"], spec["algorithm"]))
            self.assertEqual(np.isnan(spec["newton_rtol"]), not row["newton"], (spec["package"], spec["algorithm"]))

    def test_no_shipped_set_pins_dt_min_or_dt_max(self):
        for name, specs in self.expanded.items():
            for field in ("dt_min", "dt_max"):
                self.assertTrue(all(np.isnan(s[field]) for s in specs), (name, field))

    def test_perf_pi_stepping_skips_the_shipped_dirk_controller(self):
        specs = [s for s in self.expanded["perf"] if s["stepping"] == "pi" and s["problem"] == "lorenz"
                 and s["package"] == "cubie" and s["n"] == 8]
        names = sorted(s["algorithm"] for s in specs)
        self.assertEqual(names, sorted(r.name for r in stepping_algorithms(sets.load_set("perf"), 2, "cubie", "lorenz")))
        self.assertEqual(names, ["cash-karp-54", "rosenbrock23_sciml", "tsit5", "vern7"])
        for absent in ("kvaerno3", "kvaerno5"):
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
        self.assertEqual(sorted({json.loads(s["system_params"])["states"] for s in specs}),
                         [4, 8, 16, 32, 64, 128])
        loaded = sets.load_set("states")
        self.assertNotIn("julia_cpu", loaded["set"]["packages"])
        expected = {package: leg_count(loaded, package, "lorenz96") * 6 for package in loaded["set"]["packages"]}
        self.assertEqual(dict(by_package_kind(specs)), expected)
        self.assertEqual(expected["cubie"], 102)
        self.assertEqual(expected["julia_gpu"], 60)
        built = self.trials["states"]
        # Every line is its own cold build, one solve each.
        self.assertEqual(line_counts(built, "cubie"), (102, 102, 102, 102))
        self.assertEqual(line_counts(built, "jax"), (36, 0, 36, 36))
        self.assertEqual({t["cold"] for t in built}, {True})

    def test_golden_grid_counts(self):
        specs = self.expanded["golden_grid"]
        self.assertEqual({tuple(s["transfers"]) for s in specs}, {("none",)})
        self.assertEqual({s["finals"] for s in specs}, {True})
        tables = [p for p in PROBLEMS if sets.controllers_table(KEY, p, DATA)]
        self.assertEqual({s["stepping"] for s in specs},
                         {"fixed", "default", "pi"} | ({"matched"} if tables else set()))
        loaded = sets.load_set("golden_grid")
        expected = {}
        for package in loaded["set"]["packages"]:
            count = 0
            for problem in problems_of(package):
                for index, stepping in enumerate(loaded["stepping"]):
                    count += len(stepping_algorithms(loaded, index, package, problem.name)) * stepping_values(stepping)
            expected[package] = count
        self.assertEqual(dict(by_package_kind(specs)), expected)
        # The cubie counts grow with the julia_cpu controller tables under data/; the others are fixed.
        self.assertEqual({p: expected[p] for p in ("jax", "pytorch", "myokit_cuda", "cpp", "julia_gpu", "julia_cpu")},
                         {"jax": 315, "pytorch": 180, "myokit_cuda": 30, "cpp": 100, "julia_gpu": 800,
                          "julia_cpu": 3192})
        matched = sum(1 for s in specs if s["stepping"] == "matched" and s["package"] == "cubie")
        self.assertEqual(expected["cubie"], 4104 + matched)
        # Every algorithm a package runs at a fixed step is stepped: the two fixed steppings together name them all.
        fixed_named = set(loaded["stepping"][0]["algorithms"]) | set(loaded["stepping"][1]["algorithms"])
        self.assertEqual(fixed_named, set(FACTS))
        for package in loaded["set"]["packages"]:
            self.assertEqual({s["algorithm"] for s in specs if s["package"] == package and s["controller"] == "fixed"},
                             {r.name for r in capable(package, "fixed")}, package)
        kvaerno3 = sorted({s["dt"] for s in specs if s["algorithm"] == "kvaerno3" and s["problem"] == "lorenz"
                           and s["controller"] == "fixed"})
        self.assertEqual(kvaerno3, [2.0 ** -k for k in range(13, 0, -1)])
        # euler alone steps 2^-8 .. 2^-17.
        euler = sorted({s["dt"] for s in specs if s["algorithm"] == "euler" and s["problem"] == "lorenz"})
        self.assertEqual(euler, [2.0 ** -k for k in range(17, 7, -1)])
        self.assertEqual({s["controller"] for s in specs if s["algorithm"] == "euler"}, {"fixed"})
        self.assertEqual(sorted({s["atol"] for s in specs if s["controller"] != "fixed"}), sorted(TOLS))
        built = self.trials["golden_grid"]
        lines, optimized, cold, builds = line_counts(built, "cubie")
        self.assertEqual((lines, optimized, cold), (expected["cubie"], expected["cubie"], 0))
        # One build per (problem, algorithm, controller, gains); a matched entry resolving to pi shares the pi build.
        self.assertGreaterEqual(builds, 8 * (23 + 17 + 14))
        self.assertEqual(line_counts(built, "julia_cpu"), (3192, 0, 0, 8 * (21 + 18)))
        self.assertEqual(line_counts(built, "pytorch"), (180, 0, 0, 15))

    def test_golden_grid_optimizes_every_cubie_kernel_once(self):
        built = [t for t in self.trials["golden_grid"] if t["package"] == "cubie" and t["problem"] == "lorenz"]
        self.assertEqual({t["optimize"] for t in built}, {"kernel"})
        self.assertEqual(len({trials.kernel_key(t) for t in built}), len(built))
        states = [t for t in self.trials["states"] if t["package"] == "cubie"]
        self.assertEqual({t["optimize"] for t in states}, {"kernel"})
        self.assertEqual(trials.optimizes_of(states), len(states))
        self.assertEqual({t["optimize"] for t in self.trials["golden_grid"] if t["package"] != "cubie"
                          and t["package"] != "cubie_mlir"}, {None})

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
        self.assertEqual(line_counts(built, "julia_cpu"), (8, 0, 0, 8))

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

    def test_declarations_expand_every_set_file_and_declared_counts_union_the_grids(self):
        every = sets.declarations(KEY, root=DATA, packages=["julia_cpu"], problems=["lorenz"])
        self.assertEqual({s["set"] for s in every}, {"golden", "golden_grid"})
        self.assertEqual({s["n"] for s in every}, {1024, 131072})
        self.assertEqual(sets.declared_counts(["golden_grid"]), [1024, 131072])
        self.assertEqual(sets.declared_counts(["perf", "golden_grid"]), sorted(set(PERF_N) | {1024}))
        self.assertEqual(sets.declared_counts(["states", "golden"]), [131072])

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
            self.assertEqual(gains["integral_gain"], 0.3 * 4 / FACTS[spec["algorithm"]]["order"])
            self.assertEqual(gains["proportional_gain"], 0.4 * 4 / FACTS[spec["algorithm"]]["order"])
            self.assertEqual(spec["stepping"], "pi")

    def test_explicit_gains_and_a_named_controller_pass_through(self):
        self.write_set("g", self.ADAPTIVE.format(controller="pid", gains="gains = {kp = 0.7, ki = 0.4}"))
        specs = sets.expand(["g"], KEY, self.root, algorithms=["tsit5"], sets_dir=self.sets_dir)
        self.assertEqual(len(specs), 1)
        self.assertEqual((specs[0]["controller"], specs[0]["gains"]), ("pid", '{"ki":0.4,"kp":0.7}'))

    def test_newton_resolves_from_the_catalogue_column(self):
        # The same stepping asks for newton = "tol"; the (package, algorithm) row decides who gets it.
        text = self.ADAPTIVE.format(controller="default", gains="").replace('["cubie"]', '["cubie", "julia_gpu", "jax"]')
        self.write_set("nw", text)
        specs = sets.expand(["nw"], KEY, self.root, algorithms=["kvaerno3", "tsit5"], sets_dir=self.sets_dir)
        by_key = {(s["package"], s["algorithm"]): s for s in specs}
        self.assertEqual(set(by_key), {("cubie", "kvaerno3"), ("cubie", "tsit5"), ("julia_gpu", "kvaerno3"),
                                       ("julia_gpu", "tsit5"), ("jax", "kvaerno3"), ("jax", "tsit5")})
        self.assertEqual((by_key[("cubie", "kvaerno3")]["newton_atol"], by_key[("jax", "kvaerno3")]["newton_rtol"]),
                         (1e-5, 1e-5))
        for pair in (("julia_gpu", "kvaerno3"), ("cubie", "tsit5"), ("julia_gpu", "tsit5"), ("jax", "tsit5")):
            self.assertTrue(np.isnan(by_key[pair]["newton_atol"]), pair)
            self.assertTrue(np.isnan(by_key[pair]["newton_rtol"]), pair)

    def test_matched_and_dirk_defaults_are_cubie_only(self):
        text = self.ADAPTIVE.format(controller="matched", gains="").replace('["cubie"]', '["jax"]')
        self.write_set("bad", text)
        self.write_controllers("lorenz", [("tsit5", "PIController", "0.28", "0.04", "0.2", "10.0", "0.9", "5")])
        with self.assertRaises(sets.SetError):
            sets.expand(["bad"], KEY, self.root, sets_dir=self.sets_dir)


class MergeAndNarrowTests(unittest.TestCase):
    """Merge, ordinals and narrowing against an empty data root, so no controllers table resolves a matched stepping."""

    def setUp(self):
        import cubie_adapter
        for name in ("default_controller", "pi_tier_controller"):
            original = getattr(cubie_adapter, name)
            setattr(cubie_adapter, name, getattr(FakeControllers, name))
            self.addCleanup(setattr, cubie_adapter, name, original)
        self.root = tempfile.mkdtemp(prefix="sets_narrow_")
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_a_trial_shared_by_two_sets_merges_transfers_and_finals(self):
        specs = sets.expand(["perf", "golden_grid"], KEY, self.root, packages=["jax"], problems=["lorenz"])
        built = trials.build_trials(specs)
        shared = [t for t in built if t["n"] == 131072 and t["controller"] == "fixed"
                  and t["algorithm"] == "kvaerno3" and t["dt"] == 2.0 ** -10]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]["transfers"], ["both", "none"])
        self.assertTrue(shared[0]["finals"])
        self.assertEqual(shared[0]["sets"], ["golden_grid", "perf"])
        ids = [t["trial_id"] for t in built]
        self.assertEqual(len(ids), len(set(ids)))
        # The kvaerno3 fixed build: perf's counts below 131072, then the thirteen steps at 131072, then the rest.
        kvaerno3 = [(t["n"], t["dt"]) for t in built if t["algorithm"] == "kvaerno3" and t["controller"] == "fixed"]
        at_131072 = [dt for n, dt in kvaerno3 if n == 131072]
        self.assertEqual(at_131072, [2.0 ** -k for k in range(1, 14)])
        self.assertEqual([n for n, _ in kvaerno3], sorted(n for n, _ in kvaerno3))
        reverse = trials.build_trials(sets.expand(["golden_grid", "perf"], KEY, self.root,
                                                  packages=["jax"], problems=["lorenz"]))
        self.assertEqual(reverse, built)

    def test_every_set_file_declares_the_contract_of_a_requested_point(self):
        """The lorenz96 default-states point at n = 131072: perf (warm, per-solve optimize), states (cold, per-kernel) and golden_grid (finals, none, per-kernel) declare it; each request alone yields the same line."""
        declared = sets.declarations(KEY, self.root, packages=["cubie"], problems=["lorenz96"],
                                     algorithms=["kvaerno3"])
        expected = None
        for names in (["perf"], ["states"], ["golden_grid"], ["golden_grid", "perf"], ["perf", "states", "golden_grid"]):
            specs = sets.narrow(sets.expand(names, KEY, self.root, packages=["cubie"], problems=["lorenz96"],
                                            algorithms=["kvaerno3"]), mode="fixed")
            built = trials.build_trials(specs, declared)
            point = [t for t in built if t["n"] == 131072 and t["system_params"] == '{"states":32}'
                     and t["dt"] == 2.0 ** -10]
            self.assertEqual(len(point), 1, names)
            contract = {k: point[0][k] for k in ("cold", "finals", "transfers", "optimize", "watchdog_s", "sets")}
            if expected is None:
                expected = contract
            self.assertEqual(contract, expected, names)
        self.assertEqual(expected, {"cold": True, "finals": True, "transfers": ["both", "none"], "optimize": "solve",
                                    "watchdog_s": protocol.WATCHDOG_SECONDS,
                                    "sets": ["golden_grid", "perf", "states"]})
        # perf alone: the other eleven counts of the 32-state build stay warm and unshared.
        perf = trials.build_trials(sets.narrow(sets.expand(["perf"], KEY, self.root, packages=["cubie"],
                                                           problems=["lorenz96"], algorithms=["kvaerno3"]),
                                               mode="fixed"), declared)
        self.assertEqual([t["n"] for t in perf], PERF_N)
        self.assertEqual([(t["cold"], t["finals"], t["sets"]) for t in perf if t["n"] != 131072],
                         [(False, False, ["perf"])] * 11)
        # Without the declarations the request alone decides.
        alone = trials.build_trials(sets.narrow(sets.expand(["perf"], KEY, self.root, packages=["cubie"],
                                                            problems=["lorenz96"], algorithms=["kvaerno3"]),
                                                mode="fixed"))
        self.assertEqual({t["finals"] for t in alone}, {False})
        self.assertEqual({t["cold"] for t in alone}, {False})

    def test_the_canonical_merge_is_true_solve_or_largest_over_the_declarations(self):
        base = dict(problem="lorenz", system_params="{}", duration=1.0, precision="float32", parameter="rho",
                    grid_scale="linear", grid_min=0.0, grid_max=21.0, n=8, grid_dtype="float32",
                    algorithm="tsit5", controller="fixed", dt=2.0 ** -10, dt_min=NAN, dt_max=NAN, atol=NAN,
                    rtol=NAN, gains="{}", newton_atol=NAN, newton_rtol=NAN, package="cubie")
        a = dict(base, transfers=["none"], finals=True, build="warm", optimize={"per": "kernel"},
                 watchdog_s=60.0, set="a", stepping="fixed")
        b = dict(base, transfers=["both"], finals=False, build="cold", optimize={"per": "solve"},
                 watchdog_s=600.0, set="b", stepping="fixed")
        c = dict(base, transfers=["both"], finals=False, build="warm", optimize=None,
                 watchdog_s=30.0, set="c", stepping="fixed")
        for order in ((a, b, c), (c, b, a), (b, a, c)):
            built = trials.build_trials(list(order))
            self.assertEqual(len(built), 1, order)
            line = built[0]
            self.assertEqual({k: line[k] for k in ("n", "cold", "transfers", "finals", "watchdog_s", "sets", "optimize")},
                             {"n": 8, "cold": True, "transfers": ["both", "none"], "finals": True, "watchdog_s": 600.0,
                              "sets": ["a", "b", "c"], "optimize": "solve"}, order)
        # A point requested once and declared elsewhere takes the declarations; a declaration of another point is ignored.
        other = dict(a, n=32, set="d", build="cold")
        built = trials.build_trials([a], declared=[b, c, other])
        self.assertEqual([(t["n"], t["cold"], t["optimize"], t["sets"]) for t in built],
                         [(8, True, "solve", ["a", "b", "c"])])
        self.assertEqual(trials.canonical_optimize([None, {"per": "kernel"}]), "kernel")
        self.assertEqual(trials.canonical_optimize([{"per": "kernel"}, {"per": "solve"}]), "solve")
        self.assertIsNone(trials.canonical_optimize([None, None]))
        # Optimize counts: one per per-solve line, one per kernel among per-kernel lines.
        kernel_lines = trials.build_trials([a, dict(a, n=32), dict(a, dt=0.5)])
        self.assertEqual(trials.optimizes_of(kernel_lines), 2)
        self.assertEqual(trials.counts(kernel_lines)[1], 2)
        self.assertEqual(trials.optimizes_of(trials.build_trials([b, dict(b, n=32), dict(b, dt=0.5)])), 3)
        # File order: states, then n ascending, dt descending, tolerance descending, per build.
        lines = [dict(a, n=32), dict(a, dt=0.5), dict(a, n=8), dict(a, system_params='{"states":4}', problem="lorenz96"),
                 dict(a, controller="default", dt=NAN, atol=1e-3, rtol=1e-3, gains="{}"),
                 dict(a, controller="default", dt=NAN, atol=1e-5, rtol=1e-5, gains="{}")]
        built = trials.build_trials(lines)
        self.assertEqual([(t["problem"], t["controller"], t["n"], t["dt"] if t["controller"] == "fixed" else t["atol"])
                          for t in built],
                         [("lorenz", "default", 8, 1e-3), ("lorenz", "default", 8, 1e-5),
                          ("lorenz", "fixed", 8, 0.5), ("lorenz", "fixed", 8, 2.0 ** -10), ("lorenz", "fixed", 32, 2.0 ** -10),
                          ("lorenz96", "fixed", 8, 2.0 ** -10)])
        self.assertEqual([len(v) for _, v in trials.builds_of(built)], [2, 3, 1])
        # The difficulty order: harder means every entry at least as hard and one harder.
        self.assertTrue(trials.harder(dict(a, n=32), a))
        self.assertTrue(trials.harder(dict(a, dt=2.0 ** -12), a))
        self.assertTrue(trials.harder(dict(a, n=32, dt=2.0 ** -12), a))
        self.assertFalse(trials.harder(dict(a, n=32, dt=0.5), a))
        self.assertFalse(trials.harder(a, a))
        self.assertTrue(trials.harder(dict(a, system_params='{"states":64}'), dict(a, system_params='{"states":32}')))
        tol = dict(a, controller="default", dt=NAN, atol=1e-5, rtol=1e-5)
        self.assertTrue(trials.harder(dict(tol, atol=1e-6, rtol=1e-6), tol))
        self.assertFalse(trials.harder(dict(tol, atol=1e-4, rtol=1e-4), tol))

    def test_the_file_order_runs_the_easy_lines_first(self):
        built = trials.build_trials(sets.expand(["golden_grid"], KEY, self.root, packages=["jax"],
                                                problems=["lorenz"], algorithms=["kvaerno3"]))
        self.assertEqual([t["dt"] for t in built if t["controller"] == "fixed"], [2.0 ** -k for k in range(1, 14)])
        self.assertEqual([t["atol"] for t in built if t["controller"] != "fixed"], TOLS)
        self.assertEqual(len(trials.builds_of(built)), 2)
        perf = trials.build_trials(sets.expand(["perf"], KEY, self.root, packages=["jax"], problems=["lorenz"],
                                               algorithms=["tsit5"]))
        self.assertEqual([t["n"] for t in perf if t["controller"] == "fixed"], PERF_N)

    def test_narrowing_by_package_problem_algorithm_and_n(self):
        specs = sets.expand(["perf"], KEY, self.root, packages=["pytorch"], problems=["pollu", "lorenz"],
                            algorithms=["euler", "tsit5"], n=[8, 32])
        self.assertEqual({s["package"] for s in specs}, {"pytorch"})
        self.assertEqual({s["problem"] for s in specs}, {"lorenz", "pollu"})
        self.assertEqual({s["algorithm"] for s in specs}, {"euler", "tsit5"})
        self.assertEqual(sorted({s["n"] for s in specs}), [8, 32])
        self.assertEqual(len(specs), 2 * 2 * 2)
        # A grid keeps only the counts n names.
        golden = sets.expand(["golden_grid"], KEY, self.root, packages=["jax"], problems=["lorenz"],
                             algorithms=["kvaerno3"], n=[16, 131072])
        self.assertEqual({s["n"] for s in golden}, {131072})
        self.assertEqual(sets.expand(["golden_grid"], KEY, self.root, packages=["jax"], problems=["lorenz"],
                                     algorithms=["kvaerno3"], n=[16]), [])

    def test_narrowing_by_mode_controller_tol_and_dt(self):
        specs = sets.expand(["golden_grid"], KEY, self.root, packages=["cubie"], problems=["lorenz"],
                            algorithms=["rosenbrock23_sciml"])
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
        self.assertEqual(head["watchdog"], protocol.WATCHDOG_SECONDS)
        self.assertEqual(loaded["grid"][0]["parameter"], "default")
        self.assertEqual(loaded["stepping"][0]["newton"], "none")
        self.assertIsNone(head["optimize"])
        specs = sets.expand(["s"], KEY, sets_dir=self.tmp)
        self.assertEqual(len(specs), len(capable("jax", "fixed")))
        self.assertEqual({s["dt"] for s in specs}, {0.5})
        self.assertTrue(all(np.isnan(s["newton_atol"]) for s in specs))
        self.assertEqual({s["optimize"] for s in specs}, {None})
        self.assertEqual({s["watchdog_s"] for s in specs}, {protocol.WATCHDOG_SECONDS})
        built = trials.build_trials(specs)
        self.assertEqual({t["watchdog_s"] for t in built}, {protocol.WATCHDOG_SECONDS})
        self.assertEqual({t["optimize"] for t in built}, {None})
        self.assertEqual(len(built), len(specs))

    def test_optimize_table_names_the_policy(self):
        text = self.MINIMAL.replace("[[grid]]\nn = [8]", "[[grid]]\nn = [8, 32]") + \
            '\n[set.optimize]\npackages = ["jax"]\nper = "solve"\n'
        self.write(text)
        specs = sets.expand(["s"], KEY, sets_dir=self.tmp, algorithms=["euler"])
        self.assertEqual({s["optimize"]["per"] for s in specs}, {"solve"})
        built = trials.build_trials(specs)
        self.assertEqual([(t["n"], t["optimize"]) for t in built], [(8, "solve"), (32, "solve")])
        self.write(text.replace('per = "solve"', 'per = "kernel"'))
        built = trials.build_trials(sets.expand(["s"], KEY, sets_dir=self.tmp, algorithms=["euler"]))
        self.assertEqual([(t["n"], t["optimize"]) for t in built], [(8, "kernel"), (32, "kernel")])
        self.write(text.replace('["jax"]\nper = "solve"', '["cubie"]\nper = "solve"'))
        built = trials.build_trials(sets.expand(["s"], KEY, sets_dir=self.tmp, algorithms=["euler"]))
        self.assertEqual({t["optimize"] for t in built}, {None})
        for bad in ('per = "leg"', 'n = 64\nper = "solve"', 'per = "solve"\nshape = 1', ''):
            self.write(text.replace('per = "solve"', bad))
            with self.assertRaises(sets.SetError, msg=bad):
                sets.expand(["s"], KEY, sets_dir=self.tmp)
        self.write(text.replace('packages = ["jax"]\nper = "solve"', 'packages = ["fortran"]\nper = "solve"'))
        with self.assertRaises(sets.SetError):
            sets.expand(["s"], KEY, sets_dir=self.tmp)

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
            "bad watchdog": self.MINIMAL.replace("[set]", "[set]\nwatchdog = 0"),
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
        self.assertEqual(len(sets.expand(["s"], KEY, sets_dir=self.tmp)), 4)


class WatchdogBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sets_watchdog_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, name, text):
        with open(os.path.join(self.tmp, name + ".toml"), "w", encoding="utf-8") as handle:
            handle.write(text)

    def test_the_golden_set_allows_a_day_per_solve(self):
        self.assertEqual(sets.load_set("golden")["set"]["watchdog"], 86400.0)
        for name in ("perf", "states", "golden_grid"):
            self.assertEqual(sets.load_set(name)["set"]["watchdog"], protocol.WATCHDOG_SECONDS)
        specs = sets.expand(["golden"], KEY, problems=["lorenz"])
        self.assertEqual({s["watchdog_s"] for s in specs}, {86400.0})
        self.assertEqual({t["watchdog_s"] for t in trials.build_trials(specs)}, {86400.0})

    def test_a_shared_trial_keeps_the_larger_budget_and_the_file_carries_it(self):
        base = '[set]\npackages = ["jax"]\nproblems = ["lorenz"]\nalgorithms = ["euler"]\n{0}[[grid]]\nn = [8]\n[[stepping]]\ncontroller = "fixed"\ndt = [0.5]\n'
        self.write("short", base.format("watchdog = 60\n"))
        self.write("long", base.format("watchdog = 600\n"))
        specs = sets.expand(["short", "long"], KEY, sets_dir=self.tmp)
        built = trials.build_trials(specs)
        self.assertEqual(len(built), 1)
        self.assertEqual({t["watchdog_s"] for t in built}, {600.0})
        path = trials.write_jsonl(os.path.join(self.tmp, "jax.jsonl"), built)
        back = trials.read_jsonl(path)
        self.assertEqual([t["watchdog_s"] for t in back], [600.0])
        with open(path, encoding="utf-8") as handle:
            self.assertIn('"watchdog_s": 600.0', handle.readline())


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
        first = json.loads(lines[1])
        self.assertEqual(list(first), list(trials.TRIAL_KEYS))
        self.assertNotIn("nan", lines[1].lower())
        self.assertIsNone(first["atol"])
        self.assertIs(first["cold"], False)
        self.assertIsNone(first["optimize"])
        self.assertEqual(first["sets"], ["perf"])
        back = trials.read_jsonl(path)
        self.assertEqual(len(back), len(built))
        self.assertTrue(np.isnan(back[1]["atol"]))
        self.assertEqual(back[1]["trial_id"], store.trial_id(back[1]))
        self.assertEqual(trials.counts(back), (2, 0, 0, 2))
        self.assertEqual([t["algorithm"] for t in back], ["cash-karp-54", "classical-rk4"])


if __name__ == "__main__":
    unittest.main()
