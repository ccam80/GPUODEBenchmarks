"""Cubie adapter: backend selection, controller mappings and the optimize store."""

import math
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cubie_adapter as adapter  # noqa: E402
from problems import get_problem  # noqa: E402


class FakeLaunch:
    def __init__(self, blocksize, resident_blocks, best_ms=1.0):
        self.blocksize = blocksize
        self.resident_blocks = resident_blocks
        self.best_ms = best_ms
        self.label = "state=shared @bs{0}".format(blocksize)


class FakeResult:
    def __init__(self, best, settings):
        self.best = best
        self.applied_settings = settings


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.get("CUBIE_CUDA_BACKEND")
        self.loaded = sys.modules.pop("cubie.cuda_backend", None)

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("CUBIE_CUDA_BACKEND", None)
        else:
            os.environ["CUBIE_CUDA_BACKEND"] = self.saved
        sys.modules.pop("cubie.cuda_backend", None)
        if self.loaded is not None:
            sys.modules["cubie.cuda_backend"] = self.loaded

    def test_each_package_sets_its_backend_unconditionally(self):
        os.environ["CUBIE_CUDA_BACKEND"] = "mlir"
        self.assertEqual(adapter.select_backend("cubie"), "numba-cuda")
        self.assertEqual(os.environ["CUBIE_CUDA_BACKEND"], "numba-cuda")
        self.assertEqual(adapter.select_backend("cubie_mlir"), "mlir")
        self.assertEqual(os.environ["CUBIE_CUDA_BACKEND"], "mlir")

    def test_an_imported_backend_cannot_be_switched(self):
        import types
        loaded = types.ModuleType("cubie.cuda_backend")
        loaded.CUDA_BACKEND = "mlir"
        sys.modules["cubie.cuda_backend"] = loaded
        with self.assertRaises(RuntimeError):
            adapter.select_backend("cubie")
        self.assertEqual(adapter.select_backend("cubie_mlir"), "mlir")

    def test_system_suffix_per_package(self):
        self.assertEqual(adapter.SYSTEM_SUFFIX["cubie"], "")
        self.assertEqual(adapter.SYSTEM_SUFFIX["cubie_mlir"], "_mlir")
        self.assertEqual(adapter.package_for_backend("mlir"), "cubie_mlir")


class ControllerMappingTests(unittest.TestCase):
    def test_julia_pi_constants_map_to_cubie_gains(self):
        # cubie: (I + P) / (2 (order + 1)) on the squared norm = beta1 / 2.
        constants = {"controller": "PIController", "beta1": 0.28,
                     "beta2": 0.04, "qmin": 0.2, "qmax": 10.0, "gamma": 0.9}
        settings, why = adapter.matched_controller(constants, 4)
        self.assertIsNone(why)
        self.assertEqual(settings["step_controller"], "pi")
        self.assertAlmostEqual(settings["proportional_gain"], 0.04 * 5)
        self.assertAlmostEqual(settings["integral_gain"], 0.28 * 5 - 0.04 * 5)
        self.assertAlmostEqual(settings["safety"], 0.9)
        self.assertAlmostEqual(settings["min_step_shrink"], 0.2)
        self.assertAlmostEqual(settings["max_step_growth"], 10.0)

    def test_predictive_controller_maps_to_gustafsson(self):
        constants = {"controller": "PredictiveController", "beta1": None,
                     "beta2": None, "qmin": 0.2, "qmax": 8.0, "gamma": 0.9}
        settings, why = adapter.matched_controller(constants, 5)
        self.assertEqual(settings, {"step_controller": "gustafsson",
                                    "safety": 0.9})

    def test_unmapped_controllers_give_a_reason(self):
        self.assertEqual(adapter.matched_controller(None, 3)[0], None)
        settings, why = adapter.matched_controller(
            {"controller": "Other"}, 3)
        self.assertIsNone(settings)
        self.assertIn("Other", why)

    def test_controllers_equal_compares_names_and_numbers(self):
        a = {"step_controller": "pi", "integral_gain": 0.3, "safety": 0.9}
        self.assertTrue(adapter.controllers_equal(a, dict(a)))
        self.assertFalse(adapter.controllers_equal(a, dict(a, safety=0.8)))
        self.assertFalse(adapter.controllers_equal(
            a, dict(a, step_controller="i")))
        self.assertFalse(adapter.controllers_equal(a, {"step_controller": "pi"}))
        self.assertFalse(adapter.controllers_equal(a, None))

    def test_shipped_tables_resolve_order_dependent_gains(self):
        dirk = adapter.default_controller("kvaerno3", "dirk", 3)
        self.assertEqual(dirk["step_controller"], "pi")
        self.assertAlmostEqual(dirk["integral_gain"], 0.3 * 4 / 3)
        self.assertAlmostEqual(dirk["proportional_gain"], 0.4 * 4 / 3)
        self.assertNotIn("attempt_dense_prediction", dirk)
        erk = adapter.default_controller("tsit5", "erk", 5)
        self.assertEqual(erk["step_controller"], "i")
        self.assertNotIn("proportional_gain", erk)
        self.assertIsNone(adapter.default_controller("euler", "explicit", 1))
        self.assertTrue(adapter.controllers_equal(
            adapter.pi_tier_controller(3), dirk))


class OptimizeStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmp)
        self.problem = get_problem("lorenz")

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_store_lives_under_the_package_partition(self):
        path = adapter.optimize_path("cubie_mlir", "k")
        self.assertEqual(os.path.relpath(path, self.tmp),
                         os.path.join("data", "key=k", "package=cubie_mlir", "optimize.csv"))

    def test_a_row_without_a_setting_serves_every_setting_of_the_leg(self):
        result = FakeResult(FakeLaunch(256, 3),
                            {"state_location": "shared", "blocksize": 256})
        adapter.record_optimized("cubie", "k", self.problem, "tsit5",
                                 "fixed", None, result)
        for setting in (2.0 ** -10, 0.0625, 2.0 ** -13):
            tuned = adapter.load_optimized("cubie", "k", self.problem,
                                           "tsit5", "fixed", setting)
            self.assertEqual(tuned["settings"],
                             {"state_location": "shared", "blocksize": 256})
            self.assertEqual(tuned["resident_blocks"], 3)

    def test_implicit_rows_are_per_setting(self):
        result = FakeResult(FakeLaunch(64, None), {"blocksize": 64})
        adapter.record_optimized("cubie", "k", self.problem, "kvaerno3",
                                 "adaptive", 1e-3, result)
        self.assertIsNotNone(adapter.load_optimized(
            "cubie", "k", self.problem, "kvaerno3", "adaptive", 1e-3))
        self.assertIsNone(adapter.load_optimized(
            "cubie", "k", self.problem, "kvaerno3", "adaptive", 1e-4))
        self.assertIsNone(adapter.load_optimized(
            "cubie_mlir", "k", self.problem, "kvaerno3", "adaptive", 1e-3))

    def test_states_and_rerecording_replace_by_identity(self):
        first = FakeResult(FakeLaunch(64, 2), {"blocksize": 64})
        second = FakeResult(FakeLaunch(128, None), {"blocksize": 128})
        adapter.record_optimized("cubie", "k", "lorenz96", "tsit5", "fixed",
                                 None, first, states=8)
        adapter.record_optimized("cubie", "k", "lorenz96", "tsit5", "fixed",
                                 None, second, states=8)
        tuned = adapter.load_optimized("cubie", "k", "lorenz96", "tsit5",
                                       "fixed", None, states=8)
        self.assertEqual(tuned["settings"]["blocksize"], 128)
        self.assertIsNone(tuned["resident_blocks"])
        self.assertIsNone(adapter.load_optimized(
            "cubie", "k", "lorenz96", "tsit5", "fixed", None, states=16))
        with open(adapter.optimize_path("cubie", "k")) as handle:
            self.assertEqual(sum(1 for _ in handle) - 1, 1)

    def test_controller_and_gains_separate_rows_of_one_algorithm(self):
        result = FakeResult(FakeLaunch(64, None), {"blocksize": 64})
        other = FakeResult(FakeLaunch(256, 1), {"blocksize": 256})
        adapter.record_optimized("cubie", "k", self.problem, "tsit5",
                                 "adaptive", None, result, controller="default",
                                 gains="{}")
        adapter.record_optimized("cubie", "k", self.problem, "tsit5",
                                 "adaptive", None, other, controller="pi",
                                 gains='{"integral_gain":0.3}')
        shipped = adapter.load_optimized("cubie", "k", self.problem, "tsit5",
                                         "adaptive", 1e-5, controller="default",
                                         gains="{}")
        self.assertEqual(shipped["settings"]["blocksize"], 64)
        tuned = adapter.load_optimized("cubie", "k", self.problem, "tsit5",
                                       "adaptive", 1e-5, controller="pi",
                                       gains='{"integral_gain":0.3}')
        self.assertEqual(tuned["settings"]["blocksize"], 256)
        self.assertIsNone(adapter.load_optimized(
            "cubie", "k", self.problem, "tsit5", "adaptive", 1e-5))
        with open(adapter.optimize_path("cubie", "k")) as handle:
            self.assertEqual(sum(1 for _ in handle) - 1, 2)

    def test_clear_narrows_by_algorithm_and_problem(self):
        result = FakeResult(FakeLaunch(64, None), {"blocksize": 64})
        for algorithm, problem in (("tsit5", "lorenz"), ("euler", "lorenz"),
                                   ("tsit5", "pollu")):
            adapter.record_optimized("cubie", "k", problem, algorithm,
                                     "fixed", None, result)
        self.assertEqual(adapter.clear_optimized("cubie", "k", "tsit5",
                                                 "lorenz"), 1)
        self.assertEqual(adapter.clear_optimized("cubie", "k", "all",
                                                 "pollu"), 1)
        self.assertEqual(adapter.clear_optimized("cubie", "k"), 1)
        self.assertEqual(adapter.clear_optimized("cubie", "k"), 0)

    def test_unroll_enums_round_trip(self):
        from cubie.cuda_simsafe import UnrollChoice
        result = FakeResult(FakeLaunch(64, 1), {
            "unroll_other_small": UnrollChoice.ROLLED,
            "state_location": "local", "blocksize": 64})
        adapter.record_optimized("cubie", "k", self.problem, "tsit5",
                                 "adaptive", None, result)
        tuned = adapter.load_optimized("cubie", "k", self.problem, "tsit5",
                                       "adaptive", 1e-5)
        self.assertIs(tuned["settings"]["unroll_other_small"],
                      UnrollChoice.ROLLED)

    def test_find_optimized_reads_the_rows_an_optimize_line_identifies(self):
        result = FakeResult(FakeLaunch(64, 2), {"blocksize": 64})
        adapter.record_optimized("cubie", "k", self.problem, "tsit5", "fixed", None, result,
                                 controller="fixed", gains="{}", source="S")
        adapter.record_optimized("cubie", "k", self.problem, "kvaerno3", "adaptive", 1e-3, result,
                                 controller="default", gains="{}", source="S")
        rows = adapter.optimize_rows("cubie", "k")
        self.assertEqual([r["algorithm"] for r in rows], ["tsit5", "kvaerno3"])
        line = dict(package="cubie", problem="lorenz", algorithm="tsit5", controller="fixed", gains="{}",
                    system_params="{}", axis="dt", dt=2.0 ** -10, atol=math.nan)
        ident = adapter.optimize_ident(line, "k")
        self.assertEqual((ident["mode"], ident["setting"], ident["states"], ident["controller"]),
                         ("fixed", "0.0009765625", "3", "fixed"))
        # The leg-wide tsit5 row serves the stepped line; kvaerno3 matches its own tolerance alone.
        self.assertEqual(adapter.find_optimized(rows, ident)["source"], "S")
        tol = dict(line, algorithm="kvaerno3", controller="default", axis="tol", dt=math.nan, atol=1e-3)
        self.assertEqual(adapter.find_optimized(rows, adapter.optimize_ident(tol, "k"))["setting"], "0.001")
        self.assertIsNone(adapter.find_optimized(rows, adapter.optimize_ident(dict(tol, atol=1e-4), "k")))
        self.assertIsNone(adapter.find_optimized(rows, adapter.optimize_ident(dict(line, system_params='{"states":4}'), "k")))
        self.assertEqual(adapter.optimize_rows("cubie_mlir", "k"), [])
        # A batch in the identity matches rows of that batch alone.
        adapter.record_optimized("cubie", "k", self.problem, "tsit5", "fixed", None, result, n=64,
                                 controller="fixed", gains="{}", source="S")
        rows = adapter.optimize_rows("cubie", "k")
        self.assertIsNotNone(adapter.find_optimized(rows, adapter.optimize_ident(dict(line, optimize=64), "k")))
        self.assertIsNone(adapter.find_optimized(rows, adapter.optimize_ident(dict(line, optimize=128), "k")))
        self.assertEqual(adapter.load_optimized("cubie", "k", self.problem, "tsit5", "fixed", None, n=64,
                                                controller="fixed", gains="{}")["resident_blocks"], 2)
        self.assertIsNone(adapter.load_optimized("cubie", "k", self.problem, "tsit5", "fixed", None, n=128,
                                                 controller="fixed", gains="{}"))

    def test_source_hashes_come_from_the_package_interpreter(self):
        systems = [("lorenz", "{}", "float32"), ("lorenz96", '{"states":8}', "float32"), ("lorenz", "{}", "float32")]
        try:
            hashes = adapter.source_hashes("cubie", systems)
        except RuntimeError as exc:
            self.skipTest("cubie is not importable by the package interpreter: {0}".format(exc))
        self.assertEqual(set(hashes), set(systems))
        self.assertEqual({len(h) for h in hashes.values()}, {16})
        self.assertNotEqual(hashes[systems[0]], hashes[systems[1]])
        self.assertEqual(adapter.source_hashes("cubie", []), {})
        with self.assertRaises(RuntimeError):
            adapter.source_hashes("cubie", [("nosuchproblem", "{}", "float32")])

    def test_optimize_point_records_the_batch_size(self):
        import numpy as np

        class Solver:
            def optimize(self, initial_values, parameters, duration,
                         verbose, force=False):
                self.seen = (initial_values.shape, duration, force)
                return FakeResult(FakeLaunch(64, None, 2.5),
                                  {"blocksize": 64})

        solver = Solver()
        row = adapter.optimize_point(
            solver, self.problem, np.zeros((3, 512)), np.zeros((1, 512)),
            "cubie", "k", "tsit5", "fixed", None)
        self.assertEqual(solver.seen, ((3, 512), self.problem["duration"], False))
        self.assertEqual(row["n"], "512")
        self.assertEqual(float(row["best_ms"]), 2.5)
        forced = adapter.optimize_point(
            solver, self.problem, np.zeros((3, 512)), np.zeros((1, 512)),
            "cubie", "k", "tsit5", "fixed", None, root=os.path.join(self.tmp, "elsewhere"),
            force=True)
        self.assertEqual(solver.seen[2], True)
        self.assertEqual(forced["n"], "512")
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "elsewhere", "key=k", "package=cubie",
                                                    "optimize.csv")))


if __name__ == "__main__":
    unittest.main()
