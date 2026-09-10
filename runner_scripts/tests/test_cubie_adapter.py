"""Cubie adapter: backend selection, controller mappings and the optimize store."""

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


NAN = float("nan")


def line(n=8, optimize="kernel", **overrides):
    """A cubie trial line: lorenz, fixed tsit5 at dt 2^-10."""
    fields = dict(package="cubie", problem="lorenz", system_params="{}", precision="float32", n=n,
                  algorithm="tsit5", controller="fixed", gains="{}", dt=2.0 ** -10, dt_min=NAN, dt_max=NAN,
                  atol=NAN, rtol=NAN, newton_atol=NAN, newton_rtol=NAN, duration=1.0, optimize=optimize)
    fields.update(overrides)
    return fields


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

    def test_the_identity_is_the_kernel_and_the_policy(self):
        ident = adapter.optimize_ident(line(), "k")
        self.assertEqual(ident, {"package": "cubie", "key": "k", "problem": "lorenz", "states": "3",
                                 "precision": "float32", "algorithm": "tsit5", "controller": "fixed",
                                 "gains": "{}", "per": "kernel",
                                 "stepping": "dt=0.0009765625;dt_min=;dt_max=;atol=;rtol=;newton_atol=;newton_rtol="})
        solve = adapter.optimize_ident(line(n=64, optimize="solve"), "k")
        self.assertEqual((solve["per"], solve["n"]), ("solve", "64"))
        self.assertEqual(adapter.kernel_ident(line(n=64, optimize="solve"), "k"), adapter.kernel_ident(line(), "k"))
        resized = adapter.optimize_ident(line(problem="lorenz96", system_params='{"states":8}'), "k")
        self.assertEqual((resized["problem"], resized["states"]), ("lorenz96", "8"))
        adaptive = adapter.optimize_ident(line(controller="default", dt=NAN, atol=1e-5, rtol=1e-5,
                                               newton_atol=1e-6, newton_rtol=1e-6), "k")
        self.assertEqual(adaptive["stepping"],
                         "dt=;dt_min=;dt_max=;atol=1e-05;rtol=1e-05;newton_atol=1e-06;newton_rtol=1e-06")

    def test_a_kernel_record_serves_every_n_and_a_solve_record_its_own(self):
        result = FakeResult(FakeLaunch(256, 3), {"state_location": "shared", "blocksize": 256})
        adapter.record_optimized(line(n=8), "k", result, 71680)
        for n in (8, 32, 131072):
            tuned = adapter.load_optimized(line(n=n), "k")
            self.assertEqual(tuned["settings"], {"state_location": "shared", "blocksize": 256})
            self.assertEqual(tuned["resident_blocks"], 3)
        self.assertIsNone(adapter.load_optimized(line(n=8, optimize="solve"), "k"))
        adapter.record_optimized(line(n=8, optimize="solve"), "k",
                                 FakeResult(FakeLaunch(64, None), {"blocksize": 64}), 8)
        self.assertEqual(adapter.load_optimized(line(n=8, optimize="solve"), "k")["settings"], {"blocksize": 64})
        self.assertIsNone(adapter.load_optimized(line(n=32, optimize="solve"), "k"))
        self.assertEqual(adapter.load_optimized(line(n=32), "k")["resident_blocks"], 3)
        rows = adapter.optimize_rows("cubie", "k")
        self.assertEqual([(r["per"], r["n"]) for r in rows], [("kernel", "71680"), ("solve", "8")])
        self.assertEqual(adapter.optimize_rows("cubie_mlir", "k"), [])

    def test_every_stepping_value_precision_and_package_separate_kernels(self):
        result = FakeResult(FakeLaunch(64, None), {"blocksize": 64})
        base = line(algorithm="kvaerno3", controller="default", dt=NAN, atol=1e-3, rtol=1e-3,
                    newton_atol=1e-6, newton_rtol=1e-6)
        adapter.record_optimized(base, "k", result, 8)
        self.assertIsNotNone(adapter.load_optimized(base, "k"))
        self.assertIsNotNone(adapter.load_optimized(dict(base, n=131072, duration=7.0), "k"))
        for other in (dict(atol=1e-4), dict(rtol=1e-4), dict(newton_atol=1e-7), dict(dt_min=1e-9),
                      dict(dt=2.0 ** -10), dict(precision="float64"), dict(package="cubie_mlir"),
                      dict(controller="pi"), dict(gains='{"integral_gain":0.3}'), dict(algorithm="kvaerno5"),
                      dict(problem="lorenz96", system_params='{"states":4}')):
            self.assertIsNone(adapter.load_optimized(dict(base, **other), "k"), other)

    def test_rerecording_replaces_by_identity(self):
        first = FakeResult(FakeLaunch(64, 2), {"blocksize": 64})
        second = FakeResult(FakeLaunch(128, None), {"blocksize": 128})
        resized = line(problem="lorenz96", system_params='{"states":8}')
        adapter.record_optimized(resized, "k", first, 8)
        adapter.record_optimized(resized, "k", second, 8)
        tuned = adapter.load_optimized(resized, "k")
        self.assertEqual(tuned["settings"]["blocksize"], 128)
        self.assertIsNone(tuned["resident_blocks"])
        self.assertIsNone(adapter.load_optimized(dict(resized, system_params='{"states":16}'), "k"))
        with open(adapter.optimize_path("cubie", "k")) as handle:
            self.assertEqual(sum(1 for _ in handle) - 1, 1)

    def test_a_source_narrows_the_rows_served(self):
        result = FakeResult(FakeLaunch(64, 2), {"blocksize": 64})
        adapter.record_optimized(line(), "k", result, 8, source="S")
        self.assertEqual(adapter.load_optimized(line(), "k", source="S")["resident_blocks"], 2)
        self.assertIsNone(adapter.load_optimized(line(), "k", source="T"))
        self.assertEqual(adapter.load_optimized(line(), "k")["resident_blocks"], 2)
        rows = adapter.optimize_rows("cubie", "k")
        self.assertEqual(adapter.find_optimized(rows, adapter.optimize_ident(line(), "k"))["source"], "S")
        self.assertIsNone(adapter.find_optimized(rows, adapter.optimize_ident(line(dt=0.5), "k")))

    def test_a_timeout_row_replaces_the_lines_record_with_no_settings(self):
        adapter.record_optimized(line(n=64, optimize="solve"), "k",
                                 FakeResult(FakeLaunch(64, 2), {"blocksize": 64}), 64, source="S")
        row = adapter.record_optimize_timeout(line(n=64, optimize="solve"), "k")
        self.assertEqual((row["label"], row["n"], row["per"], row["settings"], row["source"]),
                         ("timeout", "64", "solve", "", ""))
        rows = adapter.optimize_rows("cubie", "k")
        self.assertEqual(len(rows), 1)
        self.assertEqual(adapter.load_optimized(line(n=64, optimize="solve"), "k"),
                         {"settings": {}, "resident_blocks": None})
        self.assertIsNone(adapter.load_optimized(line(n=64, optimize="solve"), "k", source="S"))

    def test_clear_narrows_by_algorithm_and_problem(self):
        result = FakeResult(FakeLaunch(64, None), {"blocksize": 64})
        for algorithm, problem in (("tsit5", "lorenz"), ("euler", "lorenz"),
                                   ("tsit5", "pollu")):
            adapter.record_optimized(line(algorithm=algorithm, problem=problem), "k", result, 8)
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
        adapter.record_optimized(line(), "k", result, 8)
        tuned = adapter.load_optimized(line(), "k")
        self.assertIs(tuned["settings"]["unroll_other_small"],
                      UnrollChoice.ROLLED)

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
        row = adapter.optimize_point(solver, line(n=8, optimize="solve", duration=7.0),
                                     np.zeros((3, 512)), np.zeros((1, 512)), "k")
        self.assertEqual(solver.seen, ((3, 512), 7.0, False))
        self.assertEqual((row["n"], row["per"], row["duration"]), ("512", "solve", "7"))
        self.assertEqual(float(row["best_ms"]), 2.5)
        forced = adapter.optimize_point(solver, line(n=8), np.zeros((3, 512)), np.zeros((1, 512)), "k",
                                        root=os.path.join(self.tmp, "elsewhere"), force=True, source="S",
                                        duration=0.25)
        self.assertEqual(solver.seen, ((3, 512), 0.25, True))
        self.assertEqual((forced["n"], forced["per"], forced["source"], forced["duration"]),
                         ("512", "kernel", "S", "0.25"))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "elsewhere", "key=k", "package=cubie",
                                                    "optimize.csv")))

    def test_optimize_batch_fills_the_waves_of_the_compiled_kernel(self):
        class Kernel:
            kernel = "dispatcher"
            compile_settings = type("S", (), {"blocksize": 32})()
            single_integrator = type("I", (), {"threads_per_step": 1})()

            def launch_geometry(self, blocksize):
                return int(blocksize), 16512

        class Solver:
            kernel = Kernel()

            def __init__(self):
                self.compiled = []

            def compile(self, initial_values, parameters, duration):
                self.compiled.append((initial_values, duration))

        seen = []
        for name, fake in (("_occupancy", lambda kernel, blocksize, dynamic: seen.append((blocksize, dynamic)) or 5),
                           ("_multiprocessors", lambda: 56)):
            self.addCleanup(setattr, adapter, name, getattr(adapter, name))
            setattr(adapter, name, fake)
        solver = Solver()
        self.assertEqual(adapter.optimize_batch(solver, "i", "p", 1.0, waves=5), 5 * 56 * 5 * 32)
        self.assertEqual(solver.compiled, [("i", 1.0)])
        self.assertEqual(seen, [(32, 16512)])
        self.assertEqual(adapter.optimize_batch(solver, "i", "p", 1.0, waves=2), 2 * 56 * 5 * 32)
        # Two threads per run halve the runs a block holds.
        Kernel.single_integrator = type("I", (), {"threads_per_step": 2})()
        self.assertEqual(adapter.optimize_batch(solver, "i", "p", 1.0, waves=5), 5 * 56 * 5 * 16)

    def test_optimize_duration_ramps_probes_to_the_target(self):
        class ProbeSolver:
            """A device solve of a probe takes ms_per_unit milliseconds per unit of duration."""

            device_initial_values = "resident inits"
            device_parameters = "resident params"

            def __init__(self, ms_per_unit):
                self.ms_per_unit = ms_per_unit
                self.solves = []

            def solve(self, initial_values, parameters, duration, on_device=False):
                self.solves.append((duration, on_device))
                ticks.append(ticks[-1] + (duration * self.ms_per_unit / 1000.0 if on_device else 0.0))
                return type("R", (), {"stream": type("S", (), {"synchronize": staticmethod(lambda: None)})()})()

        ticks = [0.0]
        self.addCleanup(setattr, adapter.timeit, "default_timer", adapter.timeit.default_timer)
        adapter.timeit.default_timer = lambda: ticks[-1]
        # 1/100 of the duration takes 5 ms, 1/10 takes 50 ms: the 1/10 probe scales to 20 ms.
        solver = ProbeSolver(500.0)
        self.assertAlmostEqual(adapter.optimize_duration(solver, "i", "p", 1.0, target_ms=20.0), 0.1 * 20.0 / 50.0)
        self.assertEqual(solver.solves, [(0.01, False), (0.01, True), (0.01, True), (0.1, True)])
        # 1/100 already takes 40 ms: it alone is timed and the duration stays at the 1/100 floor.
        solver = ProbeSolver(4000.0)
        self.assertEqual(adapter.optimize_duration(solver, "i", "p", 1.0, target_ms=20.0), 0.01)
        self.assertEqual(solver.solves, [(0.01, False), (0.01, True), (0.01, True)])
        self.assertEqual(adapter.optimize_duration(ProbeSolver(4000.0), "i", "p", 3.0, target_ms=20.0), 0.03)
        # A fast kernel scales past the duration and is clamped to it.
        solver = ProbeSolver(1.0)
        self.assertEqual(adapter.optimize_duration(solver, "i", "p", 2.0, target_ms=20.0), 2.0)
        self.assertEqual(solver.solves, [(0.02, False), (0.02, True), (0.02, True), (0.2, True)])
        # A zero-time probe yields the duration.
        self.assertEqual(adapter.optimize_duration(ProbeSolver(0.0), "i", "p", 3.0, target_ms=20.0), 3.0)


if __name__ == "__main__":
    unittest.main()
