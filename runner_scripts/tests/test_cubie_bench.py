"""The cubie adapter: Solver keywords from a trial, gains applied after construction, a build's stepping updates and resident inputs, finals with status codes, the optimize rows, cold cache roots, and the version string."""

import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cubie_bench  # noqa: E402
import cubie_adapter  # noqa: E402
import store  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
NAN = float("nan")
NAMES = ("x", "y", "z")


def trial(n=8, transfers=("both", "none"), finals=False, cold=False, optimize=None, **overrides):
    """A trial record: lorenz, fixed tsit5 at dt 2^-10, cubie."""
    fields = dict(problem="lorenz", system_params="{}", duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0, n=n,
                  grid_dtype="float32", algorithm="tsit5", controller="fixed", dt=2.0 ** -10,
                  dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains="{}", newton_atol=NAN,
                  newton_rtol=NAN, package="cubie")
    fields.update(overrides)
    fields["trial_id"] = store.trial_id(fields)
    fields.update(finals=finals, transfers=list(transfers), cold=cold, optimize=optimize,
                  watchdog_s=120.0, sets=[])
    return fields


def adaptive(**overrides):
    fields = dict(controller="default", dt=2.0 ** -10, atol=1e-5, rtol=1e-5)
    fields.update(overrides)
    return fields


class FakeIndices:
    def __init__(self, names):
        self.index_map = list(names)


class FakeSystem:
    """Three named states and no observables."""

    def __init__(self):
        self.indices = type("I", (), {})()
        self.indices.states = FakeIndices(NAMES)
        self.sizes = type("S", (), {"observables": 0})()


class FakeDeviceArray:
    def __init__(self, host):
        self.shape = host.shape


class FakeStream:
    def synchronize(self):
        pass


class FakeDeviceResult:
    def __init__(self):
        self.stream = FakeStream()


class FakeSolution:
    """state (time, variables, runs) with one save; status codes per run."""

    def __init__(self, n, codes=None):
        self.state = np.zeros((1, len(NAMES), n), dtype=np.float32)
        self.state[0] = np.arange(len(NAMES) * n, dtype=np.float32).reshape(len(NAMES), n)
        self.status_codes = np.zeros(n, dtype=np.int32) if codes is None else np.asarray(codes, np.int32)


class FakeLaunch:
    def __init__(self):
        self.blocksize, self.resident_blocks, self.best_ms = 128, 2, 1.5
        self.label = "state=shared @bs128 x2"


class FakeOptimizeResult:
    def __init__(self):
        self.best = FakeLaunch()
        self.applied_settings = {"blocksize": 128, "state_location": "shared"}


class FakeSolver:
    """Records its construction keywords, updates, compiles, optimizes and solves; a device solve needs the resident inputs of the last host solve."""

    made = []

    def __init__(self, system, **kwargs):
        self.system = system
        self.kwargs = kwargs
        self.updates = []
        self.calls = []
        self.compiled = []
        self.optimized = []
        self.resident = None
        self.closed = False
        self.codes = None
        FakeSolver.made.append(self)

    def update(self, updates):
        self.updates.append(dict(updates))

    def build_grid(self, initial_values, parameters):
        values = next(iter(parameters.values()))
        n = len(values)
        initials = np.zeros((len(initial_values), n), np.float32)
        params = np.asarray(values, np.float32).reshape(1, n)
        return initials, params

    def compile(self, initial_values, parameters, duration):
        self.compiled.append((initial_values.shape[1], duration))

    def optimize(self, initial_values, parameters, duration, verbose, force=False):
        self.optimized.append((initial_values.shape[1], duration, force))
        return FakeOptimizeResult()

    def solve(self, initial_values, parameters, duration, on_device=False):
        n = initial_values.shape[1]
        self.calls.append((n, on_device))
        if on_device:
            if self.resident is None or initial_values is not self.resident[0]:
                raise AssertionError("device solve without the resident inputs")
            return FakeDeviceResult()
        self.resident = (FakeDeviceArray(initial_values), FakeDeviceArray(parameters))
        return FakeSolution(n, self.codes)

    @property
    def device_initial_values(self):
        return self.resident[0]

    @property
    def device_parameters(self):
        return self.resident[1]

    def close(self):
        self.closed = True


class AdapterCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cubie_bench_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "data")
        self.saved = cubie_adapter.build_system
        cubie_adapter.build_system = self.fake_build_system
        self.addCleanup(setattr, cubie_adapter, "build_system", self.saved)
        self.built = []
        FakeSolver.made = []
        self.adapter = cubie_bench.CubieAdapter("cubie", KEY, self.root, solver_class=FakeSolver)

    def fake_build_system(self, problem, package, precision=None, states=None):
        self.built.append((problem.name, problem["states"], package, precision, states))
        return FakeSystem(), {name: 0.0 for name in NAMES}

    def values(self, n):
        import grid
        return grid.grid(trial(n))


class KeywordTests(unittest.TestCase):
    def test_a_fixed_stepping_sets_the_controller_and_the_step(self):
        self.assertEqual(cubie_bench.stepping_kwargs(trial()),
                         {"step_controller": "fixed", "dt": 2.0 ** -10})
        self.assertEqual(cubie_bench.stepping_kwargs(trial(dt=NAN)), {"step_controller": "fixed"})

    def test_an_adaptive_stepping_sets_tolerances_and_only_finite_pins(self):
        self.assertEqual(cubie_bench.stepping_kwargs(trial(**adaptive())),
                         {"atol": 1e-5, "rtol": 1e-5, "dt": 2.0 ** -10})
        self.assertEqual(cubie_bench.stepping_kwargs(trial(**adaptive(dt=NAN, dt_min=1e-7, dt_max=0.5))),
                         {"atol": 1e-5, "rtol": 1e-5, "dt_min": 1e-7, "dt_max": 0.5})
        named = cubie_bench.stepping_kwargs(trial(**adaptive(controller="pi")))
        self.assertEqual(named["step_controller"], "pi")
        # Newton tolerances reach the solver only when the trial carries them.
        newton = cubie_bench.stepping_kwargs(trial(algorithm="kvaerno3", newton_atol=1e-6, newton_rtol=1e-6))
        self.assertEqual(newton, {"step_controller": "fixed", "dt": 2.0 ** -10,
                                  "newton_atol": 1e-6, "newton_rtol": 1e-6})
        self.assertNotIn("newton_atol", cubie_bench.stepping_kwargs(trial(algorithm="kvaerno3")))

    def test_make_solver_passes_the_algorithm_and_applies_gains_after_construction(self):
        FakeSolver.made = []
        gains = '{"integral_gain":0.3,"proportional_gain":0.4,"safety":0.9}'
        solver = cubie_bench.make_solver(FakeSystem(), trial(**adaptive(controller="pi", gains=gains)),
                                         solver_class=FakeSolver)
        self.assertEqual(solver.kwargs["algorithm"], "tsit5")
        self.assertEqual(solver.kwargs["save_every"], 1.0)
        self.assertEqual(solver.kwargs["output_types"], ["state"])
        self.assertIsNone(solver.kwargs["time_logging_level"])
        self.assertEqual(solver.kwargs["step_controller"], "pi")
        self.assertEqual(solver.updates, [{"integral_gain": 0.3, "proportional_gain": 0.4, "safety": 0.9}])
        plain = cubie_bench.make_solver(FakeSystem(), trial(), solver_class=FakeSolver)
        self.assertEqual(plain.updates, [])

    def test_optimize_setting_is_the_lines_step_or_tolerance(self):
        self.assertEqual(cubie_bench.optimize_setting(trial()), ("fixed", 2.0 ** -10))
        self.assertEqual(cubie_bench.optimize_setting(trial(dt=0.5)), ("fixed", 0.5))
        self.assertEqual(cubie_bench.optimize_setting(trial(**adaptive())), ("adaptive", 1e-5))

    def test_controllers_version_and_states(self):
        import importlib.metadata
        self.assertEqual(cubie_bench.CONTROLLERS, ("fixed", "default", "i", "pi", "pid", "gustafsson"))
        version = importlib.metadata.version("cubie")
        self.assertEqual(cubie_bench.CubieAdapter("cubie", KEY, "data").version(), version + "+numba-cuda")
        self.assertEqual(cubie_bench.CubieAdapter("cubie_mlir", KEY, "data").version(), version + "+mlir")
        adapter = cubie_bench.CubieAdapter("cubie", KEY, "data")
        self.assertEqual(adapter.states(trial()), 3)
        self.assertEqual(adapter.states(trial(problem="lorenz96", system_params='{"states":8}', parameter="F",
                                              grid_max=16.0)), 8)
        self.assertEqual(adapter.states(trial(problem="pollu", parameter="k1", grid_scale="log",
                                              grid_min=3.5e-2, grid_max=3.5, duration=60.0)), 20)


class BuildTests(AdapterCase):
    def test_build_sizes_the_system_from_system_params(self):
        leg = self.adapter.build(trial(problem="lorenz96", system_params='{"states":8}', parameter="F",
                                           grid_max=16.0))
        self.assertEqual(self.built, [("lorenz96", 8, "cubie", np.float32, 8)])
        self.assertEqual(leg.states, 8)
        leg.close()
        leg = self.adapter.build(trial(precision="float64"))
        self.assertEqual(self.built[-1], ("lorenz", 3, "cubie", np.float64, None))
        self.assertEqual(leg.states, 3)
        self.assertEqual(leg.precision, np.float64)
        leg.close()
        self.assertTrue(all(s.closed for s in FakeSolver.made))

    def test_compile_builds_the_grid_at_the_trials_n(self):
        leg = self.adapter.build(trial(n=8))
        self.adapter.compile(leg, trial(n=8, kind="warm"), self.values(8))
        self.assertEqual(leg.solver.compiled, [(8, 1.0)])
        leg.close()

    def test_a_changed_stepping_updates_the_solver_and_drops_the_resident_inputs(self):
        first = trial(dt=2.0 ** -10, axis="dt")
        leg = self.adapter.build(first)
        self.adapter.solve(leg, first, self.values(8), "both")
        self.adapter.solve(leg, first, self.values(8), "none")
        self.assertEqual(leg.solver.updates, [])
        self.assertEqual(leg.solver.calls, [(8, False), (8, True)])
        second = trial(dt=2.0 ** -11, axis="dt", ordinal=1)
        self.adapter.solve(leg, second, self.values(8), "none")
        self.assertEqual(leg.solver.updates, [{"step_controller": "fixed", "dt": 2.0 ** -11}])
        # The update dropped the resident inputs, so the device solve uploaded through a host solve.
        self.assertEqual(leg.solver.calls[2:], [(8, False), (8, True)])
        # A changed controller or gains rebuilds the solver so no earlier gain lingers.
        old = leg.solver
        third = trial(**adaptive(controller="pi", gains='{"integral_gain":0.3}'), axis="tol")
        self.adapter.solve(leg, third, self.values(8), "both")
        self.assertTrue(old.closed)
        self.assertIsNot(leg.solver, old)
        self.assertEqual(len(FakeSolver.made), 2)
        self.assertEqual(leg.solver.kwargs["step_controller"], "pi")
        self.assertEqual(leg.solver.kwargs["atol"], 1e-5)
        self.assertEqual(leg.solver.updates, [{"integral_gain": 0.3}])
        self.assertEqual(leg.solver.calls, [(8, False)])
        fourth = trial(**adaptive(controller="pi", gains='{"integral_gain":0.3}', atol=1e-6, rtol=1e-6),
                       axis="tol", ordinal=1)
        self.adapter.solve(leg, fourth, self.values(8), "both")
        self.assertEqual(len(FakeSolver.made), 2)
        self.assertEqual(leg.solver.updates[-1], {"atol": 1e-6, "rtol": 1e-6, "dt": 2.0 ** -10,
                                                  "step_controller": "pi"})
        leg.close()

    def test_device_solves_reuse_the_host_solves_inputs_per_n(self):
        leg = self.adapter.build(trial(n=8))
        self.adapter.solve(leg, trial(n=8), self.values(8), "both")
        self.adapter.solve(leg, trial(n=8), self.values(8), "none")
        self.adapter.solve(leg, trial(n=8), self.values(8), "none")
        self.adapter.solve(leg, trial(n=32, ordinal=1), self.values(32), "none")
        self.assertEqual(leg.solver.calls, [(8, False), (8, True), (8, True), (32, False), (32, True)])
        self.assertEqual(leg.grid_n, 32)
        leg.close()

    def test_finals_carry_the_status_flags_and_the_duration_of_clean_runs(self):
        leg = self.adapter.build(trial(n=4))
        leg.solver.codes = [0, 8, 0, 2 | 256]
        result = self.adapter.solve(leg, trial(n=4), self.values(4), "both")
        finals, t_final, retcode = self.adapter.finals(leg, result)
        self.assertEqual(finals.shape, (4, 3))
        self.assertEqual(finals.dtype, np.float32)
        self.assertEqual(list(finals[:, 0]), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(list(finals[:, 2]), [8.0, 9.0, 10.0, 11.0])
        self.assertEqual(retcode, ["", "STEP_TOO_SMALL", "", "MAX_NEWTON_ITERATIONS_EXCEEDED|NEWTON_DIVERGENCE"])
        self.assertEqual(t_final[0], 1.0)
        self.assertTrue(math.isnan(t_final[1]))
        self.assertEqual(store.errored_pct(finals, t_final, retcode, 1.0), 50.0)
        # A device solve hands back the host result its inputs came from.
        device = self.adapter.solve(leg, trial(n=4), self.values(4), "none")
        self.assertIs(device, result)
        leg.close()

    def test_optimize_records_the_winner_at_the_lines_step_or_tolerance(self):
        line = trial(n=8, optimize=64)
        leg = self.adapter.build(trial(n=8))
        self.adapter.optimize(leg, line, self.values(64))
        self.assertEqual(leg.solver.optimized, [(64, 1.0, True)])
        tuned = cubie_adapter.load_optimized("cubie", KEY, "lorenz", "tsit5", "fixed", 2.0 ** -10,
                                             root=self.root, controller="fixed", gains="{}")
        self.assertEqual(tuned["settings"], {"blocksize": 128, "state_location": "shared"})
        self.assertEqual(tuned["resident_blocks"], 2)
        self.assertIsNone(cubie_adapter.load_optimized("cubie", KEY, "lorenz", "tsit5", "fixed", 2.0 ** -13,
                                                       root=self.root, controller="fixed", gains="{}"))
        leg.close()
        tol_line = trial(n=8, optimize=64, **adaptive(atol=1e-4, rtol=1e-4))
        leg = self.adapter.build(trial(**adaptive(atol=1e-4, rtol=1e-4)))
        self.adapter.optimize(leg, tol_line, self.values(64))
        self.assertIsNotNone(cubie_adapter.load_optimized("cubie", KEY, "lorenz", "tsit5", "adaptive", 1e-4,
                                                          root=self.root, controller="default", gains="{}"))
        self.assertIsNone(cubie_adapter.load_optimized("cubie", KEY, "lorenz", "tsit5", "adaptive", 1e-5,
                                                       root=self.root, controller="default", gains="{}"))
        # The controller and gains keep one algorithm's builds apart in the record.
        pi_line = trial(n=8, optimize=64,
                        **adaptive(atol=1e-4, rtol=1e-4, controller="pi", gains='{"integral_gain":0.3}'))
        self.adapter.optimize(leg, pi_line, self.values(64))
        self.assertIsNotNone(cubie_adapter.load_optimized("cubie", KEY, "lorenz", "tsit5", "adaptive", 1e-4,
                                                          root=self.root, controller="default", gains="{}"))
        self.assertIsNotNone(cubie_adapter.load_optimized(
            "cubie", KEY, "lorenz", "tsit5", "adaptive", 1e-4, root=self.root, controller="pi",
            gains='{"integral_gain":0.3}'))
        self.assertIsNone(cubie_adapter.load_optimized("cubie", KEY, "lorenz", "tsit5", "adaptive", 1e-4,
                                                       root=self.root, controller="pi", gains="{}"))
        path = os.path.join(self.root, "key=" + KEY, "package=cubie", "optimize.csv")
        with open(path) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(len(lines) - 1, 3)
        self.assertTrue(lines[0].startswith("package,key,problem,algorithm,mode,controller,gains,"))
        leg.close()

    def test_a_point_recorded_from_the_same_source_is_applied_and_compiled_not_optimized(self):
        line = trial(n=8, optimize=64)
        leg = self.adapter.build(trial(n=8))
        self.adapter.optimize(leg, line, self.values(64))
        self.adapter.optimize(leg, line, self.values(64))
        self.assertEqual(len(leg.solver.optimized), 1)
        self.assertEqual(leg.solver.updates[-1], {"blocksize": 128, "state_location": "shared"})
        self.assertEqual(leg.solver.compiled[-1], (64, 1.0))
        path = os.path.join(self.root, "key=" + KEY, "package=cubie", "optimize.csv")
        source = cubie_adapter.source_hash(leg.solver)
        with open(path) as handle:
            text = handle.read()
        self.assertIn(source, text)
        # A row from another source is replaced by a fresh optimize.
        with open(path, "w") as handle:
            handle.write(text.replace(source, "0" * 16))
        self.adapter.optimize(leg, line, self.values(64))
        self.assertEqual(len(leg.solver.optimized), 2)
        with open(path) as handle:
            self.assertEqual(handle.read().count(source), 1)
        leg.close()

    def test_a_cold_build_uses_a_fresh_cache_root_and_restores_it(self):
        from cubie.cache_root import get_cache_root_override
        before = get_cache_root_override()
        leg = self.adapter.build(trial(cold=True), cold=True)
        override = get_cache_root_override()
        self.assertIsNotNone(override)
        self.assertNotEqual(override, before)
        self.assertTrue(os.path.isdir(str(override)))
        self.assertEqual(str(override), leg.cache_dir)
        leg.close()
        self.assertEqual(get_cache_root_override(), before)
        self.assertFalse(os.path.isdir(str(override)))
        warm = self.adapter.build(trial(), cold=False)
        self.assertIsNone(warm.cache_dir)
        self.assertEqual(get_cache_root_override(), before)
        warm.close()

    def test_a_failed_cold_build_restores_the_cache_root(self):
        from cubie.cache_root import get_cache_root_override

        def broken(*args, **kwargs):
            raise RuntimeError("codegen failed")

        cubie_adapter.build_system = broken
        before = get_cache_root_override()
        with self.assertRaises(RuntimeError):
            self.adapter.build(trial(cold=True), cold=True)
        self.assertEqual(get_cache_root_override(), before)


if __name__ == "__main__":
    unittest.main()
