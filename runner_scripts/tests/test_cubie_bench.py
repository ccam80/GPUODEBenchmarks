"""The cubie adapter: Solver keywords from a trial, gains applied after construction, a build's stepping updates and resident inputs, finals with status codes, the optimize rows, a compile under the kernel's record, cold cache roots, the precompile worker, and the version string."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import abandon  # noqa: E402
import cubie_bench  # noqa: E402
import cubie_adapter  # noqa: E402
import cubie_precompile  # noqa: E402
import store  # noqa: E402
import trials as trials_mod  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
NAN = float("nan")
NAMES = ("x", "y", "z")


def trial(n=8, transfers=("both", "none"), finals=False, cold=False, optimize=False, **overrides):
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


def spec(n=8, **overrides):
    """A run spec for trials.build_trials: the trial fields plus the expansion's own."""
    fields = trial(n=n, **overrides)
    fields.update(build="warm", optimize=False, transfers=["both", "none"], finals=False, watchdog_s=120.0)
    return fields


class FakeIndices:
    def __init__(self, names):
        self.index_map = list(names)


class FakeSystem:
    """Three named states and no observables; `broken` makes every compile of it raise."""

    def __init__(self, broken=False):
        self.broken = broken
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
    """state (time, variables, runs) with one save, or `saves` of them; status codes per run."""

    def __init__(self, n, codes=None, saves=1):
        self.state = np.zeros((saves, len(NAMES), n), dtype=np.float32)
        self.state[:] = np.arange(len(NAMES) * n, dtype=np.float32).reshape(len(NAMES), n)
        self.state += np.arange(saves, dtype=np.float32)[:, None, None]
        self.status_codes = np.zeros(n, dtype=np.int32) if codes is None else np.asarray(codes, np.int32)


class FakeLaunch:
    def __init__(self):
        self.blocksize, self.resident_blocks, self.best_ms = 128, 2, 1.5
        self.label = "state=shared @bs128 x2"


class FakeOptimizeResult:
    """The winner over the runs and duration the optimize timed."""

    def __init__(self, runs, duration):
        self.best = FakeLaunch()
        self.applied_settings = {"blocksize": 128, "state_location": "shared"}
        self.runs = runs
        self.duration = duration


# The batch and duration cubie's auto_size settles on: five waves of the kernel, timed over 1/25 of the duration.
SIZED_RUNS = 71680
SIZED_FRACTION = 0.04


class FakeSolver:
    """Records its construction keywords, updates, compiles, optimizes and solves; a device solve needs the resident inputs of the last host solve."""

    made = []

    def __init__(self, system, **kwargs):
        self.system = system
        self.kwargs = kwargs
        self.updates = []
        self.calls = []
        self.compiled = []
        self.compile_kwargs = []
        self.optimized = []
        self.resident = None
        self.closed = False
        self.codes = None
        FakeSolver.made.append(self)

    def update(self, updates):
        self.updates.append(dict(updates))

    def build_grid(self, initial_values, parameters):
        self.grids = getattr(self, "grids", []) + [dict(parameters)]
        values = next(iter(parameters.values()))
        n = len(values)
        initials = np.zeros((len(initial_values), n), np.float32)
        params = np.stack([np.asarray(v, np.float32) for v in parameters.values()])
        return initials, params

    def compile(self, **kwargs):
        self.compiled.append(dict(kwargs))
        if getattr(self.system, "broken", False):
            raise RuntimeError("ptxas failed")

    def optimize(self, initial_values, parameters, duration, verbose, force=False, auto_size=False):
        self.optimized.append((initial_values.shape[1], duration, force, auto_size))
        if not auto_size:
            raise AssertionError("optimize without auto_size")
        return FakeOptimizeResult(SIZED_RUNS, duration * SIZED_FRACTION)

    def solve(self, initial_values, parameters, duration, on_device=False, nan_error_trajectories=True):
        n = initial_values.shape[1]
        self.calls.append((n, on_device))
        self.nan_error_trajectories = nan_error_trajectories
        if on_device:
            if self.resident is None or initial_values is not self.resident[0]:
                raise AssertionError("device solve without the resident inputs")
            return FakeDeviceResult()
        self.resident = (FakeDeviceArray(initial_values), FakeDeviceArray(parameters))
        if "save_every" in self.kwargs:
            return FakeSolution(n, self.codes, saves=int(round(duration / self.kwargs["save_every"])) + 1)
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


class GridTests(AdapterCase):
    def test_a_build_passes_the_problem_parameter_arrays_of_the_grid(self):
        import fabbri
        import grid
        leg = self.adapter.build(trial(n=4))
        self.adapter.solve(leg, trial(n=4), self.values(4), "both")
        self.assertEqual(list(leg.solver.grids[-1]), ["rho"])
        np.testing.assert_array_equal(leg.solver.grids[-1]["rho"], np.float32([0, 7, 14, 21]))
        leg.close()
        record = trial(n=4, problem="fabbri_linder", duration=2.0, parameter="ach_iso", grid_max=131071.0,
                       algorithm="euler", dt=2.0 * 2.0 ** -15)
        leg = self.adapter.build(record)
        self.assertEqual(leg.states, 35)
        self.assertEqual(self.built[-1][:2], ("fabbri_linder", 35))
        values = grid.grid(record)
        self.adapter.solve(leg, record, values, "both")
        passed = leg.solver.grids[-1]
        self.assertEqual(list(passed), [fabbri.ACH_PARAMETER, fabbri.ISO_PARAMETER])
        ach, iso = fabbri.inputs(values, np.float32)
        np.testing.assert_array_equal(passed[fabbri.ACH_PARAMETER], ach)
        np.testing.assert_array_equal(passed[fabbri.ISO_PARAMETER], iso)
        self.assertEqual(leg.grid_arrays[1].shape, (2, 4))
        leg.close()


class TraceTests(AdapterCase):
    def test_a_trace_runs_a_second_solver_that_saves_every_sample(self):
        from protocol import TRACE_EVERY_S, TRACE_SAMPLES, TRACE_SPAN_S
        leg = self.adapter.build(trial(n=4))
        record = trial(n=4)
        self.adapter.solve(leg, record, self.values(4), "both")
        states, retcode = self.adapter.trace(leg, record, self.values(4))
        self.assertEqual(states.shape, (4, TRACE_SAMPLES, 3))
        self.assertEqual(retcode, [""] * 4)
        tracer = leg.trace_solver
        self.assertIsNot(tracer, leg.solver)
        self.assertEqual(tracer.kwargs["save_every"], TRACE_EVERY_S)
        self.assertEqual({k: v for k, v in tracer.kwargs.items() if k != "save_every"}, leg.solver.kwargs)
        self.assertEqual(tracer.calls, [(4, False)])
        # The initial save is dropped; sample s of run r, variable v is v * n + r + s.
        self.assertEqual(states[1, 0, 2], 2 * 4 + 1 + 1)
        self.assertEqual(states[3, -1, 0], 3 + TRACE_SAMPLES)
        self.assertEqual(len(FakeSolver.made), 2)
        # A run with a failure status carries its flags beside states left as the solve wrote them.
        tracer.codes = [0, 8, 0, 2 | 256]
        failed, retcode = self.adapter.trace(leg, record, self.values(4))
        self.assertEqual(retcode, ["", "STEP_TOO_SMALL", "", "MAX_NEWTON_ITERATIONS_EXCEEDED|NEWTON_DIVERGENCE"])
        np.testing.assert_array_equal(failed, states)
        tracer.codes = None
        # A changed stepping drops the trace solver with the kernel; close closes both.
        self.adapter.solve(leg, trial(n=4, dt=2.0 ** -11), self.values(4), "both")
        self.assertIsNone(leg.trace_solver)
        self.assertTrue(tracer.closed)
        self.adapter.trace(leg, trial(n=4, dt=2.0 ** -11), self.values(4))
        second = leg.trace_solver
        leg.close()
        self.assertTrue(second.closed and leg.solver is None and leg.trace_solver is None)
        self.assertEqual(TRACE_SPAN_S, 1.0)


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
        self.assertNotIn("save_every", solver.kwargs)
        self.assertEqual(solver.kwargs["output_types"], ["state"])
        self.assertIsNone(solver.kwargs["time_logging_level"])
        self.assertEqual(solver.kwargs["step_controller"], "pi")
        self.assertEqual(solver.updates, [{"integral_gain": 0.3, "proportional_gain": 0.4, "safety": 0.9}])
        plain = cubie_bench.make_solver(FakeSystem(), trial(), solver_class=FakeSolver)
        self.assertEqual(plain.updates, [])

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


class PrecompileWorkerTests(AdapterCase):
    """cubie_precompile.Worker over kernel lines: one warm build per kernel, compiled in this process with its optimize candidates when the line optimizes, closed, and the progress file tallying the outcomes."""

    def lines(self):
        first = {}
        for trial in trials_mod.build_trials([
                dict(spec(n=8), set="t"), dict(spec(n=32), set="t"), dict(spec(n=8, dt=2.0 ** -12), set="t"),
                dict(spec(n=8, algorithm="euler"), set="t", optimize=True),
                dict(spec(n=8, **adaptive()), set="t"),
                dict(spec(n=8, problem="lorenz96", system_params='{"states":8}', parameter="F", grid_max=16.0), set="t",
                     optimize=True)]):
            first.setdefault(trials_mod.kernel_key(trial), trial)
        return list(first.values())

    def test_every_kernel_is_built_warm_and_compiled_with_candidates_where_it_optimizes(self):
        lines = self.lines()
        # One kernel per dt, shared across n: euler, tsit5 default, tsit5 fixed at two dt, lorenz96 tsit5 fixed.
        self.assertEqual(len(lines), 5)
        path = os.path.join(self.tmp, "cubie.jsonl.precompile0.progress")
        worker = cubie_precompile.Worker("cubie", KEY, self.root, lines, (0, 5), path, solver_class=FakeSolver)
        self.assertEqual(worker.run(), 0)
        self.assertEqual(len(FakeSolver.made), 5)
        # euler and lorenz96 optimize; the tsit5 kernels compile the default kernel alone.
        self.assertEqual([s.compiled for s in FakeSolver.made],
                         [[{"optimize_candidates": True, "max_parallel": 1}],
                          [{"optimize_candidates": False, "max_parallel": 1}],
                          [{"optimize_candidates": False, "max_parallel": 1}],
                          [{"optimize_candidates": False, "max_parallel": 1}],
                          [{"optimize_candidates": True, "max_parallel": 1}]])
        for solver in FakeSolver.made:
            self.assertEqual(solver.optimized, [])
            self.assertEqual(solver.calls, [])
            self.assertTrue(solver.closed)
        # Warm builds: the cache root is untouched, and the resized system was built at its states.
        from cubie.cache_root import get_cache_root_override
        self.assertIsNone(get_cache_root_override())
        self.assertIn(("lorenz96", 8, "cubie", np.float32, 8), self.built)
        with open(path) as handle:
            progress = json.load(handle)
        self.assertEqual(progress["compiled"], [0, 1, 2, 3, 4])
        self.assertEqual(progress["failed"], [])
        self.assertIsNone(progress["under_way"])

    def test_a_span_takes_its_slice_and_a_failed_compile_is_tallied(self):
        lines = self.lines()
        path = os.path.join(self.tmp, "cubie.jsonl.precompile3.progress")

        def fragile(problem, package, precision=None, states=None):
            if problem.name == "lorenz96":
                raise RuntimeError("codegen failed")
            return FakeSystem(), {name: 0.0 for name in NAMES}

        cubie_adapter.build_system = fragile
        worker = cubie_precompile.Worker("cubie", KEY, self.root, lines, (3, 5), path, solver_class=FakeSolver)
        self.assertEqual(worker.run(), 0)
        self.assertEqual(len(FakeSolver.made), 1)
        with open(path) as handle:
            progress = json.load(handle)
        self.assertEqual(progress["compiled"], [3])
        self.assertEqual(progress["failed"], [[4, "error: RuntimeError: codegen failed"]])
        self.assertIsNone(progress["under_way"])
        # A compile that raises after the build closes the solver and is tallied the same way.
        FakeSolver.made = []
        cubie_adapter.build_system = lambda *a, **k: (FakeSystem(broken=True), {name: 0.0 for name in NAMES})
        worker = cubie_precompile.Worker("cubie", KEY, self.root, lines, (0, 1), path, solver_class=FakeSolver)
        self.assertEqual(worker.run(), 0)
        self.assertTrue(FakeSolver.made[0].closed)
        with open(path) as handle:
            self.assertEqual(json.load(handle)["failed"], [[0, "error: RuntimeError: ptxas failed"]])

    def test_a_worker_past_its_memory_budget_stops_after_the_kernel_and_names_the_next(self):
        lines = self.lines()
        path = os.path.join(self.tmp, "cubie.jsonl.precompile0.progress")
        with mock.patch.object(cubie_precompile, "private_bytes", side_effect=[1 << 30, 8 << 30]):
            worker = cubie_precompile.Worker("cubie", KEY, self.root, lines, (0, 5), path, solver_class=FakeSolver,
                                             memory_bytes=6 << 30)
            self.assertEqual(worker.run(), 0)
        self.assertEqual(len(FakeSolver.made), 2)
        with open(path) as handle:
            progress = json.load(handle)
        self.assertEqual((progress["compiled"], progress["under_way"], progress["next"]), ([0, 1], None, 2))

    def test_a_kernel_whose_group_the_store_records_a_compile_timeout_of_is_skipped(self):
        lines = self.lines()
        # Lines 2 and 3 are the fixed tsit5 kernels; euler at 0 and the adaptive tsit5 at 1 still compile.
        self.assertEqual([(t["algorithm"], t["controller"]) for t in lines[:4]],
                         [("euler", "fixed"), ("tsit5", "default"), ("tsit5", "fixed"), ("tsit5", "fixed")])
        abandon.abandon_compile(store.Store(self.root), KEY, lines, lines[2])
        path = os.path.join(self.tmp, "cubie.jsonl.precompile0.progress")
        worker = cubie_precompile.Worker("cubie", KEY, self.root, lines, (0, 4), path, solver_class=FakeSolver)
        self.assertEqual(worker.run(), 0)
        with open(path) as handle:
            progress = json.load(handle)
        self.assertEqual((progress["compiled"], progress["skipped"]), ([0, 1], [2, 3]))
        self.assertEqual(len(FakeSolver.made), 2)

    def test_each_kernel_compiles_under_the_optimize_watchdog(self):
        budgets = []

        def recording(run, on_breach, budget_s=None):
            budgets.append(budget_s)
            return run()

        with mock.patch.object(cubie_precompile, "run_watchdogged", recording):
            lines = self.lines()
            path = os.path.join(self.tmp, "cubie.jsonl.precompile0.progress")
            cubie_precompile.Worker("cubie", KEY, self.root, lines, (0, 2), path, solver_class=FakeSolver).run()
        self.assertEqual(budgets, [cubie_precompile.OPTIMIZE_SECONDS] * 2)


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

    def test_compile_compiles_under_the_kernels_record(self):
        leg = self.adapter.build(trial(n=8))
        self.adapter.compile(leg, trial(n=8, kind="warm"), self.values(8))
        self.assertEqual(leg.solver.compiled, [{}])
        self.assertEqual(leg.solver.updates, [])
        # A recorded optimize is applied before the compile, so a cold build times the optimized kernel.
        self.adapter.optimize(leg, trial(n=64, optimize=True))
        leg.close()
        leg = self.adapter.build(trial(n=8, cold=True), cold=True)
        self.adapter.compile(leg, trial(n=8), self.values(8))
        self.assertEqual(leg.solver.updates, [{"blocksize": 128, "state_location": "shared"}])
        self.assertEqual(leg.solver.compiled, [{}])
        leg.close()
        # An overwriting run applies only its own records.
        with mock.patch.dict(os.environ, {store.RUN_ENV: "other", store.OVERWRITE_ENV: "1"}):
            leg = self.adapter.build(trial(n=8))
            self.adapter.compile(leg, trial(n=8), self.values(8))
            self.assertEqual(leg.solver.updates, [])
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

    def test_solves_keep_the_states_of_failed_runs(self):
        leg = self.adapter.build(trial(n=8))
        self.adapter.solve(leg, trial(n=8), self.values(8), "both")
        self.assertIs(leg.solver.nan_error_trajectories, False)
        leg.close()

    def test_finals_carry_the_status_flags_and_the_duration(self):
        leg = self.adapter.build(trial(n=4))
        leg.solver.codes = [0, 8, 0, 2 | 256]
        result = self.adapter.solve(leg, trial(n=4), self.values(4), "both")
        finals, t_final, retcode = self.adapter.finals(leg, result)
        self.assertEqual(finals.shape, (4, 3))
        self.assertEqual(finals.dtype, np.float32)
        self.assertEqual(list(finals[:, 0]), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(list(finals[:, 2]), [8.0, 9.0, 10.0, 11.0])
        self.assertEqual(retcode, ["", "STEP_TOO_SMALL", "", "MAX_NEWTON_ITERATIONS_EXCEEDED|NEWTON_DIVERGENCE"])
        self.assertEqual(list(t_final), [1.0] * 4)
        self.assertEqual(store.errored_pct(finals, t_final, retcode, 1.0), 50.0)
        # A device solve hands back the host result its inputs came from.
        device = self.adapter.solve(leg, trial(n=4), self.values(4), "none")
        self.assertIs(device, result)
        leg.close()

    def test_optimize_records_the_winner_under_the_lines_kernel(self):
        line = trial(n=64, optimize=True)
        leg = self.adapter.build(trial(n=8))
        # Cubie sizes the batch and duration from the line's own grid; the runner compiles and solves nothing.
        self.assertEqual(self.adapter.optimize(leg, line), "state=shared @bs128 x2 on 71680 runs over 0.04")
        self.assertEqual(leg.solver.optimized, [(64, 1.0, True, True)])
        self.assertEqual(leg.solver.compiled, [])
        self.assertEqual(leg.solver.calls, [])
        self.assertEqual(leg.grid_n, 64)
        tuned = cubie_adapter.load_optimized(line, KEY, root=self.root)
        self.assertEqual(tuned["settings"], {"blocksize": 128, "state_location": "shared"})
        self.assertEqual(tuned["resident_blocks"], 2)
        # An explicit fixed-step build shares its optimize across dt.
        self.assertIsNotNone(cubie_adapter.load_optimized(trial(dt=2.0 ** -13, optimize=True), KEY, root=self.root))
        self.assertIsNone(cubie_adapter.load_optimized(trial(algorithm="euler", optimize=True), KEY, root=self.root))
        leg.close()
        tol_line = trial(n=8, optimize=True, **adaptive(atol=1e-4, rtol=1e-4))
        leg = self.adapter.build(trial(**adaptive(atol=1e-4, rtol=1e-4)))
        self.adapter.optimize(leg, tol_line)
        self.assertIsNotNone(cubie_adapter.load_optimized(tol_line, KEY, root=self.root))
        self.assertIsNone(cubie_adapter.load_optimized(dict(tol_line, atol=1e-5, rtol=1e-5), KEY, root=self.root))
        # The controller and gains keep one algorithm's builds apart in the record.
        pi_line = trial(n=8, optimize=True,
                        **adaptive(atol=1e-4, rtol=1e-4, controller="pi", gains='{"integral_gain":0.3}'))
        self.adapter.optimize(leg, pi_line)
        self.assertIsNotNone(cubie_adapter.load_optimized(tol_line, KEY, root=self.root))
        self.assertIsNotNone(cubie_adapter.load_optimized(pi_line, KEY, root=self.root))
        self.assertIsNone(cubie_adapter.load_optimized(dict(pi_line, gains="{}"), KEY, root=self.root))
        path = os.path.join(self.root, "key=" + KEY, "package=cubie", "optimize.csv")
        with open(path) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(len(lines) - 1, 3)
        self.assertEqual(lines[0], ",".join(cubie_adapter.OPTIMIZE_FIELDS))
        leg.close()

    def test_an_overwriting_run_optimizes_again_a_kernel_another_run_recorded(self):
        leg = self.adapter.build(trial(n=8))
        with mock.patch.dict(os.environ, {store.RUN_ENV: "old", store.OVERWRITE_ENV: "1"}):
            self.adapter.optimize(leg, trial(n=8, optimize=True))
        with mock.patch.dict(os.environ, {store.RUN_ENV: "new", store.OVERWRITE_ENV: "1"}):
            self.assertNotEqual(self.adapter.optimize(leg, trial(n=32, optimize=True)), "recorded")
            # The run's own record serves the kernel's later lines.
            self.assertEqual(self.adapter.optimize(leg, trial(n=64, optimize=True)), "recorded")
        self.assertEqual(len(leg.solver.optimized), 2)
        self.assertEqual([r["run"] for r in cubie_adapter.optimize_rows("cubie", KEY, self.root)], ["new"])
        # --resume and --no-overwrite apply a record whatever run wrote it.
        with mock.patch.dict(os.environ, {store.RUN_ENV: "later", store.OVERWRITE_ENV: ""}):
            self.assertEqual(self.adapter.optimize(leg, trial(n=128, optimize=True)), "recorded")
        self.assertEqual(len(leg.solver.optimized), 2)
        leg.close()

    def test_a_kernel_record_is_applied_and_compiled_whatever_source_recorded_it(self):
        line = trial(n=8, optimize=True)
        leg = self.adapter.build(trial(n=8))
        self.adapter.optimize(leg, line)
        self.assertEqual(self.adapter.optimize(leg, trial(n=64, optimize=True)), "recorded")
        self.assertEqual(len(leg.solver.optimized), 1)
        self.assertEqual(leg.solver.updates[-1], {"blocksize": 128, "state_location": "shared"})
        self.assertEqual(leg.solver.compiled[-1], {})
        path = os.path.join(self.root, "key=" + KEY, "package=cubie", "optimize.csv")
        with open(path) as handle:
            text = handle.read()
        self.assertEqual(text.splitlines()[0], ",".join(cubie_adapter.OPTIMIZE_FIELDS))
        self.assertNotIn("source", text.splitlines()[0])
        # A row an earlier suite recorded with a source hash of its own is applied the same way.
        lines = text.splitlines()
        with open(path, "w") as handle:
            handle.write(lines[0].replace("recorded_utc", "source,recorded_utc") + "\n")
            for row in lines[1:]:
                head, stamp = row.rsplit(",", 1)
                handle.write(head + "," + "0" * 16 + "," + stamp + "\n")
        self.assertEqual(self.adapter.optimize(leg, line), "recorded")
        self.assertEqual(len(leg.solver.optimized), 1)
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
