"""Cubie sweep legs: the device leg reuses the host leg's resident inputs, and each leg's failure lands in its own column."""

import math
import os
import sys
import tempfile
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cubie_julia_overlap"))

cubie = types.ModuleType("cubie")
cubie.cache_root = types.ModuleType("cubie.cache_root")
cubie.cache_root.set_cache_root = lambda path: None
sys.modules["cubie"] = cubie
sys.modules["cubie.cache_root"] = cubie.cache_root

import cubie_bench  # noqa: E402
import cubie_worker  # noqa: E402
from problems import get_problem  # noqa: E402


class FakeDeviceArray:
    def __init__(self, host):
        self.shape = host.shape
        self.dtype = host.dtype


class FakeStream:
    def __init__(self):
        self.synchronised = 0

    def synchronize(self):
        self.synchronised += 1


class FakeDeviceResult:
    def __init__(self):
        self.stream = FakeStream()


class FakeSolution:
    def __init__(self, n):
        self.finals = np.full((n, 3), float(n), dtype=np.float32)


class FakeSolver:
    """A host solve uploads into per-size resident buffers; a device solve must be given those buffers back."""

    def __init__(self, chunk_at=None, host_fail_at=None):
        self.calls = []
        self.chunk_at = chunk_at
        self.host_fail_at = host_fail_at
        self.closed = False
        self.resident = None
        self.chunked = False
        self.last_n = None
        self.device_results = []

    def solve(self, initial_values, parameters, duration, on_device=False):
        n = initial_values.shape[1]
        self.calls.append((n, on_device))
        if on_device:
            if self.resident is None or (
                    initial_values is not self.resident[0]
                    or parameters is not self.resident[1]):
                raise AssertionError("device leg was not given the "
                                     "solver's resident inputs")
            result = FakeDeviceResult()
            self.device_results.append(result)
            return result
        if isinstance(initial_values, FakeDeviceArray):
            raise AssertionError("host leg was given device arrays")
        if self.host_fail_at is not None and n == self.host_fail_at:
            raise MemoryError("allocating bytes")
        if self.last_n != n:
            self.resident = (FakeDeviceArray(initial_values),
                             FakeDeviceArray(parameters))
        self.last_n = n
        self.chunked = self.chunk_at is not None and n >= self.chunk_at
        return FakeSolution(n)

    def _resident_input(self, index):
        if self.chunked:
            raise ValueError("The device buffer holds one chunk of the "
                             "last run")
        return self.resident[index]

    @property
    def device_initial_values(self):
        return self._resident_input(0)

    @property
    def device_parameters(self):
        return self._resident_input(1)

    def build_grid(self, initial_values, parameters):
        n = len(next(iter(parameters.values())))
        return (np.zeros((len(initial_values), n), np.float32),
                np.zeros((1, n), np.float32))

    def close(self):
        self.closed = True


def grid(solver, n):
    return np.zeros((3, n), np.float32), np.zeros((1, n), np.float32)


def read_rows(analysis):
    """{n or states: [t_both, t_none, (build_s,) errored_pct]} from the CUBIE test store."""
    import results
    rows = {}
    for row in results.load(results.store_path("cubie", "test_key")):
        if row["analysis"] != analysis:
            continue
        key = int(row["states"] if analysis == "states" else row["n"])
        entry = rows.setdefault(key, {})
        entry[row["transfers"]] = float(row["min_ms"])
        entry["build_s"] = float(row["build_s"])
        entry["errored_pct"] = float(row["errored_pct"])
    out = {}
    for key, entry in rows.items():
        values = [entry.get("both", float("nan")),
                  entry.get("none", float("nan"))]
        if analysis == "states":
            values.append(entry["build_s"])
        values.append(entry["errored_pct"])
        out[key] = values
    return out


def sample_legs(analysis="times"):
    """{(n, transfers): attempt count} for the rows of the CUBIE test store that carry attempts."""
    import results
    counts = {}
    for row in results.load(results.store_path("cubie", "test_key")):
        if row["analysis"] != analysis:
            continue
        attempts = results.samples_of(row)
        if attempts:
            counts[(int(row["n"]), row["transfers"])] = len(attempts)
    return counts


class SweepCase(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)
        adapter = cubie_bench.adapter
        self.saved = (adapter.make_solver, adapter.build_system,
                      adapter.optimize_point, adapter.load_optimized,
                      cubie_bench._device_leg, cubie_bench.final_states)
        cubie_bench.final_states = (
            lambda system, solution, problem: solution.finals)
        # The states sweep optimises each size; the fake solver has no kernels.
        adapter.optimize_point = lambda *args, **kwargs: {"label": "fake"}
        adapter.load_optimized = lambda *args, **kwargs: None

    def tearDown(self):
        adapter = cubie_bench.adapter
        (adapter.make_solver, adapter.build_system, adapter.optimize_point,
         adapter.load_optimized, cubie_bench._device_leg,
         cubie_bench.final_states) = self.saved
        os.chdir(self.cwd)

    def opts(self, ns):
        return {"ns": ns, "algorithms": ["classical-rk4"],
                "fixed": ["classical-rk4"], "adaptive": [],
                "framework": "cubie", "dataset_key": "test_key",
                "numerical_tag": "cubie"}

    def run_times(self, solver, ns):
        cubie_bench.adapter.make_solver = (
            lambda system, problem, algorithm, mode, setting=None, **kw: solver)
        problem = get_problem("lorenz")
        cubie_bench._run_times(problem, self.opts(ns), object(), grid)
        return read_rows("times"), sample_legs("times")


class TestTimesResidency(SweepCase):
    def test_device_leg_reuses_the_host_legs_inputs(self):
        solver = FakeSolver()
        rows, legs = self.run_times(solver, [1024, 4096])
        device = [n for n, on_device in solver.calls if on_device]
        self.assertEqual(sorted(set(device)), [1024, 4096])
        # Every device solve synchronised its own result stream.
        self.assertTrue(solver.device_results)
        self.assertTrue(all(result.stream.synchronised == 1
                            for result in solver.device_results))
        for n in (1024, 4096):
            self.assertTrue(all(math.isfinite(v) for v in rows[n]))
            self.assertIn((n, "both"), legs)
            self.assertIn((n, "none"), legs)

    def test_finals_are_saved_from_the_host_leg(self):
        rows, _ = self.run_times(FakeSolver(), [32768])
        path = os.path.join("data", "numerical", "test_key", "lorenz",
                            "cubie_unadaptive.csv")
        saved = np.loadtxt(path, delimiter=",")
        self.assertEqual(saved.shape, (32768, 3))
        self.assertTrue(np.all(saved == 32768.0))


class TestTimesLegIsolation(SweepCase):
    def test_chunked_host_leg_keeps_its_number(self):
        solver = FakeSolver(chunk_at=4096)
        rows, legs = self.run_times(solver, [1024, 4096, 16384])
        self.assertTrue(all(math.isfinite(rows[n][0])
                            for n in (1024, 4096, 16384)))
        self.assertTrue(math.isfinite(rows[1024][1]))
        self.assertTrue(math.isnan(rows[4096][1]))
        self.assertTrue(math.isnan(rows[16384][1]))
        # The chunked sizes never launch a device solve.
        device = [n for n, on_device in solver.calls if on_device]
        self.assertEqual(device, [1024] * len(device))
        self.assertIn((1024, "none"), legs)
        self.assertNotIn((4096, "none"), legs)
        self.assertIn((16384, "both"), legs)

    def test_device_breach_abandons_only_the_device_column(self):
        attempted = []

        def breaching_leg(solver, duration, repeats):
            attempted.append(solver.last_n)
            if solver.last_n >= 4096:
                return None, [1.0]
            return 5.0, [1.0, 5.0]

        cubie_bench._device_leg = breaching_leg
        solver = FakeSolver()
        rows, _ = self.run_times(solver, [1024, 4096, 16384])
        self.assertEqual(attempted, [1024, 4096])
        self.assertTrue(all(math.isfinite(rows[n][0])
                            for n in (1024, 4096, 16384)))
        self.assertEqual(rows[1024][1], 5.0)
        self.assertTrue(math.isnan(rows[4096][1]))
        self.assertTrue(math.isnan(rows[16384][1]))

    def test_host_failure_skips_the_device_leg_and_continues(self):
        solver = FakeSolver(host_fail_at=4096)
        rows, _ = self.run_times(solver, [1024, 4096, 16384])
        self.assertTrue(all(math.isnan(v) for v in rows[4096][:-1]))
        self.assertEqual(100.0, rows[4096][-1])
        self.assertTrue(all(math.isfinite(v) for v in rows[16384]))
        self.assertFalse(any(n == 4096 and on_device
                             for n, on_device in solver.calls))


class TestStatesLegIsolation(SweepCase):
    def test_chunked_host_leg_keeps_host_time_and_build_time(self):
        solvers = {}

        def make_solver(system, row, algorithm, mode, setting=None, **kw):
            solver = FakeSolver(chunk_at=1)   # every host leg chunks
            solvers[row["states"]] = solver
            return solver

        cubie_bench.adapter.make_solver = make_solver
        cubie_bench.adapter.build_system = (
            lambda problem, package, precision=None, states=None:
            (object(), {"x{0}".format(i): 8.0 for i in range(1, states + 1)}))
        opts = self.opts([4, 8])
        cubie_bench._run_states(opts)
        rows = read_rows("states")
        for nstates in (4, 8):
            t_ms, t_dev, build_s, pct = rows[nstates]
            self.assertTrue(math.isfinite(t_ms))
            self.assertTrue(math.isnan(t_dev))
            self.assertTrue(math.isfinite(build_s))
            self.assertTrue(solvers[nstates].closed)

    def test_device_leg_reuses_each_sizes_inputs(self):
        solvers = {}

        def make_solver(system, row, algorithm, mode, setting=None, **kw):
            solver = FakeSolver()
            solvers[row["states"]] = solver
            return solver

        cubie_bench.adapter.make_solver = make_solver
        cubie_bench.adapter.build_system = (
            lambda problem, package, precision=None, states=None:
            (object(), {"x{0}".format(i): 8.0 for i in range(1, states + 1)}))
        cubie_bench._run_states(self.opts([4, 8]))
        rows = read_rows("states")
        for nstates in (4, 8):
            self.assertTrue(all(math.isfinite(v) for v in rows[nstates]))
            self.assertTrue(any(on_device
                                for _, on_device in solvers[nstates].calls))


class TestWorkPrecisionNe(SweepCase):
    def setUp(self):
        super().setUp()
        import wp_common
        self.saved_golden = wp_common.load_golden
        wp_common.load_golden = lambda problem: np.zeros((131072, 3))

    def tearDown(self):
        import wp_common
        wp_common.load_golden = self.saved_golden
        super().tearDown()

    def run_wp(self, algorithms, fixed, adaptive):
        self.solvers = []

        def make_solver(system, problem, algorithm, mode, setting=None, **kw):
            self.solvers.append(FakeSolver())
            return self.solvers[-1]

        cubie_bench.adapter.make_solver = make_solver
        opts = dict(self.opts([131072]), algorithms=algorithms, fixed=(),
                    adaptive=(), wp_fixed=fixed, wp_adaptive=adaptive)
        cubie_bench._run_wp(get_problem("lorenz"), opts, object(), grid)
        import results
        return [row for row in results.load(results.store_path("cubie", "test_key"))
                if row["analysis"] == "wp"]

    def test_a_wp_point_is_timed_on_the_resident_inputs(self):
        rows = self.run_wp(["euler"], ("euler",), ())
        self.assertTrue(rows)
        self.assertTrue(all(row["transfers"] == "none" for row in rows))
        self.assertTrue(all(math.isfinite(float(row["min_ms"])) for row in rows))
        for solver in self.solvers:
            # One untimed host solve for the finals, then only device solves.
            self.assertEqual([on_device for _, on_device in solver.calls][:2],
                             [False, True])
            self.assertTrue(all(on_device for _, on_device in solver.calls[1:]))

    def test_an_ne_leg_times_the_ne_grid_and_writes_its_finals(self):
        from protocol import N_NE
        from problems import get_problem as problem_row
        rows = self.run_wp(["backwards_euler"], ("backwards_euler",), ())
        dts = problem_row("lorenz").ne_dts()
        self.assertEqual(len(rows), len(dts))
        self.assertTrue(all(row["tier"] == "default" for row in rows))
        path = os.path.join("data", "numerical_equivalence", "cubie", "test_key",
                            "lorenz", "backwards_euler.csv")
        self.assertTrue(os.path.isfile(path))
        # The MLIR package writes beside, never over, the numba-cuda files.
        opts = dict(self.opts([131072]), algorithms=["backwards_euler"], fixed=(),
                    adaptive=(), wp_fixed=("backwards_euler",), wp_adaptive=(),
                    framework="cubie_mlir")
        cubie_bench._run_wp(get_problem("lorenz"), opts, object(), grid)
        self.assertTrue(os.path.isfile(os.path.join(
            "data", "numerical_equivalence", "cubie_mlir", "test_key", "lorenz",
            "backwards_euler.csv")))
        with open(path) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[0], "dt,traj,s1,s2,s3")
        self.assertEqual(len(lines) - 1, N_NE * len(dts))

    def test_a_timed_only_leg_writes_no_ne_file(self):
        rows = self.run_wp(["euler"], ("euler",), ())
        self.assertEqual(len(rows), 10)
        self.assertFalse(os.path.exists(os.path.join("data", "numerical_equivalence")))


class TestWorkerDeviceLeg(unittest.TestCase):
    def test_samples_reuse_the_resident_inputs(self):
        solver = FakeSolver()
        initials, parameters = grid(solver, 256)
        solver.solve(initials, parameters, 1.0)
        samples = cubie_worker.time_device_leg(solver, 1.0, 20)
        self.assertEqual(len(samples), 20)
        self.assertEqual(len(solver.device_results), 20)
        self.assertTrue(all(result.stream.synchronised == 1
                            for result in solver.device_results))

    def test_chunked_host_leg_raises_before_any_device_solve(self):
        solver = FakeSolver(chunk_at=1)
        initials, parameters = grid(solver, 256)
        solver.solve(initials, parameters, 1.0)
        with self.assertRaises(ValueError):
            cubie_worker.time_device_leg(solver, 1.0, 20)
        self.assertEqual(solver.device_results, [])


if __name__ == "__main__":
    unittest.main()
