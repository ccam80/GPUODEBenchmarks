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
from problems import STATES_PROBLEM, get_problem  # noqa: E402


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

    def solve(self, initial_values, parameters, blocksize, duration,
              on_device=False):
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


def read_rows(path):
    rows = {}
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            rows[int(float(fields[0]))] = [float(v) for v in fields[1:]]
    return rows


def sample_legs(path):
    """{(n, transfers): count} from a samples log."""
    counts = {}
    with open(path) as handle:
        header = handle.readline().strip().split(",")
        for line in handle:
            row = dict(zip(header, line.strip().split(",")))
            key = (int(row["n"]), row["transfers"])
            counts[key] = counts.get(key, 0) + 1
    return counts


class SweepCase(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)
        self.saved = (cubie_bench._make_fixed_solver, cubie_bench._device_leg,
                      cubie_bench.final_states, cubie_bench.build_system)
        cubie_bench.final_states = (
            lambda system, solution, problem: solution.finals)

    def tearDown(self):
        (cubie_bench._make_fixed_solver, cubie_bench._device_leg,
         cubie_bench.final_states, cubie_bench.build_system) = self.saved
        os.chdir(self.cwd)

    def opts(self, ns):
        return {"ns": ns, "algorithms": ["classical-rk4"],
                "fixed": ["classical-rk4"], "adaptive": [],
                "framework": "cubie", "framework_dir": "CUBIE",
                "prefix": "Cubie", "dataset_key": "test_key",
                "numerical_tag": "cubie", "name_suffix": ""}

    def run_times(self, solver, ns):
        cubie_bench._make_fixed_solver = (
            lambda system, problem, algorithm, dt=None: solver)
        problem = get_problem("lorenz")
        cubie_bench._run_times(problem, self.opts(ns), object(), grid)
        base = os.path.join("data", "CUBIE", "test_key", "lorenz")
        return (read_rows(os.path.join(
                    base, "Cubie_times_fixed_classical-rk4.txt")),
                sample_legs(os.path.join(
                    base, "Cubie_samples_times_fixed_classical-rk4.csv")))


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

        def make_solver(system, row, algorithm, dt=None):
            solver = FakeSolver(chunk_at=1)   # every host leg chunks
            solvers[row["states"]] = solver
            return solver

        cubie_bench._make_fixed_solver = make_solver
        cubie_bench.build_system = (
            lambda row, precision, name_suffix="":
            (object(), {"x{0}".format(i): 8.0
                        for i in range(1, row["states"] + 1)}))
        opts = self.opts([4, 8])
        cubie_bench._run_states(opts)
        path = os.path.join("data", "CUBIE", "test_key", STATES_PROBLEM,
                            "Cubie_states_fixed_classical-rk4.txt")
        rows = read_rows(path)
        for nstates in (4, 8):
            t_ms, t_dev, build_s, pct = rows[nstates]
            self.assertTrue(math.isfinite(t_ms))
            self.assertTrue(math.isnan(t_dev))
            self.assertTrue(math.isfinite(build_s))
            self.assertTrue(solvers[nstates].closed)

    def test_device_leg_reuses_each_sizes_inputs(self):
        solvers = {}

        def make_solver(system, row, algorithm, dt=None):
            solver = FakeSolver()
            solvers[row["states"]] = solver
            return solver

        cubie_bench._make_fixed_solver = make_solver
        cubie_bench.build_system = (
            lambda row, precision, name_suffix="":
            (object(), {"x{0}".format(i): 8.0
                        for i in range(1, row["states"] + 1)}))
        cubie_bench._run_states(self.opts([4, 8]))
        path = os.path.join("data", "CUBIE", "test_key", STATES_PROBLEM,
                            "Cubie_states_fixed_classical-rk4.txt")
        rows = read_rows(path)
        for nstates in (4, 8):
            self.assertTrue(all(math.isfinite(v) for v in rows[nstates]))
            self.assertTrue(any(on_device
                                for _, on_device in solvers[nstates].calls))


class TestWorkerDeviceLeg(unittest.TestCase):
    def test_samples_reuse_the_resident_inputs(self):
        solver = FakeSolver()
        initials, parameters = grid(solver, 256)
        solver.solve(initials, parameters, 64, 1.0)
        samples = cubie_worker.time_device_leg(solver, 1.0, 20)
        self.assertEqual(len(samples), 20)
        self.assertEqual(len(solver.device_results), 20)
        self.assertTrue(all(result.stream.synchronised == 1
                            for result in solver.device_results))

    def test_chunked_host_leg_raises_before_any_device_solve(self):
        solver = FakeSolver(chunk_at=1)
        initials, parameters = grid(solver, 256)
        solver.solve(initials, parameters, 64, 1.0)
        with self.assertRaises(ValueError):
            cubie_worker.time_device_leg(solver, 1.0, 20)
        self.assertEqual(solver.device_results, [])


if __name__ == "__main__":
    unittest.main()
