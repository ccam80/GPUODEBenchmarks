"""Cubie sweep legs: device inputs resident for the device leg only, and a
device-only failure or breach leaves the host-path number standing."""

import math
import os
import sys
import tempfile
import types
import unittest
import weakref

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cubie_julia_overlap"))


class FakeDeviceArray:
    def __init__(self, host):
        self.shape = host.shape
        self.dtype = host.dtype


class FakeCuda:
    """numba.cuda stand-in: uploads are tracked weakly so residency is observable."""

    def __init__(self):
        self.uploads = []
        self.flushes = 0
        self.oom_at = None

    def to_device(self, host):
        if self.oom_at is not None and host.shape[-1] >= self.oom_at:
            raise MemoryError("CUDA_ERROR_OUT_OF_MEMORY")
        array = FakeDeviceArray(host)
        self.uploads.append(weakref.ref(array))
        return array

    def synchronize(self):
        pass

    def current_context(self):
        return self

    @property
    def memory_manager(self):
        return self

    @property
    def deallocations(self):
        return self

    def clear(self):
        self.flushes += 1

    def live(self):
        return [ref for ref in self.uploads if ref() is not None]


FAKE_CUDA = FakeCuda()
numba = types.ModuleType("numba")
numba.cuda = FAKE_CUDA
sys.modules["numba"] = numba
cubie = types.ModuleType("cubie")
cubie.cache_root = types.ModuleType("cubie.cache_root")
cubie.cache_root.set_cache_root = lambda path: None
sys.modules["cubie"] = cubie
sys.modules["cubie.cache_root"] = cubie.cache_root

import cubie_bench  # noqa: E402
import cubie_worker  # noqa: E402
from problems import STATES_PROBLEM, get_problem  # noqa: E402


class FakeSolution:
    def __init__(self, n):
        self.finals = np.full((n, 3), float(n), dtype=np.float32)


class FakeSolver:
    """solve() records (n, on_device, live device uploads) per call."""

    def __init__(self, device_fail_at=None, host_fail_at=None):
        self.calls = []
        self.device_fail_at = device_fail_at
        self.host_fail_at = host_fail_at
        self.closed = False

    def solve(self, initial_values, parameters, blocksize, duration,
              on_device=False):
        n = initial_values.shape[1]
        self.calls.append((n, on_device, len(FAKE_CUDA.live())))
        if on_device:
            if not isinstance(initial_values, FakeDeviceArray):
                raise AssertionError("device leg was given host arrays")
            if self.device_fail_at is not None and n >= self.device_fail_at:
                raise ValueError("Device-resident results require the batch "
                                 "to fit in a single chunk")
            return object()
        if isinstance(initial_values, FakeDeviceArray):
            raise AssertionError("host leg was given device arrays")
        if self.host_fail_at is not None and n == self.host_fail_at:
            raise MemoryError("allocating bytes")
        return FakeSolution(n)

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
        FAKE_CUDA.uploads.clear()
        FAKE_CUDA.flushes = 0
        FAKE_CUDA.oom_at = None
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
    def test_device_inputs_live_only_during_the_device_leg(self):
        solver = FakeSolver()
        rows, _ = self.run_times(solver, [1024, 4096])
        host = [live for n, on_device, live in solver.calls if not on_device]
        device = [live for n, on_device, live in solver.calls if on_device]
        self.assertTrue(host and all(live == 0 for live in host))
        self.assertTrue(device and all(live == 2 for live in device))
        self.assertEqual(FAKE_CUDA.live(), [])
        self.assertEqual(FAKE_CUDA.flushes, 2)
        for n in (1024, 4096):
            self.assertTrue(all(math.isfinite(v) for v in rows[n]))

    def test_finals_are_saved_from_the_host_leg(self):
        rows, _ = self.run_times(FakeSolver(), [32768])
        path = os.path.join("data", "numerical", "test_key", "lorenz",
                            "cubie_unadaptive.csv")
        saved = np.loadtxt(path, delimiter=",")
        self.assertEqual(saved.shape, (32768, 3))
        self.assertTrue(np.all(saved == 32768.0))


class TestTimesLegIsolation(SweepCase):
    def test_device_failure_keeps_the_host_number(self):
        solver = FakeSolver(device_fail_at=4096)
        rows, legs = self.run_times(solver, [1024, 4096, 16384])
        self.assertTrue(all(math.isfinite(rows[n][0])
                            for n in (1024, 4096, 16384)))
        self.assertTrue(math.isfinite(rows[1024][1]))
        self.assertTrue(math.isnan(rows[4096][1]))
        self.assertTrue(math.isnan(rows[16384][1]))
        # The device leg is attempted again at the next size.
        self.assertTrue(any(n == 16384 and on_device
                            for n, on_device, _ in solver.calls))
        self.assertIn((1024, "none"), legs)
        self.assertNotIn((4096, "none"), legs)
        self.assertIn((16384, "both"), legs)
        self.assertEqual(FAKE_CUDA.live(), [])

    def test_upload_oom_keeps_the_host_number_and_frees(self):
        FAKE_CUDA.oom_at = 4096
        solver = FakeSolver()
        rows, _ = self.run_times(solver, [1024, 4096])
        self.assertTrue(math.isfinite(rows[4096][0]))
        self.assertTrue(math.isnan(rows[4096][1]))
        self.assertTrue(math.isfinite(rows[1024][1]))
        self.assertEqual(FAKE_CUDA.live(), [])
        self.assertEqual(FAKE_CUDA.flushes, 2)

    def test_device_breach_abandons_only_the_device_column(self):
        attempted = []

        def breaching_leg(solver, initials, parameters, duration, repeats):
            attempted.append(initials.shape[1])
            if initials.shape[1] >= 4096:
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
        self.assertTrue(all(math.isnan(v) for v in rows[4096]))
        self.assertTrue(all(math.isfinite(v) for v in rows[16384]))
        self.assertFalse(any(n == 4096 and on_device
                             for n, on_device, _ in solver.calls))


class TestStatesLegIsolation(SweepCase):
    def test_device_failure_keeps_host_time_and_build_time(self):
        solvers = {}

        def make_solver(system, row, algorithm, dt=None):
            solver = FakeSolver(device_fail_at=1)   # every device leg fails
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
            t_ms, t_dev, build_s = rows[nstates]
            self.assertTrue(math.isfinite(t_ms))
            self.assertTrue(math.isnan(t_dev))
            self.assertTrue(math.isfinite(build_s))
            self.assertTrue(solvers[nstates].closed)
        self.assertEqual(FAKE_CUDA.live(), [])


class TestWorkerDeviceLeg(unittest.TestCase):
    def setUp(self):
        FAKE_CUDA.uploads.clear()
        FAKE_CUDA.flushes = 0
        FAKE_CUDA.oom_at = None

    def test_inputs_freed_after_the_samples(self):
        solver = FakeSolver()
        initials, parameters = grid(solver, 256)
        samples = cubie_worker.time_device_leg(solver, initials, parameters,
                                               1.0, 20)
        self.assertEqual(len(samples), 20)
        self.assertTrue(all(live == 2 for _, _, live in solver.calls))
        self.assertEqual(FAKE_CUDA.live(), [])
        self.assertEqual(FAKE_CUDA.flushes, 1)

    def test_inputs_freed_when_the_solve_raises(self):
        solver = FakeSolver(device_fail_at=1)
        initials, parameters = grid(solver, 256)
        with self.assertRaises(ValueError):
            cubie_worker.time_device_leg(solver, initials, parameters, 1.0,
                                         20)
        self.assertEqual(FAKE_CUDA.live(), [])
        self.assertEqual(FAKE_CUDA.flushes, 1)


if __name__ == "__main__":
    unittest.main()
