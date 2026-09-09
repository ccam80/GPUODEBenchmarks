"""grid.py against a numpy reference, the committed tests/grids files, the containment rule, and the Julia and C++ reproductions bit for bit."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import grid  # noqa: E402
from launch import julia_command  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
CPP_TEST = os.path.join(REPO_ROOT, "GPU_ODE_MPGOS", "tests", "test_grid.cu")
NVCC_FLAGS = ["-O3", "-std=c++17"]


def numpy_reference(scale, lo, hi, n):
    """The 1.2 formula written out in numpy, independently of grid.py."""
    i = np.arange(n, dtype=np.float64)
    if scale == "linear":
        v = lo + i * ((hi - lo) / (n - 1))
    else:
        a, b = np.log10(np.float64(lo)), np.log10(np.float64(hi))
        v = np.array([np.float64(10.0) ** e for e in a + i * ((b - a) / (n - 1))])
    v[n - 1] = hi
    return v.astype(np.float32)


class FormulaTests(unittest.TestCase):
    def test_every_problem_grid_matches_the_numpy_reference_at_131072(self):
        problems = grid.problem_grids()
        self.assertGreaterEqual(len(problems), 8)
        for problem, scale, lo, hi in problems:
            ours = grid.grid_values(scale, lo, hi, grid.REFERENCE_N)
            self.assertEqual(ours.dtype, np.float32)
            np.testing.assert_array_equal(ours, numpy_reference(scale, lo, hi, grid.REFERENCE_N),
                                          err_msg=problem)
            self.assertEqual(ours[0], np.float32(lo))
            self.assertEqual(ours[-1], np.float32(hi))

    def test_small_grids_endpoints_and_the_log_formula(self):
        v = grid.grid_values("linear", 0.0, 21.0, 8)
        np.testing.assert_array_equal(v, np.float32([0, 3, 6, 9, 12, 15, 18, 21]))
        w = grid.grid_values("log", 1e-3, 1.0, 4)
        np.testing.assert_array_equal(w, np.float32([1e-3, 1e-2, 1e-1, 1.0]))
        self.assertEqual(w[0], np.float32(1e-3))
        self.assertEqual(w[-1], np.float32(1.0))

    def test_the_run_precision_widens_the_float32_values(self):
        spec = dict(grid_scale="log", grid_min=3.5e-2, grid_max=3.5, n=1024,
                    grid_dtype="float32", precision="float64")
        wide = grid.grid(spec)
        self.assertEqual(wide.dtype, np.float64)
        narrow = grid.grid(dict(spec, precision="float32"))
        self.assertEqual(narrow.dtype, np.float32)
        np.testing.assert_array_equal(wide, narrow.astype(np.float64))
        # Widened values carry no more digits than float32 holds.
        self.assertTrue(np.array_equal(wide.astype(np.float32).astype(np.float64), wide))
        with self.assertRaises(ValueError):
            grid.grid(dict(spec, precision="float16"))

    def test_bad_grids_are_refused(self):
        with self.assertRaises(ValueError):
            grid.grid_values("linear", 0.0, 1.0, 1)
        with self.assertRaises(ValueError):
            grid.grid_values("log", 0.0, 1.0, 8)
        with self.assertRaises(ValueError):
            grid.grid_values("cubic", 0.0, 1.0, 8)
        with self.assertRaises(ValueError):
            grid.grid_values("linear", 0.0, 1.0, 8, grid_dtype="float64")
        with self.assertRaises(ValueError):
            grid.grid_values("linear", float("nan"), 1.0, 8)


class ReferenceFileTests(unittest.TestCase):
    def test_the_committed_npy_files_are_the_current_grids(self):
        for problem, scale, lo, hi in grid.problem_grids():
            path = grid.reference_path(problem)
            self.assertTrue(os.path.isfile(path), path)
            stored = np.load(path)
            self.assertEqual(stored.dtype, np.float32)
            self.assertEqual(stored.shape, (grid.REFERENCE_N,))
            np.testing.assert_array_equal(stored, grid.grid_values(scale, lo, hi, grid.REFERENCE_N),
                                          err_msg=problem)

    def test_write_reference_grids_writes_one_file_per_problem(self):
        tmp = tempfile.mkdtemp(prefix="grids_")
        self.addCleanup(shutil.rmtree, tmp, True)
        paths = grid.write_reference_grids(tmp, n=64)
        self.assertEqual(sorted(os.path.basename(p) for p in paths),
                         sorted(p + "_64.npy" for p, *_ in grid.problem_grids()))
        self.assertEqual(np.load(paths[0]).shape, (64,))


class ContainmentTests(unittest.TestCase):
    def test_the_ne_prefix_grid_is_contained_by_the_131072_grid_and_equals_its_prefix(self):
        for problem, scale, lo, hi in grid.problem_grids():
            full = dict(grid_scale=scale, grid_min=lo, grid_max=hi, n=grid.REFERENCE_N,
                        grid_dtype="float32")
            values = grid.grid_values(scale, lo, hi, grid.REFERENCE_N)
            point = grid.grid_point(scale, lo, hi, grid.REFERENCE_N, 1023)
            self.assertEqual(np.float32(point), values[1023], problem)
            prefix = dict(full, n=1024, grid_max=point)
            self.assertTrue(grid.grid_contains(full, prefix), problem)
            self.assertFalse(grid.grid_contains(prefix, full), problem)
            np.testing.assert_array_equal(grid.grid(prefix), values[:1024], err_msg=problem)
        # The float32 point as grid_max is contained too, but does not reproduce the prefix everywhere.
        values = grid.grid_values("linear", 0.0, 21.0, grid.REFERENCE_N)
        widened = dict(grid_scale="linear", grid_min=0.0, grid_max=float(values[1023]),
                       n=1024, grid_dtype="float32")
        self.assertTrue(grid.grid_contains(dict(widened, n=grid.REFERENCE_N, grid_max=21.0), widened))
        self.assertFalse(np.array_equal(grid.grid(widened), values[:1024]))
        self.assertEqual(grid.grid_point("linear", 0.0, 21.0, 8, 7), 21.0)
        self.assertEqual(grid.grid_point("linear", 0.0, 21.0, 8, 0), 0.0)
        with self.assertRaises(ValueError):
            grid.grid_point("linear", 0.0, 21.0, 8, 8)

    def test_containment_needs_the_same_scale_min_and_dtype(self):
        full = dict(grid_scale="linear", grid_min=0.0, grid_max=21.0, n=131072,
                    grid_dtype="float32")
        same = dict(full)
        self.assertTrue(grid.grid_contains(full, same))
        self.assertTrue(grid.grid_equal(full, same))
        self.assertFalse(grid.grid_contains(full, dict(full, grid_scale="log", grid_min=1e-3)))
        self.assertFalse(grid.grid_contains(full, dict(full, grid_min=1.0)))
        self.assertFalse(grid.grid_contains(full, dict(full, grid_max=20.0)))
        self.assertFalse(grid.grid_contains(full, dict(full, grid_dtype="float64")))
        self.assertFalse(grid.grid_contains(full, dict(full, n=1024, grid_max=0.5)))
        self.assertFalse(grid.grid_equal(full, dict(full, n=1024)))


class JuliaGridTests(unittest.TestCase):
    def test_grid_jl_matches_the_npy_files_bit_for_bit(self):
        if shutil.which(julia_command()[0]) is None:
            self.skipTest("julia is not on PATH")
        script = os.path.join(HERE, "test_grid.jl")
        proc = subprocess.run(julia_command() + ["--project=" + REPO_ROOT, script],
                              cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("grid.jl", proc.stdout)


def nvcc_build(source, exe):
    """Build a host-only .cu test with nvcc and the Bench.cu host flags; on Windows inside the VS developer shell."""
    if os.name == "nt":
        script = (
            "$ErrorActionPreference = 'Stop'; "
            "$vswhere = \"${{env:ProgramFiles(x86)}}\\Microsoft Visual Studio\\Installer\\vswhere.exe\"; "
            "$vsPath = & $vswhere -latest -products * -requires "
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath; "
            "Import-Module (Join-Path $vsPath 'Common7\\Tools\\Microsoft.VisualStudio.DevShell.dll'); "
            "Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation "
            "-DevCmdArguments '-arch=x64' | Out-Null; "
            "nvcc -o '{exe}' '{source}' {flags}; exit $LASTEXITCODE").format(
                exe=exe, source=source, flags=" ".join(NVCC_FLAGS))
        argv = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]
    else:
        argv = ["nvcc", "-o", exe, source] + NVCC_FLAGS
    return subprocess.run(argv, capture_output=True, text=True)


class CppGridTests(unittest.TestCase):
    def test_grid_cuh_matches_the_npy_files_bit_for_bit(self):
        if shutil.which("nvcc") is None:
            self.skipTest("nvcc is not on PATH")
        tmp = tempfile.mkdtemp(prefix="grid_cuh_")
        self.addCleanup(shutil.rmtree, tmp, True)
        exe = os.path.join(tmp, "test_grid.exe" if os.name == "nt" else "test_grid")
        built = nvcc_build(CPP_TEST, exe)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        for problem, scale, lo, hi in grid.problem_grids():
            proc = subprocess.run([exe, grid.reference_path(problem), scale, repr(lo), repr(hi)],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, problem + ": " + proc.stdout + proc.stderr)
            self.assertIn("ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
