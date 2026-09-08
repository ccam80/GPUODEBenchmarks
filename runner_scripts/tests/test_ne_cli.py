"""CLI and controller-constants tests for the numerical-equivalence tooling."""

import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "numerical_equivalence"))
sys.path.insert(0, ROOT)

import ne_common  # noqa: E402


class EnsembleGridTests(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        os.chdir(ROOT)

    def tearDown(self):
        os.chdir(self.cwd)

    def test_ne_grid_is_the_wp_prefix_for_every_problem(self):
        import numpy as np
        from problems import get_problem
        from protocol import N_NE, N_WP
        for name in ("lorenz", "pollu", "nand_gate", "ring_modulator_index2"):
            problem = get_problem(name)
            sweep, states = ne_common.load_golden_ne(problem)
            self.assertTrue(np.array_equal(
                np.float32(sweep), problem.sweep(N_WP, dtype=np.float32)[:N_NE]))
            self.assertEqual(states.shape, (N_NE, problem["states"]))
        self.assertFalse(hasattr(ne_common, "golden_ne_path"))


class PackageDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_each_cubie_package_has_its_own_tree(self):
        numba = ne_common.cubie_ne_file("kvaerno3", "k", "lorenz")
        mlir = ne_common.cubie_ne_file("kvaerno3", "k", "lorenz", "cubie_mlir")
        self.assertNotEqual(numba, mlir)
        self.assertIn(os.path.join("cubie", "k", "lorenz"), numba)
        self.assertIn(os.path.join("cubie_mlir", "k", "lorenz"), mlir)
        self.assertTrue(ne_common.cubie_ne_adaptive_file(
            "kvaerno3", "matched", "k", "lorenz", "cubie_mlir").endswith("kvaerno3_adaptive_matched.csv"))
        self.assertEqual(ne_common.ne_keys("cubie"), {"k"})
        self.assertEqual(ne_common.ne_keys("cubie_mlir"), {"k"})
        self.assertEqual(ne_common.ne_keys("julia"), set())


class CompareProblemListTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_comma_list_accepted(self):
        # A parser rejection would raise SystemExit instead of returning 1.
        import compare_numerical_equivalence as cne
        argv = sys.argv
        sys.argv = ["compare_numerical_equivalence.py", "--problem",
                    "ring_modulator,nand_gate"]
        try:
            self.assertEqual(cne.main(), 1)
        finally:
            sys.argv = argv

    def test_unknown_problem_exits(self):
        import compare_numerical_equivalence as cne
        argv = sys.argv
        sys.argv = ["compare_numerical_equivalence.py", "--problem",
                    "nosuchproblem"]
        try:
            with self.assertRaises(SystemExit):
                cne.main()
        finally:
            sys.argv = argv


class ControllerConstantsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reads_per_problem_file(self):
        path = ne_common.controller_constants_csv("pollu", "test_key")
        self.assertIn(os.path.join("julia", "test_key", "pollu"), path)
        with open(path, "w", newline="") as f:
            f.write("cubie_alias,controller,beta1,beta2,qmin,qmax,gamma,order\n"
                    "tsit5,PIController,0.23333333,0.13333334,0.2,10.0,0.9,5\n"
                    "radau_iia_5,PredictiveController,,,0.2,8.0,0.9,5\n")
        out = ne_common.load_controller_constants("pollu", "test_key")
        self.assertEqual(set(out), {"tsit5", "radau_iia_5"})
        self.assertAlmostEqual(out["tsit5"]["beta1"], 0.23333333)
        self.assertIsNone(out["radau_iia_5"]["beta1"])
        self.assertEqual(out["radau_iia_5"]["controller"],
                         "PredictiveController")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ne_common.load_controller_constants("pollu", "test_key")

    def test_default_key_is_this_machine(self):
        from bench_key import dataset_key
        path = ne_common.julia_ne_dir("pollu")
        self.assertIn(os.path.join("julia", dataset_key(), "pollu"), path)


if __name__ == "__main__":
    unittest.main()
