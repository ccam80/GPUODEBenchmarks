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
        path = ne_common.controller_constants_csv("pollu")
        self.assertIn(os.path.join("julia", "pollu"), path)
        with open(path, "w", newline="") as f:
            f.write("cubie_alias,controller,beta1,beta2,qmin,qmax,gamma,order\n"
                    "tsit5,PIController,0.23333333,0.13333334,0.2,10.0,0.9,5\n"
                    "radau_iia_5,PredictiveController,,,0.2,8.0,0.9,5\n")
        out = ne_common.load_controller_constants("pollu")
        self.assertEqual(set(out), {"tsit5", "radau_iia_5"})
        self.assertAlmostEqual(out["tsit5"]["beta1"], 0.23333333)
        self.assertIsNone(out["radau_iia_5"]["beta1"])
        self.assertEqual(out["radau_iia_5"]["controller"],
                         "PredictiveController")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ne_common.load_controller_constants("pollu")


if __name__ == "__main__":
    unittest.main()
