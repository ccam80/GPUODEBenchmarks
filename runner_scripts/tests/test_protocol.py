"""protocol.toml is the one source: the Python, Julia and C++ views agree."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

import protocol  # noqa: E402
import wp_common  # noqa: E402
from problems import get_problem  # noqa: E402


class PythonViewTests(unittest.TestCase):
    def test_grids_follow_the_exponent_ranges(self):
        lorenz = get_problem("lorenz")
        self.assertEqual(lorenz.dts(), protocol.fixed_dts(1.0, protocol.WP_K))
        self.assertEqual(lorenz.dts("euler"),
                         protocol.fixed_dts(1.0, protocol.EULER_K))
        self.assertEqual(len(protocol.TOLS),
                         protocol.TOL_K[1] - protocol.TOL_K[0] + 1)
        self.assertEqual(lorenz.timing_dt, 2.0 ** -protocol.TIMING_DT_K)

    def test_newton_scale_is_one_table(self):
        self.assertEqual(protocol.NEWTON_ATOL, protocol.get("newton.atol"))
        self.assertEqual(protocol.NEWTON_RTOL, protocol.get("newton.rtol"))
        self.assertGreater(protocol.NEWTON_ATOL, 0.0)
        self.assertGreater(protocol.NEWTON_RTOL, 0.0)

    def test_wp_common_reexports(self):
        self.assertIs(wp_common.TOLS, protocol.TOLS)
        self.assertEqual(wp_common.N_WP, protocol.N_WP)
        self.assertEqual(wp_common.REPEAT_SCHEDULE, protocol.REPEAT_SCHEDULE)
        self.assertEqual(wp_common.REPEAT_SCHEDULE[-1][0], float("inf"))

    def test_performance_ns(self):
        self.assertEqual(protocol.performance_ns(512), [8, 32, 128, 512])
        self.assertEqual(protocol.parse_ns("512", 128), [128, 512])
        self.assertEqual(protocol.parse_ns("32768,134217728"),
                         [32768, 134217728])

    def test_get(self):
        self.assertEqual(protocol.get("ensemble.n_wp"), protocol.N_WP)
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "runner_scripts", "protocol.py"),
             "get", "ensemble.states_grid"],
            capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.split(),
                         [str(s) for s in protocol.get("ensemble.states_grid")])


class CxxHeaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_header_carries_the_schedule(self):
        text = protocol.cxx_header()
        self.assertIn("#define PROTOCOL_N_WP {0}".format(protocol.N_WP), text)
        self.assertIn("#define PROTOCOL_REPEAT_CAP {0}".format(
            protocol.REPEAT_CAP), text)
        self.assertIn("{INFINITY, 3, 10}", text)
        self.assertIn("PROTOCOL_REPEAT_SCHEDULE_ROWS {0}".format(
            len(protocol.REPEAT_SCHEDULE)), text)

    def test_write_is_idempotent(self):
        path = os.path.join(self.tmp, "protocol.h")
        protocol.write_cxx_header(path)
        first = os.stat(path).st_mtime_ns
        protocol.write_cxx_header(path)
        self.assertEqual(os.stat(path).st_mtime_ns, first)


@unittest.skipUnless(shutil.which("julia"), "julia not on PATH")
class JuliaViewTests(unittest.TestCase):
    def test_julia_reads_the_same_values(self):
        script = (
            'include(joinpath("{0}", "runner_scripts", "problems.jl")); '
            'include(joinpath("{0}", "runner_scripts", "watchdog.jl")); '
            'println(N_WP, " ", N_NE, " ", TIMING_TOL, " ", REPEAT_CAP, " ", '
            'WATCHDOG_SECONDS, " ", length(TOLS), " ", '
            'join(problem_dts(get_problem("lorenz")), ","), " ", '
            'NEWTON_ATOL, " ", NEWTON_RTOL)'
        ).format(ROOT.replace("\\", "/"))
        out = subprocess.run(
            ["julia", "--startup-file=no", "--project=" + ROOT, "-e", script],
            capture_output=True, text=True, check=True, cwd=ROOT)
        fields = out.stdout.split()
        self.assertEqual(int(fields[0]), protocol.N_WP)
        self.assertEqual(int(fields[1]), protocol.N_NE)
        self.assertEqual(float(fields[2]), protocol.TIMING_TOL)
        self.assertEqual(int(fields[3]), protocol.REPEAT_CAP)
        self.assertEqual(float(fields[4]), protocol.WATCHDOG_SECONDS)
        self.assertEqual(int(fields[5]), len(protocol.TOLS))
        self.assertEqual([float(v) for v in fields[6].split(",")],
                         get_problem("lorenz").dts())
        self.assertEqual(float(fields[7]), protocol.NEWTON_ATOL)
        self.assertEqual(float(fields[8]), protocol.NEWTON_RTOL)


if __name__ == "__main__":
    unittest.main()
