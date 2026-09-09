"""protocol.toml holds the repeat schedule and the watchdog alone; the Python, Julia and C++ views agree."""

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
from launch import julia_command  # noqa: E402


class PythonViewTests(unittest.TestCase):
    def test_only_the_three_tables_remain(self):
        self.assertEqual(sorted(protocol.PROTOCOL), sorted(protocol.TABLES))
        self.assertEqual(protocol.TABLES, ("repeats", "watchdog"))
        for gone in ("ensemble", "fixed", "adaptive", "newton", "plots", "optimize"):
            self.assertNotIn(gone, protocol.PROTOCOL)
        for name in ("N_WP", "STATES_GRID", "TOLS", "TIMING_TOL", "NEWTON_ATOL", "parse_ns"):
            self.assertFalse(hasattr(protocol, name), name)

    def test_values(self):
        self.assertEqual(protocol.REPEAT_CAP, protocol.get("repeats.cap"))
        self.assertEqual(protocol.REPEAT_SCHEDULE[-1][0], float("inf"))
        self.assertEqual(protocol.WATCHDOG_EXIT_CODE, 3)
        self.assertEqual(protocol.WATCHDOG_SECONDS, float(protocol.get("watchdog.seconds")))

    def test_wp_common_holds_the_timing_helpers_only(self):
        self.assertEqual(wp_common.WATCHDOG_SECONDS, protocol.WATCHDOG_SECONDS)
        public = {name for name in dir(wp_common) if not name.startswith("_") and callable(getattr(wp_common, name))}
        self.assertEqual(public, {"run_watchdogged", "repeat_bounds", "repeats_done", "timed_min_ms"})

    def test_no_environment_override(self):
        env = dict(os.environ, BENCH_WATCHDOG_SECONDS="1")
        out = subprocess.run([sys.executable, "-c",
                              "import sys; sys.path.insert(0, r'{0}'); import protocol; "
                              "print(protocol.WATCHDOG_SECONDS)".format(os.path.dirname(HERE))],
                             capture_output=True, text=True, check=True, env=env)
        self.assertEqual(float(out.stdout), protocol.WATCHDOG_SECONDS)

    def test_get(self):
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "runner_scripts", "protocol.py"),
             "get", "repeats.cap"],
            capture_output=True, text=True, check=True)
        self.assertEqual(int(out.stdout), protocol.REPEAT_CAP)


class CxxHeaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_header_carries_the_schedule_and_the_watchdog(self):
        text = protocol.cxx_header()
        self.assertIn("#define PROTOCOL_REPEAT_CAP {0}".format(protocol.REPEAT_CAP), text)
        self.assertIn("#define PROTOCOL_WATCHDOG_EXIT_CODE 3", text)
        self.assertIn("{INFINITY, 3, 10}", text)
        self.assertIn("PROTOCOL_REPEAT_SCHEDULE_ROWS {0}".format(len(protocol.REPEAT_SCHEDULE)), text)
        self.assertNotIn("N_WP", text)

    def test_write_is_idempotent(self):
        path = os.path.join(self.tmp, "protocol.h")
        protocol.write_cxx_header(path)
        first = os.stat(path).st_mtime_ns
        protocol.write_cxx_header(path)
        self.assertEqual(os.stat(path).st_mtime_ns, first)


@unittest.skipUnless(shutil.which(julia_command()[0]), "julia not on PATH")
class JuliaViewTests(unittest.TestCase):
    def test_julia_reads_the_same_values(self):
        script = (
            'include(joinpath("{0}", "runner_scripts", "watchdog.jl")); '
            'println(REPEAT_CAP, " ", REPEAT_SPREAD, " ", length(REPEAT_SCHEDULE), " ", '
            'WATCHDOG_SECONDS, " ", WATCHDOG_EXIT_CODE, " ", '
            'isdefined(@__MODULE__, :N_WP), " ", isdefined(@__MODULE__, :OPTIMIZE_N))'
        ).format(ROOT.replace("\\", "/"))
        out = subprocess.run(
            julia_command() + ["--startup-file=no", "--project=" + ROOT, "-e", script],
            capture_output=True, text=True, check=True, cwd=ROOT)
        fields = out.stdout.split()
        self.assertEqual(int(fields[0]), protocol.REPEAT_CAP)
        self.assertEqual(float(fields[1]), protocol.REPEAT_SPREAD)
        self.assertEqual(int(fields[2]), len(protocol.REPEAT_SCHEDULE))
        self.assertEqual(float(fields[3]), protocol.WATCHDOG_SECONDS)
        self.assertEqual(int(fields[4]), protocol.WATCHDOG_EXIT_CODE)
        self.assertEqual(fields[5:7], ["false", "false"])


if __name__ == "__main__":
    unittest.main()
