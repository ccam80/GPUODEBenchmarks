"""julia_driver.py against a fake julia: one process per build with the lock and floor flags, the hard-exit abandonment and retry, the store-driven abandonment, a crashed build, the spawn cap and the RAM gate."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gpu"))
sys.path.insert(0, ROOT)

import bench  # noqa: E402
import julia_driver  # noqa: E402
import store  # noqa: E402
import trials  # noqa: E402

KEY = "windows_RTX-4070-SUPER"
NAN = float("nan")

# Fake julia: records min_ms = 1.0 per trial unless plan.json says "hang" (exit 3), "crash" (exit 2) or "skip" (once).
FAKE_JULIA = '''
import json, os, sys
sys.path.insert(0, r"{runner_scripts}")
import store
argv = sys.argv[1:]
if "-e" in argv:
    sys.exit(0)
path = argv[argv.index("--trials") + 1]
plan = json.load(open(os.path.join(os.path.dirname(path), "plan.json")))
with open(os.path.join(os.path.dirname(path), "call.%d.json" % os.getpid()), "w") as h:
    json.dump({{"path": os.path.basename(path), "argv": argv}}, h)
data = store.Store(plan["root"])
for line in open(path):
    if not line.strip():
        continue
    t = json.loads(line)
    with open(path + ".progress", "w") as h:
        json.dump({{"trial_id": t["trial_id"], "stage": "solve", "started_utc": "2026-09-09T00:00:00Z"}}, h)
    action = plan["actions"].get(t["trial_id"], "")
    if action == "hang":
        sys.exit(3)
    if action == "crash":
        sys.exit(2)
    if action == "skip":
        # Once: the retry records it.
        plan["actions"] = {{k: v for k, v in plan["actions"].items() if k != t["trial_id"]}}
        json.dump(plan, open(os.path.join(os.path.dirname(path), "plan.json"), "w"))
        continue
    if not t["transfers"]:
        continue
    spec = {{f: (None if isinstance(t[f], float) and t[f] != t[f] else t[f]) for f in store.TRIAL_FIELDS}}
    data.record_batch([dict(spec, transfers=x, key=plan["key"], states=3, min_ms=1.0) for x in t["transfers"]])
sys.exit(0)
'''.format(runner_scripts=os.path.dirname(HERE))


def plan_julia(*argv):
    args = bench.parse_args(["plan", "--set", "perf", "-p", "julia_gpu", "-s", "lorenz", "-g", "tsit5,vern7",
                             "-n", "8,32,128"] + list(argv))
    return bench.plan_trials(bench.resolve(args), KEY, os.path.join(tempfile.gettempdir(), "no-data"))["julia_gpu"]


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jd_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "data")
        self.fake = os.path.join(self.tmp, "fake_julia.py")
        with open(self.fake, "w", encoding="utf-8") as handle:
            handle.write(FAKE_JULIA)
        self.trials = plan_julia()
        self.path = os.path.join(self.tmp, "run", "julia_gpu.jsonl")
        trials.write_jsonl(self.path, self.trials)
        for patch in (mock.patch.object(julia_driver, "DATA_ROOT", self.root),
                      mock.patch.object(julia_driver, "dataset_key", lambda: KEY),
                      mock.patch.object(julia_driver, "julia_command", lambda: [sys.executable, self.fake]),
                      mock.patch.object(julia_driver.time, "sleep", lambda _s: None)):
            patch.start()
            self.addCleanup(patch.stop)

    def run_driver(self, actions=None, *argv):
        with open(os.path.join(os.path.dirname(self.path), "plan.json"), "w") as handle:
            json.dump({"root": self.root, "key": KEY, "actions": actions or {}}, handle)
        status = julia_driver.main(["--trials", self.path] + list(argv))
        calls = []
        for name in sorted(os.listdir(os.path.dirname(self.path))):
            if name.startswith("call.") and name.endswith(".json"):
                with open(os.path.join(os.path.dirname(self.path), name)) as handle:
                    calls.append(json.load(handle))
        return status, calls

    def solve(self, algorithm, controller, n):
        return [t for t in self.trials if t["algorithm"] == algorithm and t["controller"] == controller
                and t["n"] == n][0]

    def rows(self):
        return store.Store(self.root).rows()

    def test_one_process_per_build_with_the_lock_and_floor_flags(self):
        status, calls = self.run_driver(None, "--floor")
        self.assertEqual(status, 0)
        self.assertEqual(len(trials.builds_of(self.trials)), 4)
        self.assertEqual(sorted(c["path"] for c in calls),
                         ["julia_gpu.build{0:03d}.jsonl".format(i) for i in range(1, 5)])
        for call in calls:
            argv = call["argv"]
            self.assertTrue(argv[0].endswith("bench_ode_gpu.jl"))
            self.assertEqual(argv[argv.index("--gpu-lock") + 1], self.path + ".gpulock")
            self.assertEqual(argv[argv.index("--store-python") + 1], sys.executable)
            self.assertEqual(argv[-1], "--floor")
            build_path = argv[argv.index("--trials") + 1]
            back = trials.read_jsonl(build_path)
            self.assertEqual(len({trials.build_key(t) for t in back}), 1)
            self.assertEqual([t["n"] for t in back], [8, 32, 128])
        rows = self.rows()
        self.assertEqual(len(rows), 24)
        self.assertEqual({r["min_ms"] for r in rows}, {1.0})
        self.assertEqual({r["package"] for r in rows}, {"julia_gpu"})
        self.assertEqual({r["key"] for r in rows}, {KEY})

    def test_a_hard_exit_abandons_the_harder_runs_only(self):
        hung = self.solve("tsit5", "fixed", 32)
        status, calls = self.run_driver({hung["trial_id"]: "hang"})
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 4)
        abandoned = [r for r in self.rows() if r["min_ms"] != r["min_ms"]]
        self.assertEqual(sorted((r["n"], r["transfers"]) for r in abandoned),
                         [(32, "both"), (32, "none"), (128, "both"), (128, "none")])
        self.assertEqual({(r["algorithm"], r["controller"]) for r in abandoned}, {("tsit5", "fixed")})
        self.assertEqual({r["reason"] for r in abandoned}, {"abandoned: hard-exit at " + hung["trial_id"]})
        self.assertEqual({r["states"] for r in abandoned}, {3})
        self.assertTrue(all(r["suite_rev"] for r in abandoned))
        finite = [r for r in self.rows() if r["min_ms"] == 1.0]
        self.assertEqual(len(finite), 20)
        self.assertEqual([r["n"] for r in finite if (r["algorithm"], r["controller"], r["transfers"])
                          == ("tsit5", "fixed", "both")], [8])

    def test_a_hard_exit_on_the_first_line_abandons_every_run_of_the_build(self):
        first = self.solve("vern7", "default", 8)
        status, calls = self.run_driver({first["trial_id"]: "hang"})
        self.assertEqual(status, 0)
        abandoned = [r for r in self.rows() if r["min_ms"] != r["min_ms"]]
        self.assertEqual({(r["algorithm"], r["controller"]) for r in abandoned}, {("vern7", "default")})
        self.assertEqual(sorted(r["n"] for r in abandoned), [8, 8, 32, 32, 128, 128])
        self.assertEqual({r["reason"] for r in abandoned}, {"abandoned: hard-exit at " + first["trial_id"]})

    def test_a_hard_exit_reruns_the_builds_trials_still_without_a_row(self):
        skipped = self.solve("tsit5", "default", 8)
        hung = self.solve("tsit5", "default", 32)
        status, calls = self.run_driver({skipped["trial_id"]: "skip", hung["trial_id"]: "hang"})
        self.assertEqual(status, 0)
        paths = sorted(c["path"] for c in calls)
        self.assertEqual(len(paths), 5)
        retry = [p for p in paths if ".retry1." in p]
        self.assertEqual(len(retry), 1)
        back = trials.read_jsonl(os.path.join(os.path.dirname(self.path), retry[0]))
        self.assertEqual([t["n"] for t in back], [8])
        rows = [r for r in self.rows() if r["algorithm"] == "tsit5" and r["controller"] == "default"]
        self.assertEqual(sorted((r["n"], r["min_ms"] == r["min_ms"]) for r in rows),
                         [(8, True), (8, True), (32, False), (32, False), (128, False), (128, False)])

    def test_the_stores_timeouts_abandon_the_harder_runs_before_a_build_spawns(self):
        failed = self.solve("vern7", "fixed", 32)
        spec = {f: failed[f] for f in store.TRIAL_FIELDS}
        store.Store(self.root).record(dict(spec, transfers="none", key=KEY, states=3, min_ms=NAN,
                                           reason="timeout: 130.0s over the 120s cap"))
        status, calls = self.run_driver()
        self.assertEqual(status, 0)
        rows = [r for r in self.rows() if r["algorithm"] == "vern7" and r["controller"] == "fixed"]
        self.assertEqual(sorted((r["n"], r["transfers"], r["min_ms"] == r["min_ms"]) for r in rows),
                         [(8, "both", True), (8, "none", True), (32, "both", True), (32, "none", True),
                          (128, "both", True), (128, "none", False)])
        self.assertEqual([r["reason"] for r in rows if r["n"] == 128 and r["transfers"] == "none"],
                         ["abandoned: timeout at " + failed["trial_id"]])
        vern7 = [c for c in calls if any(t["algorithm"] == "vern7" and t["controller"] == "fixed"
                                         for t in trials.read_jsonl(c["argv"][c["argv"].index("--trials") + 1]))]
        self.assertEqual([t["transfers"] for t in trials.read_jsonl(vern7[0]["argv"][vern7[0]["argv"].index("--trials") + 1])],
                         [["both", "none"], ["both", "none"], ["both"]])

    def test_a_crashed_build_fails_the_run_and_the_other_builds_still_run(self):
        crashed = self.solve("vern7", "fixed", 8)
        status, calls = self.run_driver({crashed["trial_id"]: "crash"})
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 4)
        rows = self.rows()
        self.assertEqual(len(rows), 18)
        self.assertNotIn(("vern7", "fixed"), {(r["algorithm"], r["controller"]) for r in rows})
        self.assertFalse(os.path.isfile(self.path + ".gpulock"))

    def test_no_trials_with_transfers_is_a_no_op(self):
        trials.write_jsonl(self.path, [dict(t, transfers=[]) for t in self.trials])
        status, calls = self.run_driver()
        self.assertEqual(status, 0)
        self.assertEqual(calls, [])


class FakeProc:
    """A leg process that exits 0 after `ticks` polls."""

    def __init__(self, ticks=2):
        self.ticks = ticks

    def poll(self):
        self.ticks -= 1
        return None if self.ticks > 0 else 0


class SpawnTests(unittest.TestCase):
    """The spawn cap and the RAM gate, with the processes faked."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jd_spawn_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.live = []
        self.peak = 0
        self.free_gb = 999.0

        def fake_popen(argv, cwd=None):
            proc = FakeProc()
            self.live = [p for p in self.live if p.poll() is None] + [proc]
            self.peak = max(self.peak, len(self.live))
            return proc

        for patch in (mock.patch.object(julia_driver.subprocess, "Popen", fake_popen),
                      mock.patch.object(julia_driver.time, "sleep", lambda _s: None),
                      mock.patch.object(julia_driver, "_available_ram_gb", lambda: self.free_gb)):
            patch.start()
            self.addCleanup(patch.stop)
        path = os.path.join(self.tmp, "julia_gpu.jsonl")
        self.builds = [julia_driver.Build(name, rows, build_path)
                       for name, rows, build_path in julia_driver.build_files(path, plan_julia())]

    def drive(self, jobs):
        data = store.Store(os.path.join(self.tmp, "data"))
        return julia_driver.run_builds(self.builds, "lock", False, jobs, 10.0, data, KEY, "rev")

    def test_at_most_jobs_builds_run_at_once(self):
        self.drive(2)
        self.assertEqual(self.peak, 2)
        self.assertTrue(all(not build.failed for build in self.builds))

    def test_low_ram_serialises_the_spawns(self):
        self.free_gb = 5.0
        self.drive(4)
        self.assertEqual(self.peak, 1)

    def test_the_build_files_hold_one_build_each_in_file_order(self):
        self.assertEqual([build.name for build in self.builds],
                         ["/".join(str(k) for k in key) for key, _ in trials.builds_of(plan_julia())])
        self.assertEqual([os.path.basename(build.path) for build in self.builds],
                         ["julia_gpu.build{0:03d}.jsonl".format(i) for i in range(1, 5)])


class JuliaSideTests(unittest.TestCase):
    """The runner's GPU-free helpers and the errored rule, run under the repo project."""

    def run_julia(self, script):
        from launch import julia_command
        if shutil.which(julia_command()[0]) is None:
            self.skipTest("julia is not on PATH")
        proc = subprocess.run(julia_command() + ["--project=" + ROOT, os.path.join(HERE, script)],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return proc.stdout

    def test_runner_helpers(self):
        self.assertIn("bench_ode_gpu.jl helpers", self.run_julia("test_bench_ode_gpu.jl"))

    def test_errored_rule(self):
        self.assertIn("errored.jl", self.run_julia("test_errored.jl"))


if __name__ == "__main__":
    unittest.main()
