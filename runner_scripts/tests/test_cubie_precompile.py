"""cubie_precompile.py: the kernel listing and the driver loop against a fake worker script: chunks of per-worker kernels to at most --jobs workers at once, a worker's exit on a kernel or over its memory budget requeuing the rest of its chunk, failed kernels tallied."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cubie_precompile  # noqa: E402
import trials  # noqa: E402
from protocol import WATCHDOG_EXIT_CODE  # noqa: E402

NAN = float("nan")

# A worker that sleeps through its span, writing the progress file as the real one does; plan.json chooses a kernel to hang, crash, fail or pass its memory budget at, or a span that dies before its first kernel.
FAKE_WORKER = '''
import json, os, sys, time
sys.path.insert(0, r"{runner_scripts}")
import cubie_precompile
from protocol import WATCHDOG_EXIT_CODE
argv = sys.argv[1:]
path = argv[argv.index("--trials") + 1]
start, end = (int(t) for t in argv[argv.index("--worker") + 1].split(":"))
plan = json.load(open(os.path.join(os.path.dirname(path), "plan.json")))
call = {{"span": [start, end], "started": time.time()}}
calls = os.path.join(os.path.dirname(path), "call{{0}}_{{1}}.json".format(start, os.getpid()))
json.dump(call, open(calls, "w"))
if plan.get("die_before") == start:
    sys.exit(9)
progress = {{"under_way": None, "compiled": [], "failed": []}}
def write():
    json.dump(progress, open(cubie_precompile.progress_path(path, start), "w"))
for index in range(start, min(end, plan["count"])):
    progress["under_way"] = index
    write()
    time.sleep(plan.get("sleep", 0.05))
    if plan.get("hang_at") == index:
        sys.exit(WATCHDOG_EXIT_CODE)
    if plan.get("crash_at") == index:
        sys.exit(7)
    if index in plan.get("fail_at", []):
        progress["failed"].append([index, "error: RuntimeError: ptxas failed"])
    else:
        progress["compiled"].append(index)
    progress["under_way"] = None
    if plan.get("full_at") == index and index + 1 < end:
        progress["next"] = index + 1
        write()
        break
    write()
call["ended"] = time.time()
json.dump(call, open(calls, "w"))
sys.exit(0)
'''.format(runner_scripts=os.path.dirname(HERE))


def spec(n=8, **overrides):
    fields = dict(problem="lorenz", system_params="{}", duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0, n=n,
                  grid_dtype="float32", algorithm="tsit5", controller="fixed", dt=2.0 ** -10,
                  dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains="{}", newton_atol=NAN,
                  newton_rtol=NAN, package="cubie", transfers=["both", "none"], finals=False,
                  build="warm", optimize=True, watchdog_s=120.0, set="test")
    fields.update(overrides)
    return fields


class FlagTests(unittest.TestCase):
    def test_flags(self):
        args = cubie_precompile.parse_args(["--trials", "x.jsonl", "--precompile"])
        self.assertEqual((args.jobs, args.per_worker, args.memory_gb, args.worker), (4, 8, 6.0, None))
        args = cubie_precompile.parse_args(["--trials", "x.jsonl", "--precompile", "--jobs", "2", "--per-worker", "3",
                                            "--worker", "3:6"])
        self.assertEqual((args.jobs, args.per_worker, args.worker), (2, 3, (3, 6)))
        self.assertEqual(cubie_precompile.progress_path("t.jsonl", 3), "t.jsonl.precompile3.progress")


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="precompile_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.worker = os.path.join(self.tmp, "fake_worker.py")
        with open(self.worker, "w", encoding="utf-8") as handle:
            handle.write(FAKE_WORKER)
        self.path = os.path.join(self.tmp, "cubie.jsonl")
        self.lines = [dict(spec(8), trial_id=str(i)) for i in range(10)]

    def argv(self, start, end):
        return [sys.executable, self.worker, "--trials", self.path, "--precompile", "--worker", "{0}:{1}".format(start, end)]

    def drive(self, jobs=2, per_worker=3, **plan):
        with open(os.path.join(self.tmp, "plan.json"), "w") as handle:
            json.dump(dict(plan, count=len(self.lines)), handle)
        driver = cubie_precompile.Driver(self.path, self.lines, jobs, per_worker, self.argv)
        status = driver.run()
        names = [n for n in os.listdir(self.tmp) if n.startswith("call")]
        calls = sorted((json.load(open(os.path.join(self.tmp, n))) for n in names), key=lambda c: c["started"])
        return status, driver, calls

    def test_chunks_go_to_at_most_jobs_workers_at_once(self):
        status, driver, calls = self.drive(jobs=2, per_worker=3, sleep=0.15)
        self.assertEqual(status, 0)
        self.assertEqual(sorted(driver.compiled), list(range(10)))
        self.assertEqual((driver.failed, driver.lost), ([], []))
        self.assertEqual(driver.launched, 4)
        spans = [tuple(c["span"]) for c in calls]
        self.assertEqual(spans, [(0, 3), (3, 6), (6, 9), (9, 10)])
        # Two ran together, never three.
        events = [(c["started"], 1) for c in calls] + [(c["ended"], -1) for c in calls if "ended" in c]
        alive, peak = 0, 0
        for _, delta in sorted(events):
            alive += delta
            peak = max(peak, alive)
        self.assertEqual(peak, 2)
        self.assertTrue(all(os.path.isfile(cubie_precompile.progress_path(self.path, s)) for s in (0, 3, 6, 9)))

    def test_a_worker_that_exits_on_a_kernel_hands_the_rest_of_its_chunk_to_a_new_worker(self):
        status, driver, calls = self.drive(jobs=1, per_worker=4, hang_at=5)
        self.assertEqual(status, 0)
        self.assertEqual(driver.lost, [5])
        self.assertEqual(sorted(driver.compiled), [i for i in range(10) if i != 5])
        self.assertEqual([tuple(c["span"]) for c in calls], [(0, 4), (4, 8), (6, 8), (8, 10)])
        self.assertEqual(driver.launched, 4)
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.makedirs(self.tmp)
        with open(self.worker, "w", encoding="utf-8") as handle:
            handle.write(FAKE_WORKER)
        status, driver, calls = self.drive(jobs=1, per_worker=4, crash_at=7)
        self.assertEqual(status, 0)
        self.assertEqual(driver.lost, [7])
        self.assertEqual(sorted(driver.compiled), [i for i in range(10) if i != 7])
        self.assertEqual([tuple(c["span"]) for c in calls], [(0, 4), (4, 8), (8, 10)])
        # A crash on the last kernel of a chunk requeues nothing.
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.makedirs(self.tmp)
        with open(self.worker, "w", encoding="utf-8") as handle:
            handle.write(FAKE_WORKER)
        status, driver, calls = self.drive(jobs=1, per_worker=5, crash_at=9)
        self.assertEqual(driver.lost, [9])
        self.assertEqual(driver.launched, 2)

    def test_a_worker_over_its_memory_budget_hands_the_rest_of_its_chunk_to_a_new_worker(self):
        status, driver, calls = self.drive(jobs=1, per_worker=4, full_at=5)
        self.assertEqual(status, 0)
        self.assertEqual((driver.lost, driver.failed), ([], []))
        self.assertEqual(sorted(driver.compiled), list(range(10)))
        self.assertEqual([tuple(c["span"]) for c in calls], [(0, 4), (4, 8), (6, 8), (8, 10)])
        # A budget passed on the chunk's last kernel requeues nothing.
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.makedirs(self.tmp)
        with open(self.worker, "w", encoding="utf-8") as handle:
            handle.write(FAKE_WORKER)
        status, driver, calls = self.drive(jobs=1, per_worker=5, full_at=4)
        self.assertEqual(sorted(driver.compiled), list(range(10)))
        self.assertEqual(driver.launched, 2)

    def test_private_bytes_reads_this_process(self):
        self.assertGreater(cubie_precompile.private_bytes(), 1 << 20)

    def test_failed_kernels_are_tallied_and_a_chunk_whose_worker_dies_at_once_is_left_to_the_runner(self):
        status, driver, calls = self.drive(jobs=2, per_worker=3, fail_at=[2, 4])
        self.assertEqual(status, 0)
        self.assertEqual(sorted(driver.compiled), [0, 1, 3, 5, 6, 7, 8, 9])
        self.assertEqual(driver.failed, [(2, "error: RuntimeError: ptxas failed"), (4, "error: RuntimeError: ptxas failed")])
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.makedirs(self.tmp)
        with open(self.worker, "w", encoding="utf-8") as handle:
            handle.write(FAKE_WORKER)
        status, driver, calls = self.drive(jobs=2, per_worker=3, die_before=3)
        self.assertEqual(status, 0)
        self.assertEqual((driver.lost, driver.failed), ([], []))
        self.assertEqual(sorted(driver.compiled), [0, 1, 2, 6, 7, 8, 9])
        self.assertEqual(driver.launched, 4)

    def test_main_runs_the_driver_over_the_files_kernels_one_line_each(self):
        trials.write_jsonl(self.path, trials.build_trials([spec(8), spec(32), spec(8, dt=2.0 ** -12),
                                                           spec(8, algorithm="euler")]))
        # A kernel optimizes when any of its lines does.
        lines = {}
        for trial in trials.read_jsonl(self.path):
            line = lines.setdefault(trials.kernel_key(trial), dict(trial))
            line["optimize"] = line["optimize"] or trial["optimize"]
        self.assertEqual([t["optimize"] for t in lines.values()], [True, True, True])
        mixed = trials.build_trials([spec(8, optimize=False), spec(32, optimize=True)])
        self.assertEqual([t["optimize"] for t in mixed], [False, True])
        with open(os.path.join(self.tmp, "plan.json"), "w") as handle:
            json.dump({"count": 3}, handle)
        made = []

        def argv(start, end):
            made.append((start, end))
            return self.argv(start, end)

        status = cubie_precompile.main(["--trials", self.path, "--precompile", "--jobs", "4", "--per-worker", "1"],
                                       "cubie", worker_argv=argv)
        self.assertEqual(status, 0)
        self.assertEqual(made, [(0, 1), (1, 2), (2, 3)])
        trials.write_jsonl(self.path, [])
        self.assertEqual(cubie_precompile.main(["--trials", self.path, "--precompile"], "cubie", worker_argv=argv), 0)
        self.assertEqual(len(made), 3)
        default = cubie_precompile.Driver(self.path, self.lines, 4, 8, None)
        self.assertEqual(default.queue, [(0, 8), (8, 10)])


if __name__ == "__main__":
    unittest.main()
