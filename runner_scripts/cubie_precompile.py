"""The cubie precompile pass, shared by the CUBIE and CUBIE_MLIR suites: `bench_cubie.py --trials <path> --precompile [--jobs J] [--per-worker K]` compiles every kernel of a trial file, with its optimize candidates, into the package cache before the runners run it. The kernels (one line per kernel_key, file order) go in chunks of K to J worker processes (`--worker START:END`), each exiting after its chunk; a worker that hard-exits or crashes on a kernel hands the rest of its chunk to a new one. Exit 0 once every kernel was attempted (a kernel that fails to compile is reported and left to the runner's build), 1 when a worker died before its first kernel."""

import argparse
import json
import os
import subprocess
import sys
import time
import timeit
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner  # noqa: E402
import trials as trials_mod  # noqa: E402
from bench_key import dataset_key  # noqa: E402
from protocol import OPTIMIZE_SECONDS, WATCHDOG_EXIT_CODE  # noqa: E402
from wp_common import run_watchdogged  # noqa: E402

POLL_S = 0.5


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="bench_cubie.py --precompile", add_help=False)
    parser.add_argument("--trials", required=True)
    parser.add_argument("--precompile", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--per-worker", type=int, default=8)
    parser.add_argument("--worker", default="")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.per_worker < 1:
        parser.error("--jobs and --per-worker take counts of at least 1")
    if args.worker:
        try:
            start, end = (int(tok) for tok in args.worker.split(":"))
        except ValueError:
            parser.error("--worker takes START:END")
        args.worker = (start, end)
    else:
        args.worker = None
    return args


def kernel_label(trial):
    """A kernel's text for the log: problem, algorithm, controller, its step or tolerance and its system_params."""
    text = "{0} {1} {2}".format(trial["problem"], trial["algorithm"], trial["controller"])
    if trial["controller"] == "fixed":
        if trial["dt"] == trial["dt"]:
            text += " dt={0:g}".format(trial["dt"])
    else:
        text += " tol={0:g}".format(trial["atol"])
    if trial["system_params"] not in ("", "{}"):
        text += " " + trial["system_params"]
    return text


def progress_path(trials_path, start):
    return "{0}.precompile{1}.progress".format(trials_path, start)


def chunks(count, size):
    """[(start, end)] over range(count) in pieces of `size`."""
    return [(start, min(start + size, count)) for start in range(0, count, size)]


# ------------------------------------------------------------------ worker

class Worker:
    """One process over kernels[start:end]: each kernel's warm build in the package cache, compiled with its optimize candidates under the optimize watchdog; the progress file carries the kernel under way and the tallies."""

    def __init__(self, package, key, root, lines, span, path, solver_class=None):
        self.package, self.key, self.root = package, key, root
        self.lines, self.span, self.path = lines, span, path
        self.solver_class = solver_class
        self.progress = {"under_way": None, "compiled": [], "failed": []}

    def write_progress(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(dict(self.progress, written_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
                      handle)

    def compile_one(self, trial):
        import cubie_bench
        build = cubie_bench.Build(self.package, self.key, self.root, trial, cold=False, solver_class=self.solver_class)
        try:
            build.solver.compile(duration=build.duration, optimize_candidates=True, max_parallel=1)
        finally:
            build.close()

    def run(self):
        start, end = self.span
        for index in range(start, min(end, len(self.lines))):
            trial = self.lines[index]
            label = kernel_label(trial)
            self.progress["under_way"] = index
            self.write_progress()
            started = timeit.default_timer()

            def breach():
                print("WATCHDOG hard exit: precompile {0} never returned".format(label), flush=True)

            try:
                run_watchdogged(lambda: self.compile_one(trial), breach, OPTIMIZE_SECONDS)
            except Exception as exc:  # noqa: BLE001 - the runner records the failure when it builds
                reason = runner.failure_reason(runner.classify(exc), exc)
                self.progress["failed"].append([index, reason])
                print("PRECOMPILE FAILED {0}: {1}".format(label, reason), flush=True)
            else:
                self.progress["compiled"].append(index)
                print("precompiled {0} in {1:.1f}s".format(label, timeit.default_timer() - started), flush=True)
            self.progress["under_way"] = None
            self.write_progress()
        return 0


# ------------------------------------------------------------------ driver

class Driver:
    """The parent: chunks of `per_worker` kernels to at most `jobs` workers at once; a worker's exit code and progress file decide what its chunk still owes."""

    def __init__(self, path, lines, jobs, per_worker, worker_argv):
        self.path, self.lines = path, lines
        self.jobs, self.per_worker = jobs, per_worker
        self.worker_argv = worker_argv
        self.queue = chunks(len(lines), per_worker)
        self.running = {}
        self.compiled, self.failed, self.hung, self.crashed = [], [], [], []
        self.broken = 0
        self.launched = 0

    def launch(self, span):
        start, end = span
        path = progress_path(self.path, start)
        if os.path.isfile(path):
            os.remove(path)
        print("precompile worker {0}: kernels {1}-{2}".format(self.launched + 1, start + 1, end), flush=True)
        proc = subprocess.Popen(self.worker_argv(start, end), cwd=runner.REPO_ROOT)
        self.running[proc] = (span, path)
        self.launched += 1

    def settle(self, proc, code):
        """Tally a finished worker's chunk and requeue what its death left."""
        (start, end), path = self.running.pop(proc)
        try:
            with open(path, encoding="utf-8") as handle:
                progress = json.load(handle)
        except (OSError, ValueError):
            progress = None
        if progress is None:
            self.broken += 1
            print("precompile worker for kernels {0}-{1} exited {2} before its first kernel".format(
                start + 1, end, code), flush=True)
            return
        self.compiled += [int(i) for i in progress.get("compiled", [])]
        self.failed += [(int(i), reason) for i, reason in progress.get("failed", [])]
        under_way = progress.get("under_way")
        if code == 0 and under_way is None:
            return
        if under_way is None:
            # The chunk finished; the process died on the way out.
            print("precompile worker for kernels {0}-{1} exited {2} after its chunk".format(start + 1, end, code),
                  flush=True)
            return
        under_way = int(under_way)
        label = kernel_label(self.lines[under_way])
        if code == WATCHDOG_EXIT_CODE:
            self.hung.append(under_way)
            print("PRECOMPILE HUNG {0}: the worker hard-exited at the {1:g}s optimize budget".format(
                label, OPTIMIZE_SECONDS), flush=True)
        else:
            self.crashed.append(under_way)
            print("PRECOMPILE CRASHED {0}: the worker exited {1}".format(label, code), flush=True)
        if under_way + 1 < end:
            self.queue.insert(0, (under_way + 1, end))

    def run(self):
        started = timeit.default_timer()
        while self.queue or self.running:
            while self.queue and len(self.running) < self.jobs:
                self.launch(self.queue.pop(0))
            finished = [proc for proc in self.running if proc.poll() is not None]
            if not finished:
                time.sleep(POLL_S)
                continue
            for proc in finished:
                self.settle(proc, proc.returncode)
        elapsed = timeit.default_timer() - started
        print("precompile: {0} kernels, {1} compiled, {2} failed, {3} hung, {4} crashed, {5} workers in {6:.0f}s".format(
            len(self.lines), len(self.compiled), len(self.failed), len(self.hung), len(self.crashed),
            self.launched, elapsed), flush=True)
        for index, reason in self.failed:
            print("  failed: {0}: {1}".format(kernel_label(self.lines[index]), reason))
        for index in self.hung:
            print("  hung: " + kernel_label(self.lines[index]))
        for index in self.crashed:
            print("  crashed: " + kernel_label(self.lines[index]))
        if self.broken:
            print("precompile: {0} worker(s) died before their first kernel".format(self.broken), flush=True)
            return 1
        return 0


def worker_argv_for(script, trials_path):
    """The argv of a worker of this script over a kernel span."""
    def argv(start, end):
        return [sys.executable, script, "--trials", trials_path, "--precompile",
                "--worker", "{0}:{1}".format(start, end)]
    return argv


def main(argv, package, key=None, root=runner.DATA_ROOT, solver_class=None, worker_argv=None):
    """Run the pass over a trial file; a `--worker` invocation compiles its span in this process."""
    args = parse_args(argv)
    trial_list = trials_mod.read_jsonl(args.trials)
    lines = trials_mod.kernel_lines(trial_list)
    if args.worker is not None:
        key = key or dataset_key()
        return Worker(package, key, root, lines, args.worker, progress_path(args.trials, args.worker[0]),
                      solver_class=solver_class).run()
    if not lines:
        print("precompile: no kernels")
        return 0
    return Driver(args.trials, lines, args.jobs, args.per_worker,
                  worker_argv or worker_argv_for(sys.argv[0], args.trials)).run()


if __name__ == "__main__":
    sys.exit("cubie_precompile.py is an entry through bench_cubie.py --precompile")
