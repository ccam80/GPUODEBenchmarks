"""The cubie precompile pass, shared by the CUBIE and CUBIE_MLIR suites: `bench_cubie.py --trials <path> --precompile [--jobs J] [--per-worker K]` compiles every kernel of a trial file, with its optimize candidates, into the package cache before the runners run it. The kernels (one line per kernel_key, file order) go in chunks of K to J worker processes (`--worker START:END`), each exiting after its chunk; a worker that exits on a kernel hands the rest of its chunk to a new one. A kernel that fails to compile is left to the runner's build."""

import argparse
import json
import os
import subprocess
import sys
import time
import timeit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner  # noqa: E402
import trials as trials_mod  # noqa: E402
from bench_key import dataset_key  # noqa: E402
from protocol import OPTIMIZE_SECONDS  # noqa: E402
from wp_common import run_watchdogged  # noqa: E402


def parse_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trials", required=True)
    parser.add_argument("--precompile", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--per-worker", type=int, default=8)
    parser.add_argument("--worker", default="")
    args = parser.parse_args(argv)
    args.worker = tuple(int(tok) for tok in args.worker.split(":")) if args.worker else None
    return args


def progress_path(trials_path, start):
    return "{0}.precompile{1}.progress".format(trials_path, start)


class Worker:
    """One process over kernels[start:end]: each kernel's warm build in the package cache, compiled with its optimize candidates under the optimize watchdog; the progress file carries the kernel under way and the tallies."""

    def __init__(self, package, key, root, lines, span, path, solver_class=None):
        self.package, self.key, self.root = package, key, root
        self.lines, self.span, self.path = lines, span, path
        self.solver_class = solver_class
        self.progress = {"under_way": None, "compiled": [], "failed": []}

    def write_progress(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.progress, handle)

    def compile_one(self, trial):
        import cubie_bench
        build = cubie_bench.Build(self.package, self.key, self.root, trial, cold=False, solver_class=self.solver_class)
        try:
            build.solver.compile(duration=build.duration, optimize_candidates=True, max_parallel=1)
        finally:
            build.close()

    def run(self):
        for index in range(*self.span):
            trial = self.lines[index]
            self.progress["under_way"] = index
            self.write_progress()
            started = timeit.default_timer()

            def breach():
                print("WATCHDOG hard exit: precompile {0} never returned".format(runner.label(trial)), flush=True)

            try:
                run_watchdogged(lambda: self.compile_one(trial), breach, OPTIMIZE_SECONDS)
            except Exception as exc:  # noqa: BLE001 - the runner records the failure when it builds
                reason = runner.failure_reason(runner.classify(exc), exc)
                self.progress["failed"].append([index, reason])
                print("PRECOMPILE FAILED {0}: {1}".format(runner.label(trial), reason), flush=True)
            else:
                self.progress["compiled"].append(index)
                print("precompiled {0} in {1:.1f}s".format(runner.label(trial), timeit.default_timer() - started),
                      flush=True)
            self.progress["under_way"] = None
            self.write_progress()
        return 0


class Driver:
    """The parent: chunks of `per_worker` kernels to at most `jobs` workers at once; a worker that exits on a kernel loses it and the rest of its chunk goes to a new worker."""

    def __init__(self, path, lines, jobs, per_worker, worker_argv):
        self.path, self.lines = path, lines
        self.jobs, self.worker_argv = jobs, worker_argv
        self.queue = [(start, min(start + per_worker, len(lines))) for start in range(0, len(lines), per_worker)]
        self.running = {}
        self.compiled, self.failed, self.lost = [], [], []
        self.launched = 0

    def launch(self, span):
        proc = subprocess.Popen(self.worker_argv(*span), cwd=runner.REPO_ROOT)
        self.running[proc] = span
        self.launched += 1

    def settle(self, proc, code):
        """Tally a finished worker's chunk; the kernel it exited on is lost and the rest requeued."""
        start, end = self.running.pop(proc)
        try:
            with open(progress_path(self.path, start), encoding="utf-8") as handle:
                progress = json.load(handle)
        except (OSError, ValueError):
            progress = {}
        self.compiled += progress.get("compiled", [])
        self.failed += [tuple(entry) for entry in progress.get("failed", [])]
        under_way = progress.get("under_way")
        if under_way is None:
            return
        self.lost.append(under_way)
        print("PRECOMPILE LOST {0}: the worker exited {1}".format(runner.label(self.lines[under_way]), code),
              flush=True)
        if under_way + 1 < end:
            self.queue.insert(0, (under_way + 1, end))

    def run(self):
        started = timeit.default_timer()
        while self.queue or self.running:
            while self.queue and len(self.running) < self.jobs:
                self.launch(self.queue.pop(0))
            finished = [proc for proc in self.running if proc.poll() is not None]
            if not finished:
                time.sleep(0.5)
                continue
            for proc in finished:
                self.settle(proc, proc.returncode)
        print("precompile: {0} kernels, {1} compiled, {2} failed, {3} lost, {4} workers in {5:.0f}s".format(
            len(self.lines), len(self.compiled), len(self.failed), len(self.lost), self.launched,
            timeit.default_timer() - started), flush=True)
        return 0


def main(argv, package, key=None, root=runner.DATA_ROOT, solver_class=None, worker_argv=None):
    """Run the pass over a trial file; a `--worker` invocation compiles its span in this process."""
    args = parse_args(argv)
    first = {}
    for trial in trials_mod.read_jsonl(args.trials):
        first.setdefault(trials_mod.kernel_key(trial), trial)
    lines = list(first.values())
    if args.worker is not None:
        return Worker(package, key or dataset_key(), root, lines, args.worker,
                      progress_path(args.trials, args.worker[0]), solver_class=solver_class).run()
    if worker_argv is None:
        def worker_argv(start, end):
            return [sys.executable, sys.argv[0], "--trials", args.trials, "--precompile",
                    "--worker", "{0}:{1}".format(start, end)]
    return Driver(args.trials, lines, args.jobs, args.per_worker, worker_argv).run()


if __name__ == "__main__":
    sys.exit("cubie_precompile.py is an entry through bench_cubie.py --precompile")
