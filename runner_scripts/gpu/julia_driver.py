"""julia_driver.py --trials <path> [--floor]: one bench_ode_gpu.jl process per build (consecutive lines of one system, algorithm, controller and precision), one process at a time, hard exits abandoned from the progress file and re-run, trials the store's timeouts and OOMs make hopeless abandoned before their process spawns; exit 1 when a process crashed."""

import argparse
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "runner_scripts"))

import store  # noqa: E402
import trials as trials_mod  # noqa: E402
from abandon import abandon_after_hard_exit, abandon_from_store  # noqa: E402
from bench_key import dataset_key  # noqa: E402
from launch import check_julia_project, julia_project  # noqa: E402
from protocol import WATCHDOG_EXIT_CODE  # noqa: E402

BENCH = "GPU_ODE_Julia/bench_ode_gpu.jl"
# Result store root; tests point it at a scratch directory.
DATA_ROOT = os.path.join(REPO_ROOT, "data")


def julia_command():
    """The julia launcher as argv on the project of launch.julia_project(): `julia +1.13`, or JULIA when set."""
    return shlex.split(os.environ.get("JULIA", "julia +1.13")) + ["--project=" + julia_project()]


def parse_args(argv):
    p = argparse.ArgumentParser(prog="julia_driver.py")
    p.add_argument("--trials", required=True)
    p.add_argument("--floor", action="store_true")
    return p.parse_args(argv)


def build_files(path, trial_list):
    """[(name, trials, file)]: the trial file split per build in file order, written beside it as <stem>.build<k>.jsonl."""
    stem = os.path.splitext(path)[0]
    out = []
    for index, (key, rows) in enumerate(trials_mod.builds_of(trial_list), start=1):
        build_path = "{0}.build{1:03d}.jsonl".format(stem, index)
        trials_mod.write_jsonl(build_path, rows)
        out.append(("/".join(str(k) for k in key), rows, build_path))
    return out


def prepare():
    """Instantiate and precompile the Julia project once, before the builds; the kernel package builds here."""
    return subprocess.call(julia_command() + ["-e", "using Pkg; Pkg.instantiate(); Pkg.precompile()"],
                           cwd=REPO_ROOT)


class Build:
    """One build's queue of trial files: the split file, then a retry file after each hard exit that leaves trials without a row."""

    def __init__(self, name, trial_list, path):
        self.name, self.trials, self.path = name, trial_list, path
        self.stem = os.path.splitext(path)[0]
        self.retries = 0
        self.hard_exits = 0
        self.failed = False

    def command(self, floor):
        # This interpreter is the suite's; the runner records through the store CLI under it.
        argv = julia_command() + [BENCH, "--trials", self.path, "--store-python", sys.executable]
        if floor:
            argv.append("--floor")
        return argv

    def after_exit(self, code, data, key, suite_rev):
        """True when the build has more to run: a hard exit whose abandonment leaves trials without a row is re-queued with a retry file."""
        if code == 0:
            return False
        if code != WATCHDOG_EXIT_CODE:
            print("{0}: julia exited {1}".format(self.name, code))
            self.failed = True
            return False
        self.hard_exits += 1
        remaining = abandon_after_hard_exit(data, key, self.trials, self.path + ".progress", suite_rev)
        if remaining is None:
            print("{0}: hard exit without a progress file".format(self.name))
            self.failed = True
            return False
        if not remaining or self.retries >= len(self.trials):
            return False
        self.retries += 1
        self.trials = remaining
        self.path = "{0}.retry{1}.jsonl".format(self.stem, self.retries)
        trials_mod.write_jsonl(self.path, remaining)
        return True


def run_builds(builds, floor, data, key, suite_rev):
    """Run the builds' processes one after another, each spawned with the trials the store's failures leave; returns the builds."""
    pending = list(builds)
    while pending:
        build = pending.pop(0)
        if build.retries == 0:
            build.trials = abandon_from_store(data, key, build.trials, suite_rev)
            if not any(t["transfers"] for t in build.trials):
                print("{0}: every trial abandoned".format(build.name), flush=True)
                continue
            trials_mod.write_jsonl(build.path, build.trials)
        print("spawning {0} ({1} trials, {2})".format(build.name, len(build.trials), os.path.basename(build.path)),
              flush=True)
        code = subprocess.call(build.command(floor), cwd=REPO_ROOT)
        print("{0}: exit {1}".format(build.name, code), flush=True)
        if build.after_exit(code, data, key, suite_rev):
            pending.insert(0, build)
    return builds


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    trial_list = trials_mod.read_jsonl(args.trials)
    if not any(t["transfers"] for t in trial_list):
        print("julia_gpu: no trials with transfers")
        return 0
    check_julia_project()
    status = prepare()
    if status:
        print("julia_gpu: the Julia project could not be instantiated (exit {0})".format(status))
        return 1
    key = dataset_key()
    data = store.Store(DATA_ROOT)
    suite_rev = store.suite_rev(REPO_ROOT)
    builds = [Build(name, rows, path) for name, rows, path in build_files(args.trials, trial_list)]
    run_builds(builds, args.floor, data, key, suite_rev)
    failed = [build.name for build in builds if build.failed]
    hard_exits = sum(build.hard_exits for build in builds)
    print("julia_gpu: {0} builds, {1} hard exit(s), {2} failed".format(len(builds), hard_exits, len(failed)))
    for name in failed:
        print("  failed: " + name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
