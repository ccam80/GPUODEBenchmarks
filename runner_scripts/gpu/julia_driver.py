"""julia_driver.py --trials <path> [--floor]: one bench_ode_gpu.jl process per build (consecutive lines of one system, algorithm, controller and precision), one process at a time; a watchdog hard exit ends the driver with the same code and the hung build's progress file beside the trial file, naming the builds that crashed before it, for bench.py to abandon and relaunch; exit 1 when a process crashed."""

import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "runner_scripts"))

import trials as trials_mod  # noqa: E402
from launch import check_julia_project, julia_project  # noqa: E402
from protocol import WATCHDOG_EXIT_CODE  # noqa: E402

BENCH = "GPU_ODE_Julia/bench_ode_gpu.jl"


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
    """One build's trial file and the outcome of its process."""

    def __init__(self, name, trial_list, path):
        self.name, self.trials, self.path = name, trial_list, path
        self.failed = False

    def command(self, floor):
        # This interpreter is the suite's; the runner records through the store CLI under it.
        argv = julia_command() + [BENCH, "--trials", self.path, "--store-python", sys.executable]
        if floor:
            argv.append("--floor")
        return argv


def hand_over(build_progress, progress_path, failed):
    """Write the hung build's progress to progress_path with `failed`, the builds that crashed before the hard exit."""
    progress = {}
    try:
        with open(build_progress, encoding="utf-8") as handle:
            progress = json.load(handle)
    except (OSError, ValueError):
        pass
    progress["failed"] = list(failed)
    with open(progress_path, "w", encoding="utf-8") as handle:
        json.dump(progress, handle)


def run_builds(builds, floor, progress_path):
    """Run the builds' processes one after another; WATCHDOG_EXIT_CODE at the first hard exit, its progress handed over to progress_path, else 0. A crash marks its build failed and the next build runs."""
    for build in builds:
        print("spawning {0} ({1} trials, {2})".format(build.name, len(build.trials), os.path.basename(build.path)),
              flush=True)
        code = subprocess.call(build.command(floor), cwd=REPO_ROOT)
        print("{0}: exit {1}".format(build.name, code), flush=True)
        if code == WATCHDOG_EXIT_CODE:
            hand_over(build.path + ".progress", progress_path, [b.name for b in builds if b.failed])
            return code
        if code != 0:
            print("{0}: julia exited {1}".format(build.name, code))
            build.failed = True
    return 0


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
    builds = [Build(name, rows, path) for name, rows, path in build_files(args.trials, trial_list)]
    code = run_builds(builds, args.floor, args.trials + ".progress")
    if code:
        crashed = [build.name for build in builds if build.failed]
        print("julia_gpu: hard exit; bench.py abandons and relaunches from the progress file{0}".format(
            ", carrying {0} crashed build(s)".format(len(crashed)) if crashed else ""))
        for name in crashed:
            print("  failed: " + name)
        return code
    failed = [build.name for build in builds if build.failed]
    print("julia_gpu: {0} builds, {1} failed".format(len(builds), len(failed)))
    for name in failed:
        print("  failed: " + name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
