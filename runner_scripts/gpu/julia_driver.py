"""julia_driver.py --trials <path> [--floor] [--jobs N] [--min-free-gb G]: one bench_ode_gpu.jl process per build (consecutive lines of one system, algorithm, controller and precision), at most --jobs at once above the RAM floor, GPU timing serialised by a pidfile, hard exits abandoned from the progress file and re-run, trials the store's timeouts and OOMs make hopeless abandoned before their process spawns; exit 1 when a process crashed."""

import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "runner_scripts"))

import store  # noqa: E402
import trials as trials_mod  # noqa: E402
from abandon import abandon_after_hard_exit, abandon_from_store  # noqa: E402
from bench_key import dataset_key  # noqa: E402
from protocol import WATCHDOG_EXIT_CODE  # noqa: E402

BENCH = "GPU_ODE_Julia/bench_ode_gpu.jl"
# Result store root; tests point it at a scratch directory.
DATA_ROOT = os.path.join(REPO_ROOT, "data")


def julia_command():
    """The julia launcher as argv: `julia +1.13`, or JULIA when set."""
    return shlex.split(os.environ.get("JULIA", "julia +1.13")) + ["--project=."]


def _available_ram_gb():
    """Free physical memory in GB, 0.0 when unknown."""
    if os.name == "nt":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32),
                        ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]

        stat = MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullAvailPhys / 2 ** 30
        return 0.0
    try:
        return (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                / 2 ** 30)
    except (ValueError, OSError, AttributeError):
        return 0.0


def parse_args(argv):
    p = argparse.ArgumentParser(prog="julia_driver.py")
    p.add_argument("--trials", required=True)
    p.add_argument("--floor", action="store_true")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--min-free-gb", type=float, default=10.0)
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

    def command(self, lock_path, floor):
        # This interpreter is the suite's; the runner records through the store CLI under it.
        argv = julia_command() + [BENCH, "--trials", self.path, "--gpu-lock", lock_path,
                                  "--store-python", sys.executable]
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


def _ram_allows_spawn(running_count, min_free_gb):
    """One kernel compile can take tens of GB; hold spawns while free RAM is below the floor (unknown counts as enough)."""
    if running_count == 0:
        return True
    free = _available_ram_gb()
    return free == 0.0 or free >= min_free_gb


def run_builds(builds, lock_path, floor, jobs, min_free_gb, data, key, suite_rev):
    """Run the builds' processes, at most `jobs` at once while RAM allows, each spawned with the trials the store's failures leave; returns the builds."""
    pending = list(builds)
    running = {}
    while pending or running:
        while pending and len(running) < jobs and _ram_allows_spawn(len(running), min_free_gb):
            build = pending.pop(0)
            if build.retries == 0:
                build.trials = abandon_from_store(data, key, build.trials, suite_rev)
                if not any(t["transfers"] for t in build.trials):
                    print("{0}: every trial abandoned".format(build.name), flush=True)
                    continue
                trials_mod.write_jsonl(build.path, build.trials)
            print("spawning {0} ({1} trials, {2})".format(build.name, len(build.trials), os.path.basename(build.path)),
                  flush=True)
            proc = subprocess.Popen(build.command(lock_path, floor), cwd=REPO_ROOT)
            running[proc] = build
        time.sleep(2)
        for proc in list(running):
            code = proc.poll()
            if code is None:
                continue
            build = running.pop(proc)
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
    status = prepare()
    if status:
        print("julia_gpu: the Julia project could not be instantiated (exit {0})".format(status))
        return 1
    lock_path = args.trials + ".gpulock"
    # A lock left by a previous run's killed process would block every build.
    try:
        os.remove(lock_path)
    except OSError:
        pass
    key = dataset_key()
    data = store.Store(DATA_ROOT)
    suite_rev = store.suite_rev(REPO_ROOT)
    builds = [Build(name, rows, path) for name, rows, path in build_files(args.trials, trial_list)]
    run_builds(builds, lock_path, args.floor, args.jobs, args.min_free_gb, data, key, suite_rev)
    failed = [build.name for build in builds if build.failed]
    hard_exits = sum(build.hard_exits for build in builds)
    print("julia_gpu: {0} builds, {1} hard exit(s), {2} failed".format(len(builds), hard_exits, len(failed)))
    for name in failed:
        print("  failed: " + name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
