#!/usr/bin/env python

"""Julia leg orchestrator: julia_driver.py performance <N,N,...> [algorithm] [problem] | wp [algorithm] [problem] | states [algorithm], each with [--mode <fixed|adaptive|all>]. One process per (problem, algorithm, mode) leg, compiles in parallel under BENCH_JULIA_JOBS (default 4) while free host RAM stays above BENCH_JULIA_MIN_FREE_GB (default 10), GPU-timed sections serialized by a pidfile; states backfills NaN rows for processes that never wrote them."""

import os
import shlex
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "runner_scripts"))

from algorithms import resolve_algorithms, resolve_modes, supported_for  # noqa: E402
from bench_key import dataset_key  # noqa: E402
from problems import STATES_PROBLEM, resolve_problems  # noqa: E402
from protocol import STATES_GRID, STATES_N  # noqa: E402
from results import Leg  # noqa: E402

BENCH = "GPU_ODE_Julia/bench_ode_gpu.jl"


def julia_command():
    """The julia launcher as argv: `julia +1.13`, or JULIA when set."""
    return shlex.split(os.environ.get("JULIA", "julia +1.13")) + ["--project=."]

# Result store root; tests point it at a scratch directory.
DATA_ROOT = None


def _leg(analysis, problem, algorithm, mode):
    return Leg("julia", dataset_key(), analysis, problem, algorithm, mode,
               root=DATA_ROOT)


def _lock_env():
    lock_path = os.path.join(tempfile.gettempdir(), "gpuode_julia_gpu.pid")
    # A lock left by a previous run's killed process would block every child.
    try:
        os.remove(lock_path)
    except OSError:
        pass
    return lock_path


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


def _ram_allows_spawn(running_count):
    """One kernel compile can take tens of GB; hold spawns while RAM is low."""
    if running_count == 0:
        return True
    floor = float(os.environ.get("BENCH_JULIA_MIN_FREE_GB", "10"))
    free = _available_ram_gb()
    return free == 0.0 or free >= floor


def _run_pool(jobs_args, jobs):
    """Run each command with the GPU lock exported, at most `jobs` at once;
    jobs_args maps a label to its julia argv tail."""
    lock_path = _lock_env()
    pending = list(jobs_args.items())
    running = {}
    while pending or running:
        while (pending and len(running) < jobs
               and _ram_allows_spawn(len(running))):
            label, args = pending.pop(0)
            print(f"spawning {label}")
            env = dict(os.environ, BENCH_GPU_LOCK=lock_path)
            proc = subprocess.Popen(julia_command() + args, cwd=REPO_ROOT,
                                    env=env)
            running[proc] = label
        time.sleep(2)
        for proc in list(running):
            code = proc.poll()
            if code is not None:
                label = running.pop(proc)
                print(f"{label}: exit {code}")
                yield label, code


def _split_mode(argv):
    """(argv without the --mode pair, the requested modes)."""
    argv = list(argv)
    request = "all"
    if "--mode" in argv:
        position = argv.index("--mode")
        if position + 1 >= len(argv):
            raise SystemExit("--mode requires a value")
        request = argv[position + 1]
        del argv[position:position + 2]
    return argv, resolve_modes(request)


def _modes_for(algorithm, modes):
    """The modes this algorithm runs under julia among the requested ones, fixed first."""
    return tuple(mode for mode in modes
                 if algorithm in supported_for("julia_gpu", mode))


def _mode_legs(request, problem_request, modes):
    """(problem, algorithm, mode) legs, one process each."""
    algorithms = resolve_algorithms(request, "julia_gpu")
    return [(problem.name, algorithm, mode)
            for problem in resolve_problems(problem_request, "julia_gpu")
            for algorithm in algorithms
            for mode in _modes_for(algorithm, modes)]


def run_performance(argv):
    argv, modes = _split_mode(argv)
    nlist = argv[0]
    request = argv[1] if len(argv) > 1 else "all"
    problem_request = argv[2] if len(argv) > 2 else "all"

    legs = _mode_legs(request, problem_request, modes)
    if not legs:
        print("Julia (DiffEqGPU kernel path) runs none of the requested "
              "legs; skipping.")
        return 0
    jobs = int(os.environ.get("BENCH_JULIA_JOBS", "4"))
    jobs_args = {
        f"{problem} {algorithm} {mode}":
            [BENCH, nlist, algorithm, "--problem", problem, "--mode", mode]
        for problem, algorithm, mode in legs}
    status = 0
    for label, code in _run_pool(jobs_args, jobs):
        if code:
            status = 1
    return status


def run_wp(argv):
    argv, modes = _split_mode(argv)
    request = argv[0] if argv else "all"
    problem_request = argv[1] if len(argv) > 1 else "all"

    legs = _mode_legs(request, problem_request, modes)
    if not legs:
        print("Julia (DiffEqGPU kernel path) runs none of the requested "
              "legs; skipping.")
        return 0
    jobs = int(os.environ.get("BENCH_JULIA_JOBS", "4"))
    jobs_args = {
        f"wp {problem} {algorithm} {mode}":
            [BENCH, "wp", algorithm, "--problem", problem, "--mode", mode]
        for problem, algorithm, mode in legs}
    status = 0
    for label, code in _run_pool(jobs_args, jobs):
        if code:
            status = 1
    return status


def _states_succeeded(legs, algorithm, nstates):
    """True when any mode recorded a finite time for this size."""
    return any(leg.status(STATES_N, nstates) == "finite"
               for (mode, alg), leg in legs.items() if alg == algorithm)


def run_states(argv):
    argv, modes = _split_mode(argv)
    request = argv[0] if argv else "all"
    grid = list(STATES_GRID)
    ensemble = STATES_N
    algorithms = [name for name in resolve_algorithms(request, "julia_gpu")
                  if _modes_for(name, modes)]
    if not algorithms:
        print("Julia (DiffEqGPU kernel path) runs none of the requested "
              "algorithms; skipping.")
        return 0

    jobs = int(os.environ.get("BENCH_JULIA_JOBS", "4"))
    mode_arg = ["--mode", ",".join(modes)] if len(modes) < 2 else []
    legs = {(mode, algorithm): _leg("states", STATES_PROBLEM, algorithm, mode)
            for algorithm in algorithms
            for mode in _modes_for(algorithm, modes)}

    lock_path = _lock_env()
    pending = [(nstates, algorithm) for nstates in grid
               for algorithm in algorithms]
    running = {}

    def cancel_larger(algorithm, nstates, reason):
        """A failed size dooms the larger ones of the same algorithm."""
        for size, alg in list(pending):
            if alg == algorithm and size > nstates:
                pending.remove((size, alg))
                print(f"CANCELLED states={size} {algorithm}: {reason}")
        for other in list(running):
            size, alg = running[other]
            if alg == algorithm and size > nstates:
                other.kill()
                other.wait()
                del running[other]
                print(f"CANCELLED states={size} {algorithm}: {reason}")

    while pending or running:
        while (pending and len(running) < jobs
               and _ram_allows_spawn(len(running))):
            nstates, algorithm = pending.pop(0)
            print(f"spawning lorenz96 states={nstates} {algorithm} "
                  f"(N={ensemble})")
            env = dict(os.environ, BENCH_GPU_LOCK=lock_path)
            proc = subprocess.Popen(
                julia_command() + [BENCH, f"states:{nstates}:{ensemble}",
                                   algorithm] + mode_arg,
                cwd=REPO_ROOT, env=env)
            running[proc] = (nstates, algorithm)
        time.sleep(2)
        for proc in list(running):
            state = running.get(proc)
            if state is None:
                # cancel_larger removed it while this snapshot was polled.
                continue
            nstates, algorithm = state
            code = proc.poll()
            if code is not None:
                del running[proc]
                print(f"states={nstates} {algorithm}: exit {code}")
                if not _states_succeeded(legs, algorithm, nstates):
                    cancel_larger(algorithm, nstates,
                                  f"states={nstates} produced no result")

    # Rows a killed or crashed process never wrote become NaN.
    for leg in legs.values():
        leg.nan_states([nstates for nstates in grid
                        if leg.status(ensemble, nstates) == "absent"])
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "performance":
        sys.exit(run_performance(sys.argv[2:]))
    if mode == "wp":
        sys.exit(run_wp(sys.argv[2:]))
    if mode == "states":
        sys.exit(run_states(sys.argv[2:]))
    raise SystemExit("usage: julia_driver.py performance|wp|states ...")
