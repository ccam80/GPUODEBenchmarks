#!/usr/bin/env python

"""Julia leg orchestrator: julia_driver.py performance <N,N,...> [algorithm] [problem] | wp [algorithm] [problem] | states [algorithm]. One process per (problem, algorithm, mode) leg, compiles in parallel under BENCH_JULIA_JOBS (default 4) while free host RAM stays above BENCH_JULIA_MIN_FREE_GB (default 10), GPU-timed sections serialized by a pidfile; states adds BENCH_STATES_BUDGET compile kills and NaN backfill."""

import math
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "runner_scripts"))

from algorithms import resolve_algorithms, supported_for  # noqa: E402
from bench_key import dataset_key  # noqa: E402
from problems import resolve_problems  # noqa: E402
from resume import (  # noqa: E402
    active as resume_active,
    floor_enabled,
    skip_point,
    skip_wp_leg,
)
from wp_common import (  # noqa: E402
    STATES_GRID,
    STATES_N,
    states_outfile,
    times_outfile,
    wp_outfile,
)

BENCH = "GPU_ODE_Julia/bench_ode_gpu.jl"


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
            proc = subprocess.Popen(["julia", "--project=."] + args,
                                    cwd=REPO_ROOT, env=env)
            running[proc] = label
        time.sleep(2)
        for proc in list(running):
            code = proc.poll()
            if code is not None:
                label = running.pop(proc)
                print(f"{label}: exit {code}")
                yield label, code


def _julia_legs(request, problem_request):
    """(problem, algorithm) pairs julia actually runs."""
    algorithms = resolve_algorithms(request, "julia")
    problems = resolve_problems(problem_request, "julia")
    modes = {"fixed": supported_for("julia", "fixed"),
             "adaptive": supported_for("julia", "adaptive")}
    legs = []
    for problem in problems:
        for algorithm in algorithms:
            if not problem.supports("julia"):
                continue
            if not any(algorithm in supported
                       for supported in modes.values()):
                continue
            legs.append((problem.name, algorithm))
    return legs


def _modes_for(algorithm):
    """The (mode, ...) tuple this algorithm runs under julia, fixed first."""
    return tuple(mode for mode in ("fixed", "adaptive")
                 if algorithm in supported_for("julia", mode))


def _mode_legs(request, problem_request):
    """(problem, algorithm, mode) legs, one process each."""
    return [(problem, algorithm, mode)
            for problem, algorithm in _julia_legs(request, problem_request)
            for mode in _modes_for(algorithm)]


def _prune_covered(legs, pending):
    """Drop the (problem, algorithm, mode) legs whose every point is covered."""
    if not resume_active():
        return legs
    kept = []
    for leg in legs:
        if pending(*leg):
            kept.append(leg)
        else:
            print(f"-- resume: skipping {' '.join(leg)} (already covered)")
    return kept


def run_performance(argv):
    nlist = argv[0]
    request = argv[1] if len(argv) > 1 else "all"
    problem_request = argv[2] if len(argv) > 2 else "all"
    ns = sorted(int(tok) for tok in nlist.split(","))
    key = dataset_key()

    def pending(problem, algorithm, mode):
        outfile = times_outfile("Julia", "Julia", mode, algorithm, key,
                                problem)
        return any(not skip_point(problem, algorithm, mode, n, outfile)
                   for n in ns)

    legs = _prune_covered(_mode_legs(request, problem_request), pending)
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
    request = argv[0] if argv else "all"
    problem_request = argv[1] if len(argv) > 1 else "all"
    key = dataset_key()

    def pending(problem, algorithm, mode):
        outfile = wp_outfile("Julia", "Julia", mode, algorithm, key,
                             problem)
        return not skip_wp_leg(problem, algorithm, mode, outfile)

    legs = _prune_covered(_mode_legs(request, problem_request), pending)
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


def _states_succeeded(outfiles, algorithm, nstates):
    """True when any mode recorded a finite time for this size."""
    for (mode, alg), path in outfiles.items():
        if alg != algorithm:
            continue
        try:
            with open(path) as handle:
                for line in handle:
                    fields = line.split()
                    if (len(fields) >= 2 and fields[0] == str(nstates)
                            and not math.isnan(float(fields[1]))):
                        return True
        except (OSError, ValueError):
            continue
    return False


def run_states(argv):
    request = argv[0] if argv else "all"
    grid = list(STATES_GRID)
    ensemble = STATES_N
    algorithms = resolve_algorithms(request, "julia")
    if not algorithms:
        print("Julia (DiffEqGPU kernel path) runs none of the requested "
              "algorithms; skipping.")
        return 0

    jobs = int(os.environ.get("BENCH_JULIA_JOBS", "4"))
    budget = float(os.environ.get("BENCH_STATES_BUDGET", "0"))
    key = dataset_key()
    modes = {"fixed": supported_for("julia", "fixed"),
             "adaptive": supported_for("julia", "adaptive")}
    legs = [(mode, algorithm) for algorithm in algorithms
            for mode, supported in modes.items() if algorithm in supported]
    outfiles = {leg: states_outfile("Julia", "Julia", leg[0], leg[1], key)
                for leg in legs}
    for path in outfiles.values():
        # A resumed or --floor run keeps the recorded rows; a fresh one
        # starts clean.
        open(path, "a" if resume_active() or floor_enabled()
             else "w").close()

    lock_path = _lock_env()
    marker_dir = tempfile.mkdtemp(prefix="gpuode_states_")
    pending = [(nstates, algorithm) for nstates in grid
               for algorithm in algorithms]
    if resume_active():
        def covered(nstates, algorithm):
            return all(skip_point("lorenz96", algorithm, mode, nstates,
                                  outfiles[(mode, algorithm)])
                       for mode in _modes_for(algorithm))
        for nstates, algorithm in [pair for pair in pending
                                   if covered(*pair)]:
            pending.remove((nstates, algorithm))
            print(f"-- resume: skipping states={nstates} {algorithm} "
                  "(already covered)")
    running = {}

    def cancel_larger(algorithm, nstates, reason):
        """A failed size dooms the larger ones of the same algorithm."""
        for size, alg in list(pending):
            if alg == algorithm and size > nstates:
                pending.remove((size, alg))
                print(f"CANCELLED states={size} {algorithm}: {reason}")
        for other in list(running):
            size, alg, _, _ = running[other]
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
            # The marker marks first-kernel compile; the budget kills only markerless processes.
            marker = os.path.join(marker_dir, f"{nstates}_{algorithm}.done")
            env = dict(os.environ, BENCH_GPU_LOCK=lock_path,
                       BENCH_STATES_MARKER=marker)
            proc = subprocess.Popen(
                ["julia", "--project=.", BENCH,
                 f"states:{nstates}:{ensemble}", algorithm],
                cwd=REPO_ROOT, env=env)
            running[proc] = (nstates, algorithm, time.monotonic(), marker)
        time.sleep(2)
        for proc in list(running):
            state = running.get(proc)
            if state is None:
                # cancel_larger removed it while this snapshot was polled.
                continue
            nstates, algorithm, started, marker = state
            code = proc.poll()
            if code is not None:
                del running[proc]
                print(f"states={nstates} {algorithm}: exit {code}")
                if not _states_succeeded(outfiles, algorithm, nstates):
                    cancel_larger(algorithm, nstates,
                                  f"states={nstates} produced no result")
            elif (budget > 0 and time.monotonic() - started > budget
                  and not os.path.exists(marker)):
                proc.kill()
                proc.wait()
                del running[proc]
                print(f"BUDGET states={nstates} {algorithm}: no compile "
                      f"within {budget:.0f}s, killed")
                cancel_larger(algorithm, nstates, "compile budget breached")

    # Rows a killed or crashed process never wrote become NaN, sorted by size.
    for (mode, algorithm), path in outfiles.items():
        rows = {}
        with open(path) as handle:
            for line in handle:
                # Keep only complete `states t_ms t_dev_ms build_s` rows.
                fields = line.split()
                if len(fields) < 4:
                    continue
                try:
                    rows[int(fields[0])] = line.rstrip("\n")
                except ValueError:
                    continue
        with open(path, "w") as handle:
            for nstates in grid:
                handle.write(rows.get(nstates,
                                      f"{nstates} nan nan nan") + "\n")
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
