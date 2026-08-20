#!/usr/bin/env python

"""Julia leg orchestrator: julia_driver.py performance <N,N,...> [algorithm] [problem] | wp [algorithm] [problem] | states [algorithm]. One process per leg, compiles in parallel under BENCH_JULIA_JOBS (default 4), GPU-timed sections serialized by a pidfile; states adds BENCH_STATES_BUDGET compile kills and NaN backfill."""

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
from wp_common import STATES_GRID, STATES_N, states_outfile  # noqa: E402

BENCH = "GPU_ODE_Julia/bench_ode_gpu.jl"


def _lock_env():
    lock_path = os.path.join(tempfile.gettempdir(), "gpuode_julia_gpu.pid")
    # A lock left by a previous run's killed process would block every child.
    try:
        os.remove(lock_path)
    except OSError:
        pass
    return lock_path


def _run_pool(jobs_args, jobs):
    """Run each command with the GPU lock exported, at most `jobs` at once;
    jobs_args maps a label to its julia argv tail."""
    lock_path = _lock_env()
    pending = list(jobs_args.items())
    running = {}
    while pending or running:
        while pending and len(running) < jobs:
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
            if not problem.runs("julia", algorithm):
                continue
            if not any(algorithm in supported
                       for supported in modes.values()):
                continue
            legs.append((problem.name, algorithm))
    return legs


def run_performance(argv):
    nlist = argv[0]
    request = argv[1] if len(argv) > 1 else "all"
    problem_request = argv[2] if len(argv) > 2 else "all"
    legs = _julia_legs(request, problem_request)
    if not legs:
        print("Julia (DiffEqGPU kernel path) runs none of the requested "
              "legs; skipping.")
        return 0
    jobs = int(os.environ.get("BENCH_JULIA_JOBS", "4"))
    jobs_args = {
        f"{problem} {algorithm}":
            [BENCH, nlist, algorithm, "--problem", problem]
        for problem, algorithm in legs}
    status = 0
    for label, code in _run_pool(jobs_args, jobs):
        if code:
            status = 1
    return status


def run_wp(argv):
    request = argv[0] if argv else "all"
    problem_request = argv[1] if len(argv) > 1 else "all"
    legs = _julia_legs(request, problem_request)
    if not legs:
        print("Julia (DiffEqGPU kernel path) runs none of the requested "
              "legs; skipping.")
        return 0
    jobs = int(os.environ.get("BENCH_JULIA_JOBS", "4"))
    jobs_args = {
        f"wp {problem} {algorithm}":
            [BENCH, "wp", algorithm, "--problem", problem]
        for problem, algorithm in legs}
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
        open(path, "w").close()

    lock_path = _lock_env()
    marker_dir = tempfile.mkdtemp(prefix="gpuode_states_")
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
            size, alg, _, _ = running[other]
            if alg == algorithm and size > nstates:
                other.kill()
                other.wait()
                del running[other]
                print(f"CANCELLED states={size} {algorithm}: {reason}")

    while pending or running:
        while pending and len(running) < jobs:
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
            nstates, algorithm, started, marker = running[proc]
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
                fields = line.split()
                if fields:
                    rows[int(fields[0])] = line.rstrip("\n")
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
