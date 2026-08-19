#!/usr/bin/env python

"""Julia states sweep driver: julia_states_driver.py [algorithm|all] [ensemble N]. One process per (size, algorithm); BENCH_STATES_JOBS concurrent (default 4); BENCH_STATES_BUDGET seconds kills processes with no compiled kernel yet (unset disables)."""

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
from wp_common import STATES_GRID, STATES_N, states_outfile  # noqa: E402


def _succeeded(outfiles, algorithm, nstates):
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


def main(argv):
    os.chdir(REPO_ROOT)
    request = argv[0] if argv else "all"
    ensemble = int(argv[1]) if len(argv) > 1 else STATES_N
    algorithms = resolve_algorithms(request, "julia")
    if not algorithms:
        print("Julia (DiffEqGPU kernel path) runs none of the requested "
              "algorithms; skipping.")
        return 0

    jobs = int(os.environ.get("BENCH_STATES_JOBS", "4"))
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

    lock_path = os.path.join(tempfile.gettempdir(), "gpuode_states_gpu.pid")
    # A lock left by a previous run's killed process would block every child.
    try:
        os.remove(lock_path)
    except OSError:
        pass
    marker_dir = tempfile.mkdtemp(prefix="gpuode_states_")

    pending = [(nstates, algorithm) for nstates in STATES_GRID
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
            # The marker appears once the first kernel compiles, so the
            # optional budget only fells processes still compiling.
            marker = os.path.join(marker_dir, f"{nstates}_{algorithm}.done")
            env = dict(os.environ, BENCH_GPU_LOCK=lock_path,
                       BENCH_STATES_MARKER=marker)
            proc = subprocess.Popen(
                ["julia", "--project=.", "GPU_ODE_Julia/bench_ode_gpu.jl",
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
                if not _succeeded(outfiles, algorithm, nstates):
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
            for nstates in STATES_GRID:
                handle.write(rows.get(nstates,
                                      f"{nstates} nan nan nan") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
