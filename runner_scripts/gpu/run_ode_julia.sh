#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
unset LD_LIBRARY_PATH

# One julia process per algorithm, so a watchdog exit only abandons that leg.
ALGO_LIST=$(python3 ./runner_scripts/algorithms.py julia "$ALGORITHM")
if [ -z "$ALGO_LIST" ]; then
    echo "Julia (DiffEqGPU kernel path) runs none of the requested algorithms; skipping."
    exit 0
fi

if [ "$ANALYSIS" == "work-precision" ]; then
    for g in $ALGO_LIST
    do
        julia --project=. ./GPU_ODE_Julia/bench_ode_gpu.jl wp "$g" --problem "$PROBLEM"
    done
    exit 0
fi

for a in $NLIST
do
    echo "No. of trajectories = $a"
    for g in $ALGO_LIST
    do
        julia --project=. ./GPU_ODE_Julia/bench_ode_gpu.jl $a "$g" --problem "$PROBLEM"
    done
done
