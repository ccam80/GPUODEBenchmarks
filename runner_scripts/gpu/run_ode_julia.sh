#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
unset LD_LIBRARY_PATH

# One julia process per (problem, algorithm) runs that pair's whole N sweep.
ALGO_LIST=$(python3 ./runner_scripts/algorithms.py julia "$ALGORITHM")
if [ -z "$ALGO_LIST" ]; then
    echo "Julia (DiffEqGPU kernel path) runs none of the requested algorithms; skipping."
    exit 0
fi

if [ "$ANALYSIS" == "states" ]; then
    # Parallel compiles, serialized GPU sections; -n overrides the ensemble size.
    if [ "$NMAX" != "16777216" ]; then
        python3 ./runner_scripts/gpu/julia_states_driver.py "$ALGORITHM" "$NMAX"
    else
        python3 ./runner_scripts/gpu/julia_states_driver.py "$ALGORITHM"
    fi
    exit 0
fi

if [ "$ANALYSIS" == "work-precision" ]; then
    for g in $ALGO_LIST
    do
        julia --project=. ./GPU_ODE_Julia/bench_ode_gpu.jl wp "$g" --problem "$PROBLEM"
    done
    exit 0
fi

PROBLEM_LIST=$(python3 ./runner_scripts/problems.py julia "$PROBLEM")
if [ -z "$PROBLEM_LIST" ]; then
    echo "Julia runs none of the requested problems; skipping."
    exit 0
fi

NLIST_CSV=$(echo $NLIST | tr ' ' ',')
for p in $PROBLEM_LIST
do
    for g in $ALGO_LIST
    do
        echo "Problem $p, algorithm $g, N sweep = $NLIST_CSV"
        julia --project=. ./GPU_ODE_Julia/bench_ode_gpu.jl "$NLIST_CSV" "$g" --problem "$p"
    done
done
