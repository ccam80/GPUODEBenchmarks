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

if [ "$ANALYSIS" == "warm" ]; then
    # Package precompilation runs in parallel; DiffEqGPU kernels do not persist.
    julia --project=. -e 'using Pkg; Pkg.precompile()'
    echo "DiffEqGPU kernels recompile per process; the julia driver overlaps those compiles."
    exit 0
fi

if [ "$ANALYSIS" == "states" ]; then
    # Parallel compiles, serialized GPU sections; -n overrides the ensemble size.
    if [ "$NMAX" != "16777216" ]; then
        python3 ./runner_scripts/gpu/julia_driver.py states "$ALGORITHM" "$NMAX"
    else
        python3 ./runner_scripts/gpu/julia_driver.py states "$ALGORITHM"
    fi
    exit 0
fi

if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./runner_scripts/gpu/julia_driver.py wp "$ALGORITHM" "$PROBLEM"
    exit 0
fi

NLIST_CSV=$(echo $NLIST | tr ' ' ',')
echo "N sweep = $NLIST_CSV"
python3 ./runner_scripts/gpu/julia_driver.py performance "$NLIST_CSV" "$ALGORITHM" "$PROBLEM"
