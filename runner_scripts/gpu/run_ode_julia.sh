#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
unset LD_LIBRARY_PATH

if [ "$ANALYSIS" == "work-precision" ]; then
    julia --project=. ./GPU_ODE_Julia/bench_lorenz_gpu.jl 32768 wp "$ALGORITHM"
    exit 0
fi

for a in $NLIST
do
    echo "No. of trajectories = $a"
    julia --project=. ./GPU_ODE_Julia/bench_lorenz_gpu.jl $a "$ALGORITHM"
done
