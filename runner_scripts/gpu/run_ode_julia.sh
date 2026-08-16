#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
unset LD_LIBRARY_PATH

if [ "$ANALYSIS" == "work-precision" ]; then
    julia --project=. ./GPU_ODE_Julia/bench_ode_gpu.jl wp "$ALGORITHM" --problem "$PROBLEM"
    exit 0
fi

for a in $NLIST
do
    echo "No. of trajectories = $a"
    julia --project=. ./GPU_ODE_Julia/bench_ode_gpu.jl $a "$ALGORITHM" --problem "$PROBLEM"
done
