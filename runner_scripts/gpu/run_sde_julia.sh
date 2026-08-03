#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
[ "$ANALYSIS" == "performance" ] || { echo "GPU Julia SDE supports -a performance only" >&2; exit 1; }

path="SDE"
rm -rf "./data/${path}"
mkdir -p "./data/${path}"

a=8
while [ $a -le $NMAX ]
do
    echo "No. of trajectories = $a"
    julia --threads=16 --project="./GPU_ODE_Julia/" ./GPU_ODE_Julia/sde_examples/bench_gpu.jl $a
    a=$((a*4))
done
