#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
[ "$ANALYSIS" == "performance" ] || { echo "CPU Julia supports -a performance only" >&2; exit 1; }

path="CPU"
rm -rf "./data/${path}"
mkdir -p "./data/${path}"

a=8
while [ $a -le $NMAX ]
do
    echo "No. of trajectories = $a"
    julia --threads=16 --project="./GPU_ODE_Julia/" ./GPU_ODE_Julia/bench_cpu.jl $a
    a=$((a*4))
done
