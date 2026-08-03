#!/bin/bash
set -e

source ./GPU_ODE_MYOKIT_CUDA/venv/bin/activate

# Algorithm request (issue #29): forwarded to the bench script, which runs
# every supported algorithm for "all" and skips cleanly when unsupported.
ALG=${2:-all}

# Myokit CUDA exposes float32 forward Euler only. Its work-precision mode
# therefore writes only the fixed-step sweep.
if [ "$1" == "wp" ]; then
    python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py 32768 wp "$ALG"
    deactivate
    exit 0
fi

a=8
max_a=$1
while [ $a -le $max_a ]
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py "$a" "$ALG"
    a=$((a*4))
done

deactivate
