#!/bin/bash
set -e

source ./GPU_ODE_MYOKIT_CUDA/venv/bin/activate

# Myokit CUDA exposes float32 forward Euler only. Its work-precision mode
# therefore writes only the fixed-step sweep.
if [ "$1" == "wp" ]; then
    python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py 32768 wp
    deactivate
    exit 0
fi

a=8
max_a=$1
while [ $a -le $max_a ]
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py "$a"
    a=$((a*4))
done

deactivate
