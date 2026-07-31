#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
source ./GPU_ODE_MYOKIT_CUDA/venv/bin/activate

# Myokit CUDA exposes float32 forward Euler only, so work-precision is fixed-step.
if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py 32768 wp
    deactivate
    exit 0
fi

a=8
while [ $a -le $NMAX ]
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py "$a"
    a=$((a*4))
done

deactivate
