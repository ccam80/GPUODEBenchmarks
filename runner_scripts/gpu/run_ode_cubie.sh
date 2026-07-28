#!/bin/bash
set -e
# Activate venv
source ./GPU_ODE_CUBIE/venv/bin/activate

# Pin cubie to the stock numba-cuda backend. The venv is shared with the
# CUBIE_MLIR suite and holds both backends, so state it rather than relying
# on the default (backend is chosen at import time from this env var).
export CUBIE_CUDA_BACKEND=numba-cuda

# Work-precision mode: `run_ode_cubie.sh wp` sweeps dt/tolerance at N=32768.
if [ "$1" == "wp" ]; then
    python3 ./GPU_ODE_CUBIE/bench_cubie.py 32768 wp
    deactivate
    exit 0
fi

a=8
max_a=$1
while [ $a -le $max_a ]
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_CUBIE/bench_cubie.py $a
    a=$((a*4))
done

# Deactivate venv
deactivate
