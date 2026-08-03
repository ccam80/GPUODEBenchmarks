#!/bin/bash
set -e
# Activate venv
source ./GPU_ODE_CUBIE_MLIR/venv/bin/activate

# Pin cubie to the MLIR backend (single cubie install, backend chosen at
# import time via this env var).
export CUBIE_CUDA_BACKEND=mlir

# Algorithm filter; "all" runs every algorithm this framework supports.
ALG=${2:-all}

# Work-precision mode: `run_ode_cubie_mlir.sh wp` sweeps dt/tolerance at N=32768.
if [ "$1" == "wp" ]; then
    python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py 32768 wp "$ALG"
    deactivate
    exit 0
fi

a=8
max_a=$1
while [ $a -le $max_a ]
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py $a "$ALG"
    a=$((a*4))
done

# Deactivate venv
deactivate
