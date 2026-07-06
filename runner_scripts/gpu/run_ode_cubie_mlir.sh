#!/bin/bash
set -e
# Activate venv
source ./GPU_ODE_CUBIE_MLIR/venv/bin/activate

# Work-precision mode: `run_ode_cubie_mlir.sh wp` sweeps dt/tolerance at N=32768.
if [ "$1" == "wp" ]; then
    python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py 32768 wp
    deactivate
    exit 0
fi

a=8
max_a=$1
while [ $a -le $max_a ]
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py $a
    a=$((a*4))
done

# Deactivate venv
deactivate
