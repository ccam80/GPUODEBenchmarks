#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
source ./GPU_ODE_CUBIE_MLIR/venv/bin/activate

# Cubie picks its backend from this at import time.
export CUBIE_CUDA_BACKEND=mlir

if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py 131072 wp "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

for a in $NLIST
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py $a "$ALGORITHM" --problem "$PROBLEM"
done

deactivate
