#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
source ./GPU_ODE_CUBIE_MLIR/venv/bin/activate

# Cubie picks its backend from this at import time.
export CUBIE_CUDA_BACKEND=mlir

if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py wp "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

# The whole ascending N sweep runs in one process so compiled kernels are
# reused across sizes.
NLIST_CSV=$(echo $NLIST | tr ' ' ',')
echo "N sweep = $NLIST_CSV"
python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py "$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"

deactivate
