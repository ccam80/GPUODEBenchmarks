#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
source ./GPU_ODE_CUBIE_MLIR/venv/bin/activate

# Cubie picks its backend from this at import time.
export CUBIE_CUDA_BACKEND=mlir

if [ "$ANALYSIS" == "warm" ]; then
    NLIST_CSV=$(echo $NLIST | tr ' ' ',')
    python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py "warm:$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

if [ "$ANALYSIS" == "states" ]; then
    # -n (when set) overrides the states-sweep ensemble size.
    STATES_ARG=states
    [ "$NMAX" != "16777216" ] && STATES_ARG="states:$NMAX"
    python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py "$STATES_ARG" "$ALGORITHM"
    deactivate
    exit 0
fi

if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py wp "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

# The whole ascending N sweep runs in one process on kernels compiled once.
NLIST_CSV=$(echo $NLIST | tr ' ' ',')
echo "N sweep = $NLIST_CSV"
python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py "warm:$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"
python3 ./GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py "$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"

deactivate
