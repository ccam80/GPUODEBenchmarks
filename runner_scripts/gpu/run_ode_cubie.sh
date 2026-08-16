#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
source ./GPU_ODE_CUBIE/venv/bin/activate

# The venv is shared with the MLIR suite; cubie picks its backend from this at import time.
export CUBIE_CUDA_BACKEND=numba-cuda

if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_CUBIE/bench_cubie.py 131072 wp "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

for a in $NLIST
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_CUBIE/bench_cubie.py $a "$ALGORITHM" --problem "$PROBLEM"
done

deactivate
