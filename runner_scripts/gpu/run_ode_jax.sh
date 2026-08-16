#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
source ./GPU_ODE_JAX/venv/bin/activate

if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_JAX/bench_diffrax.py 131072 wp "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

for a in $NLIST
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_JAX/bench_diffrax.py $a "$ALGORITHM" --problem "$PROBLEM"
done

deactivate
