#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
source ./GPU_ODE_JAX/venv/bin/activate

if [ "$ANALYSIS" == "warm" ]; then
    NLIST_CSV=$(echo $NLIST | tr ' ' ',')
    python3 ./GPU_ODE_JAX/bench_diffrax.py "warm:$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

if [ "$ANALYSIS" == "states" ]; then
    # -n (when set) overrides the states-sweep ensemble size.
    STATES_ARG=states
    [ "$NMAX" != "16777216" ] && STATES_ARG="states:$NMAX"
    python3 ./GPU_ODE_JAX/bench_diffrax.py "$STATES_ARG" "$ALGORITHM"
    deactivate
    exit 0
fi

if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_JAX/bench_diffrax.py wp "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

# The whole ascending N sweep runs in one process on kernels compiled once.
NLIST_CSV=$(echo $NLIST | tr ' ' ',')
echo "N sweep = $NLIST_CSV"
python3 ./GPU_ODE_JAX/bench_diffrax.py "$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"

deactivate
