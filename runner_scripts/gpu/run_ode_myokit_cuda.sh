#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
source ./GPU_ODE_MYOKIT_CUDA/venv/bin/activate

# Myokit CUDA exposes float32 forward Euler only, so work-precision is fixed-step.
if [ "$ANALYSIS" == "warm" ]; then
    NLIST_CSV=$(echo $NLIST | tr ' ' ',')
    python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py "warm:$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

if [ "$ANALYSIS" == "states" ]; then
    # -n (when set) is the state-count list or ceiling.
    STATES_ARG=states
    [ "$NMAX_RAW" != "16777216" ] && STATES_ARG="states:$NMAX_RAW"
    python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py "$STATES_ARG" "$ALGORITHM"
    deactivate
    exit 0
fi

if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py wp "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

# The whole ascending N sweep runs in one process on kernels compiled once.
NLIST_CSV=$(echo $NLIST | tr ' ' ',')
echo "N sweep = $NLIST_CSV"
python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py "warm:$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"
python3 ./GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py "$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"

deactivate
