#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
source ./GPU_ODE_JAX/venv/bin/activate

if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_JAX/bench_diffrax.py 32768 wp
    deactivate
    exit 0
fi

a=8
while [ $a -le $NMAX ]
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_JAX/bench_diffrax.py $a
    a=$((a*4))
done

deactivate
