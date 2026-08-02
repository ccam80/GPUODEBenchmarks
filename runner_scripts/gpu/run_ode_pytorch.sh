#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
source ./GPU_ODE_PyTorch/venv/bin/activate

# Fixed-step only: torchdiffeq adaptive solvers are incompatible with torch.vmap.
if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py 32768 wp
    deactivate
    exit 0
fi

a=8
while [ $a -le $NMAX ]
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py $a
    a=$((a*4))
done

deactivate
