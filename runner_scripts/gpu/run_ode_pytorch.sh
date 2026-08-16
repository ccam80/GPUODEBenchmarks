#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
source ./GPU_ODE_PyTorch/venv/bin/activate

# Fixed-step only: torchdiffeq adaptive solvers are incompatible with torch.vmap.
if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py 131072 wp "$ALGORITHM"
    deactivate
    exit 0
fi

for a in $NLIST
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py $a "$ALGORITHM"
done

deactivate
