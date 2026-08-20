#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
source ./GPU_ODE_PyTorch/venv/bin/activate

# Fixed-step only: torchdiffeq adaptive solvers are incompatible with torch.vmap.
if [ "$ANALYSIS" == "warm" ]; then
    NLIST_CSV=$(echo $NLIST | tr ' ' ',')
    python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py "warm:$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

if [ "$ANALYSIS" == "states" ]; then
    python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py states "$ALGORITHM"
    deactivate
    exit 0
fi

if [ "$ANALYSIS" == "work-precision" ]; then
    python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py wp "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

# The whole ascending N sweep runs in one process on kernels compiled once.
NLIST_CSV=$(echo $NLIST | tr ' ' ',')
echo "N sweep = $NLIST_CSV"
python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py "$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"

deactivate
