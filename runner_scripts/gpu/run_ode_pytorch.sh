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
    # One process per (problem, algorithm) leg, so a watchdog
    # hard-exit (status 3) abandons that leg, not the sweep.
    WP_STATUS=0
    while read -r WP_PROB WP_ALG; do
        [ -n "$WP_PROB" ] || continue
        python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py wp "$WP_ALG" \
            --problem "$WP_PROB" && WP_RC=0 || WP_RC=$?
        case "$WP_RC" in
            0) ;;
            3) echo "wp $WP_PROB $WP_ALG: watchdog hard-exit, leg abandoned";;
            *) echo "wp $WP_PROB $WP_ALG: exit $WP_RC"; WP_STATUS=1;;
        esac
    done < <(python3 runner_scripts/gpu/wp_legs.py pytorch "$ALGORITHM" "$PROBLEM")
    deactivate
    exit $WP_STATUS
fi

# The whole ascending N sweep runs in one process on kernels compiled once.
NLIST_CSV=$(echo $NLIST | tr ' ' ',')
echo "N sweep = $NLIST_CSV"
python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py "$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"

deactivate
