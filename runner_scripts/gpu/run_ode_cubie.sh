#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"
source ./GPU_ODE_CUBIE/venv/bin/activate

# The suite holds ~100 kernels per system; the default LRU cap of 10 evicts them.
export CUBIE_MAX_CACHE_ENTRIES=0
BENCH="python3 ./GPU_ODE_CUBIE/bench_cubie.py"

if [ "$ANALYSIS" == "optimize" ]; then
    $BENCH optimize "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

if [ "$ANALYSIS" == "warm" ]; then
    NLIST_CSV=$(echo $NLIST | tr ' ' ',')
    $BENCH "warm:$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

if [ "$ANALYSIS" == "states" ]; then
    $BENCH states "$ALGORITHM"
    deactivate
    exit 0
fi

if [ "$ANALYSIS" == "work-precision" ]; then
    $BENCH optimize "$ALGORITHM" --problem "$PROBLEM"
    $BENCH wp "$ALGORITHM" --problem "$PROBLEM"
    deactivate
    exit 0
fi

# Optimize, warm the tuned kernels, then walk the ascending N sweep in one process.
NLIST_CSV=$(echo $NLIST | tr ' ' ',')
echo "N sweep = $NLIST_CSV"
$BENCH optimize "$ALGORITHM" --problem "$PROBLEM"
$BENCH "warm:$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"
$BENCH "$NLIST_CSV" "$ALGORITHM" --problem "$PROBLEM"

deactivate
