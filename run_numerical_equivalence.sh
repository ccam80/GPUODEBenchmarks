#!/bin/bash
# Run the full numerical-equivalence (ne) suite: golden reference (if
# missing), DifferentialEquations.jl Float32 reference sweeps, cubie Float32
# sweeps, and the comparison report + plots.
#
# Usage: ./run_numerical_equivalence.sh [fixed|adaptive|all]
#   fixed    - error-vs-dt convergence sweeps only
#   adaptive - error-vs-tolerance sweeps only (default + matched controllers)
#   all      - both (default)
#
# Exit code: non-zero if any step fails or the comparison finds a MISMATCH /
# DIVERGENT algorithm (compare_numerical_equivalence.py exits 2) — suitable
# as a CI gate. The Julia outputs are machine independent; if
# data/numerical_equivalence/julia/ is committed and Julia is unavailable,
# run the last two steps by hand instead.

# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1

mode="${1:-all}"
case "$mode" in
    fixed|adaptive|all) ;;
    *) echo "Usage: $0 [fixed|adaptive|all]"; exit 1;;
esac

echo "========================================="
echo "Numerical-equivalence suite (mode: $mode)"
echo "========================================="

if [ ! -f ./data/numerical/golden_ne_lorenz_1024.csv ]; then
    echo "--- Generating golden reference (Float64 Vern9, machine independent) ---"
    julia -t auto --project=. ./runner_scripts/numerical_equivalence/generate_golden_ne.jl || {
        echo "✗ golden generation failed"; exit 1; }
fi

echo "--- DifferentialEquations.jl Float32 sweeps (CPU, machine independent) ---"
julia -t auto --project=. ./runner_scripts/numerical_equivalence/ne_diffeq.jl "$mode" || {
    echo "✗ DifferentialEquations.jl sweeps failed"; exit 1; }

echo "--- cubie Float32 sweeps (GPU, keyed per machine) ---"
if [ -x ./GPU_ODE_CUBIE/venv/bin/python3 ] || [ -f ./GPU_ODE_CUBIE/venv/bin/python3 ]; then
    PYTHON=./GPU_ODE_CUBIE/venv/bin/python3
elif [ -f ./GPU_ODE_CUBIE/venv/Scripts/python.exe ]; then
    PYTHON=./GPU_ODE_CUBIE/venv/Scripts/python.exe
else
    echo "✗ GPU_ODE_CUBIE venv not found; run setup_all_environments.py first"
    exit 1
fi
"$PYTHON" ./GPU_ODE_CUBIE/numerical_equivalence.py "$mode" || {
    echo "✗ cubie sweeps failed"; exit 1; }

echo "--- Comparison report + plots ---"
"$PYTHON" compare_numerical_equivalence.py
status=$?
if [ $status -eq 0 ]; then
    echo "✓ All algorithms equivalent/tracking"
else
    echo "✗ Comparison found mismatching algorithms (see numerical_equivalence_<os>_<gpu>.md)"
fi
exit $status
