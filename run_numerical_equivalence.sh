#!/bin/bash
# Run the numerical-equivalence analysis: golden reference (if missing),
# DifferentialEquations.jl Float32 reference sweeps, cubie Float32 sweeps, and
# the comparison report + plots.
#
# Usage: ./run_numerical_equivalence.sh [-p <package>] [--controller <controller>]
#   -p, --package     all (default) | julia | cubie
#   --controller      all (default) | fixed | adaptive
#
# Exit code: non-zero if any step fails or the comparison finds a MISMATCH /
# DIVERGENT algorithm (compare_numerical_equivalence.py exits 2).

# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1

PACKAGE=all
CONTROLLER=all

while [ $# -gt 0 ]; do
    case "$1" in
        -p|--package)    PACKAGE=$2; shift 2;;
        --controller)    CONTROLLER=$2; shift 2;;
        -h|--help)       sed -n '2,11p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "Unknown option $1" >&2; exit 1;;
    esac
done
case "$PACKAGE" in all|julia|cubie) ;; *) echo "Unknown package '$PACKAGE' (all|julia|cubie)" >&2; exit 1;; esac
case "$CONTROLLER" in all|fixed|adaptive) ;; *) echo "Unknown controller '$CONTROLLER' (all|fixed|adaptive)" >&2; exit 1;; esac

echo "========================================="
echo "Numerical equivalence (package: $PACKAGE, controller: $CONTROLLER)"
echo "========================================="

if [ -x ./GPU_ODE_CUBIE/venv/bin/python3 ] || [ -f ./GPU_ODE_CUBIE/venv/bin/python3 ]; then
    PYTHON=./GPU_ODE_CUBIE/venv/bin/python3
elif [ -f ./GPU_ODE_CUBIE/venv/Scripts/python.exe ]; then
    PYTHON=./GPU_ODE_CUBIE/venv/Scripts/python.exe
else
    echo "GPU_ODE_CUBIE venv not found; run setup_all_environments.py first" >&2
    exit 1
fi

if [ "$PACKAGE" == "all" ] || [ "$PACKAGE" == "julia" ]; then
    if [ ! -f ./data/numerical/golden_ne_lorenz_1024.csv ]; then
        echo "--- Golden reference (Float64 Vern9, machine independent) ---"
        julia -t auto --project=. ./runner_scripts/numerical_equivalence/generate_golden_ne.jl || {
            echo "golden generation failed" >&2; exit 1; }
    fi
    echo "--- DifferentialEquations.jl Float32 sweeps (CPU, machine independent) ---"
    julia -t auto --project=. ./runner_scripts/numerical_equivalence/ne_diffeq.jl "$CONTROLLER" || {
        echo "DifferentialEquations.jl sweeps failed" >&2; exit 1; }
fi

if [ "$PACKAGE" == "all" ] || [ "$PACKAGE" == "cubie" ]; then
    echo "--- cubie Float32 sweeps (GPU, keyed per machine) ---"
    # The venv carries both CUDA backends; the committed dataset is the numba-cuda one.
    export CUBIE_CUDA_BACKEND="${CUBIE_CUDA_BACKEND:-numba-cuda}"
    echo "    (cubie backend: $CUBIE_CUDA_BACKEND)"
    "$PYTHON" ./GPU_ODE_CUBIE/numerical_equivalence.py "$CONTROLLER" || {
        echo "cubie sweeps failed" >&2; exit 1; }
fi

echo "--- Comparison report + plots ---"
"$PYTHON" compare_numerical_equivalence.py
status=$?
if [ $status -eq 0 ]; then
    echo "All algorithms equivalent/tracking"
else
    echo "Comparison found mismatching algorithms (see numerical_equivalence_<os>_<gpu>.md)"
fi
exit $status
