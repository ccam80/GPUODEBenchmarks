#!/bin/bash
# Run the numerical-equivalence analysis: golden reference (if missing),
# DifferentialEquations.jl Float32 reference sweeps, cubie Float32 sweeps, and
# the comparison report + plots.
#
# Usage: ./run_numerical_equivalence.sh [-p <package>] [--controller <c>] [--algorithm <a>]
#   -p, --package     all (default) | julia | cubie
#   --controller      all (default) | fixed | adaptive
#   --algorithm       all (default) | a cubie alias from algorithms.csv
#   -s, --problem     all (default) | comma list of names from runner_scripts/problems.csv
#
# Exit code: non-zero if any step fails.

# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1

PACKAGE=all
CONTROLLER=all
ALGORITHM=all
PROBLEM=all

while [ $# -gt 0 ]; do
    case "$1" in
        -p|--package)    PACKAGE=$2; shift 2;;
        --controller)    CONTROLLER=$2; shift 2;;
        --algorithm)     ALGORITHM=$2; shift 2;;
        -s|--problem)    PROBLEM=$2; shift 2;;
        -h|--help)       sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "Unknown option $1" >&2; exit 1;;
    esac
done
case "$PACKAGE" in all|julia|cubie) ;; *) echo "Unknown package '$PACKAGE' (all|julia|cubie)" >&2; exit 1;; esac
case "$CONTROLLER" in all|fixed|adaptive) ;; *) echo "Unknown controller '$CONTROLLER' (all|fixed|adaptive)" >&2; exit 1;; esac

echo "========================================="
echo "Numerical equivalence (package: $PACKAGE, controller: $CONTROLLER, algorithm: $ALGORITHM, problem: $PROBLEM)"
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
    echo "--- Golden references (Float64, machine independent) ---"
    julia -t auto --project=. ./runner_scripts/numerical_equivalence/generate_golden_ne.jl --problem "$PROBLEM" || {
        echo "golden generation failed" >&2; exit 1; }
    echo "--- DifferentialEquations.jl Float32 sweeps (CPU, machine independent) ---"
    julia -t auto --project=. ./runner_scripts/numerical_equivalence/ne_diffeq.jl --controller "$CONTROLLER" --algorithm "$ALGORITHM" --problem "$PROBLEM" || {
        echo "DifferentialEquations.jl sweeps failed" >&2; exit 1; }
fi

if [ "$PACKAGE" == "all" ] || [ "$PACKAGE" == "cubie" ]; then
    echo "--- cubie Float32 sweeps (GPU, keyed per machine) ---"
    # The venv carries both CUDA backends; the committed dataset is the numba-cuda one.
    export CUBIE_CUDA_BACKEND="${CUBIE_CUDA_BACKEND:-numba-cuda}"
    echo "    (cubie backend: $CUBIE_CUDA_BACKEND)"
    "$PYTHON" ./GPU_ODE_CUBIE/numerical_equivalence.py --controller "$CONTROLLER" --algorithm "$ALGORITHM" --problem "$PROBLEM" || {
        echo "cubie sweeps failed" >&2; exit 1; }
fi

echo "--- Comparison tables + plots ---"
"$PYTHON" compare_numerical_equivalence.py --problem "$PROBLEM"
exit $?
