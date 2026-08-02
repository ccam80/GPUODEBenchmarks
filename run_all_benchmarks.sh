#!/bin/bash
# Generate benchmark data for every package, or one, across one or more analyses.
#
# Usage: ./run_all_benchmarks.sh [-p <package>] [-a <analysis>] [-n <nmax>]
#   -p, --package   all (default) | julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
#   -a, --analysis  performance (default) | work-precision | numerical | all
#   -n, --nmax      largest trajectory count for a performance sweep (default 16777216)

# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1

PACKAGE=all
ANALYSIS=performance
NMAX=16777216

while [ $# -gt 0 ]; do
    case "$1" in
        -p|--package)  PACKAGE=$2; shift 2;;
        -a|--analysis) ANALYSIS=$2; shift 2;;
        -n|--nmax)     NMAX=$2; shift 2;;
        -h|--help)     sed -n '2,7p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "Unknown option $1" >&2; exit 1;;
    esac
done
case "$ANALYSIS" in
    performance|work-precision|numerical|all) ;;
    *) echo "Unknown analysis '$ANALYSIS' (performance|work-precision|numerical|all)" >&2; exit 1;;
esac

ALL_PACKAGES="julia cpp pytorch jax cubie cubie_mlir myokit_cuda"
if [ "$PACKAGE" == "all" ]; then
    PACKAGES="$ALL_PACKAGES"
else
    PACKAGES="$PACKAGE"
fi

run_sweep() {
    local analysis=$1
    for pkg in $PACKAGES; do
        echo "========================================="
        echo "$analysis: $pkg"
        echo "========================================="
        if bash ./run_benchmark.sh -p "$pkg" -a "$analysis" -n "$NMAX" -d gpu -m ode; then
            echo "Completed $analysis for $pkg"
        else
            echo "Error during $analysis for $pkg; continuing with the next package"
        fi
        echo ""
    done
}

if [ "$ANALYSIS" == "performance" ] || [ "$ANALYSIS" == "all" ]; then
    run_sweep performance
    echo "--- Timing comparison plot ---"
    julia --project=. ./runner_scripts/plot/plot_ode_comp.jl || echo "Timing plot failed"
fi

if [ "$ANALYSIS" == "work-precision" ] || [ "$ANALYSIS" == "all" ]; then
    run_sweep work-precision
    echo "--- Work-precision plot ---"
    julia --project=. ./runner_scripts/plot/plot_ode_wp.jl || echo "Work-precision plot failed"
fi

if [ "$ANALYSIS" == "numerical" ] || [ "$ANALYSIS" == "all" ]; then
    # The numerical-equivalence suite only covers julia and cubie.
    case "$PACKAGE" in
        all|julia|cubie)
            bash ./run_numerical_equivalence.sh -p "$PACKAGE" || echo "Numerical equivalence reported problems";;
        *)
            echo "Numerical equivalence skipped: $PACKAGE is not in the suite (all|julia|cubie)";;
    esac
fi

echo "--- Pairwise numerical comparison ---"
if [ -f ./GPU_ODE_CUBIE/venv/bin/python3 ]; then
    ./GPU_ODE_CUBIE/venv/bin/python3 compare_numerical_results.py || echo "Pairwise comparison failed"
elif [ -f ./GPU_ODE_CUBIE/venv/Scripts/python.exe ]; then
    ./GPU_ODE_CUBIE/venv/Scripts/python.exe compare_numerical_results.py || echo "Pairwise comparison failed"
else
    echo "GPU_ODE_CUBIE venv not found; skipping pairwise comparison"
fi
