#!/bin/bash
# Generate benchmark data for one or more packages across one or more analyses.
#
# Usage: ./run_all_benchmarks.sh [-p <packages>] [-a <analyses>] [-n <nmax>] [-g <algorithms>]
#   -p, --package   all (default) | comma list of julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
#   -a, --analysis  performance (default) | comma list of performance | work-precision | numerical | all
#   -n, --nmax      sweep ceiling (8, 32, ... <= n; default 16777216) or comma list of exact Ns
#   -g, --algorithm all (default) | comma list of the names in runner_scripts/algorithms.csv
#   -s, --problem   all (default) | comma list of names from runner_scripts/problems.csv
#
# e.g. ./run_all_benchmarks.sh -p cubie,julia -a performance,work-precision -g euler,tsit5 -n 8388608,134217728

# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1

PACKAGE=all
ANALYSIS=performance
NMAX=16777216
ALGORITHM=all
PROBLEM=all

while [ $# -gt 0 ]; do
    case "$1" in
        -p|--package)  PACKAGE=$2; shift 2;;
        -a|--analysis) ANALYSIS=$2; shift 2;;
        -n|--nmax)     NMAX=$2; shift 2;;
        -g|--algorithm) ALGORITHM=$2; shift 2;;
        -s|--problem)  PROBLEM=$2; shift 2;;
        -h|--help)     sed -n '2,10p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "Unknown option $1" >&2; exit 1;;
    esac
done

# Charset-check each list before its unquoted split; tokens validated below.
case "$ANALYSIS" in
    ''|*[!a-z,-]*) echo "Unknown analysis '$ANALYSIS' (performance|work-precision|numerical|all)" >&2; exit 1;;
esac
case "$PACKAGE" in
    ''|*[!a-z0-9,_-]*) echo "Unknown package '$PACKAGE'" >&2; exit 1;;
esac
case "$ALGORITHM" in
    ''|*[!a-z0-9,_-]*) echo "-g/--algorithm must be 'all' or a comma list of algorithm names" >&2; exit 1;;
esac
case "$PROBLEM" in
    ''|*[!a-z0-9_,-]*) echo "-s/--problem takes names from runner_scripts/problems.csv, got '$PROBLEM'" >&2; exit 1;;
esac

DO_PERF=false
DO_WP=false
DO_NE=false
for item in ${ANALYSIS//,/ }; do
    case "$item" in
        performance) DO_PERF=true;;
        work-precision) DO_WP=true;;
        numerical) DO_NE=true;;
        all) DO_PERF=true; DO_WP=true; DO_NE=true;;
        *) echo "Unknown analysis '$item' (performance|work-precision|numerical|all)" >&2; exit 1;;
    esac
done
if ! $DO_PERF && ! $DO_WP && ! $DO_NE; then
    echo "-a/--analysis requires a value" >&2
    exit 1
fi

ALL_PACKAGES="julia cpp pytorch jax cubie cubie_mlir myokit_cuda"
PACKAGE=${PACKAGE//-/_}
PACKAGES=
HAS_ALL_PACKAGES=false
for pkg in ${PACKAGE//,/ }; do
    case "$pkg" in
        all) HAS_ALL_PACKAGES=true;;
        julia|cpp|pytorch|jax|cubie|cubie_mlir|myokit_cuda) PACKAGES="$PACKAGES $pkg";;
        *) echo "Unknown package '$pkg' (all|${ALL_PACKAGES// /|})" >&2; exit 1;;
    esac
done
$HAS_ALL_PACKAGES && PACKAGES="$ALL_PACKAGES"
[ -n "$PACKAGES" ] || { echo "-p/--package requires a value" >&2; exit 1; }

case ",$NMAX," in
    *[!0-9,]*|*,,*)
        echo "-n/--nmax must be a positive integer or a comma list of them, got '$NMAX'" >&2
        exit 1;;
esac

run_sweep() {
    local analysis=$1
    for pkg in $PACKAGES; do
        echo "========================================="
        echo "$analysis: $pkg"
        echo "========================================="
        if bash ./run_benchmark.sh -p "$pkg" -a "$analysis" -n "$NMAX" -g "$ALGORITHM" -s "$PROBLEM" -d gpu -m ode; then
            echo "Completed $analysis for $pkg"
        else
            echo "Error during $analysis for $pkg; continuing with the next package"
        fi
        echo ""
    done
}

if $DO_PERF; then
    run_sweep performance
    echo "--- Timing comparison plot ---"
    julia --project=. ./runner_scripts/plot/plot_ode_comp.jl || echo "Timing plot failed"
fi

if $DO_WP; then
    run_sweep work-precision
    echo "--- Work-precision plot ---"
    julia --project=. ./runner_scripts/plot/plot_ode_wp.jl || echo "Work-precision plot failed"
fi

if $DO_NE; then
    # The numerical-equivalence suite only covers julia and cubie.
    NE_PACKAGE=
    HAS_JULIA=false
    HAS_CUBIE=false
    for pkg in $PACKAGES; do
        [ "$pkg" == "julia" ] && HAS_JULIA=true
        [ "$pkg" == "cubie" ] && HAS_CUBIE=true
    done
    if $HAS_ALL_PACKAGES || { $HAS_JULIA && $HAS_CUBIE; }; then
        NE_PACKAGE=all
    elif $HAS_JULIA; then
        NE_PACKAGE=julia
    elif $HAS_CUBIE; then
        NE_PACKAGE=cubie
    fi
    if [ -n "$NE_PACKAGE" ]; then
        bash ./run_numerical_equivalence.sh -p "$NE_PACKAGE" -s "$PROBLEM" || echo "Numerical equivalence reported problems"
    else
        echo "Numerical equivalence skipped: no requested package is in the suite (all|julia|cubie)"
    fi
fi

echo "--- Pairwise numerical comparison ---"
if [ -f ./GPU_ODE_CUBIE/venv/bin/python3 ]; then
    ./GPU_ODE_CUBIE/venv/bin/python3 compare_numerical_results.py || echo "Pairwise comparison failed"
elif [ -f ./GPU_ODE_CUBIE/venv/Scripts/python.exe ]; then
    ./GPU_ODE_CUBIE/venv/Scripts/python.exe compare_numerical_results.py || echo "Pairwise comparison failed"
else
    echo "GPU_ODE_CUBIE venv not found; skipping pairwise comparison"
fi
