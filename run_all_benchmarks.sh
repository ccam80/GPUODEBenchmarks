#!/bin/bash

# Script to run all GPU ODE benchmarks in sequence
# This allows for set-and-forget benchmarking while the GPU is available

# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1

echo "========================================="
echo "Starting All GPU ODE Benchmarks"
echo "========================================="
echo ""

# Optional: Parse command line arguments for custom nmax, work-precision and
# numerical-precision modes.
# -w also runs the work-precision (error-vs-time) sweeps and their plot.
# -np/--numerical-precision also runs the numerical-equivalence suite: the
# fixed-step error-vs-dt sweeps of every algorithm mutually supported by
# cubie and DifferentialEquations.jl (both in Float32) plus their comparison
# report. Parsed manually because getopts cannot distinguish -np from -n.
nmax_arg=""
wp=false
np=false
while [ $# -gt 0 ]; do
    case "$1" in
        -n) if [ $# -lt 2 ]; then echo "-n requires a value"; exit 1; fi
            nmax_arg="-n $2"; shift 2;;
        -w) wp=true; shift;;
        -np|--numerical-precision) np=true; shift;;
        *) echo "Unknown option $1"
           echo "Usage: $0 [-n nmax] [-w] [-np|--numerical-precision]"; exit 1;;
    esac
done

# Array of languages to benchmark
languages=("julia" "cpp" "pytorch" "jax" "cubie" "cubie_mlir" "myokit_cuda")

# Run timing benchmarks for each language
for lang in "${languages[@]}"
do
    echo "========================================="
    echo "Benchmarking: $lang"
    echo "========================================="

    if bash ./run_benchmark.sh -l "$lang" -d gpu -m ode $nmax_arg; then
        echo ""
        echo "✓ Successfully completed benchmarking for $lang"
        echo ""
    else
        echo ""
        echo "✗ Error occurred while benchmarking $lang"
        echo "Continuing with next language..."
        echo ""
    fi
done

# Optionally run the work-precision sweeps for each language (-w).
if $wp; then
    for lang in "${languages[@]}"
    do
        echo "========================================="
        echo "Work-precision benchmarking: $lang"
        echo "========================================="

        if bash ./run_benchmark.sh -l "$lang" -d gpu -m ode -w; then
            echo ""
            echo "✓ Successfully completed work-precision benchmarking for $lang"
            echo ""
        else
            echo ""
            echo "✗ Error occurred while work-precision benchmarking $lang"
            echo "Continuing with next language..."
            echo ""
        fi
    done
fi

# Optionally run the numerical-equivalence suite (-np/--numerical-precision):
# Float32 fixed-step error-vs-dt sweeps of every mutually supported algorithm,
# for DifferentialEquations.jl (CPU reference) and cubie (GPU), then the
# comparison report + plot.
if $np; then
    if bash ./run_numerical_equivalence.sh; then
        echo "✓ Numerical-equivalence suite completed (all algorithms equivalent/tracking)"
    else
        echo "✗ Numerical-equivalence suite reported problems (see numerical_equivalence_<os>_<gpu>.md)"
    fi
    echo ""
fi

echo "========================================="
echo "All Benchmarks Completed"
echo "========================================="
echo ""

echo "========================================="
echo "Generating timing comparison plot"
echo "========================================="
if julia --project=. ./runner_scripts/plot/plot_ode_comp.jl; then
    echo "✓ Plot saved to ./plots"
else
    echo "✗ Error occurred while generating the timing comparison plot"
fi
echo ""

# Work-precision plot (only meaningful when -w regenerated the wp data).
if $wp; then
    echo "========================================="
    echo "Generating work-precision plot"
    echo "========================================="
    if julia --project=. ./runner_scripts/plot/plot_ode_wp.jl; then
        echo "✓ Plot saved to ./plots"
    else
        echo "✗ Error occurred while generating the work-precision plot"
    fi
    echo ""
fi

echo "========================================="
echo "Comparing numerical results"
echo "========================================="
if source ./GPU_ODE_CUBIE/venv/bin/activate 2>/dev/null; then
    if python3 compare_numerical_results.py; then
        echo "✓ Numerical comparison written to ./pairwise_comparisons.md"
    else
        echo "✗ Error occurred while comparing numerical results"
    fi
    deactivate
else
    echo "✗ Could not activate ./GPU_ODE_CUBIE/venv; skipping numerical comparison"
fi
