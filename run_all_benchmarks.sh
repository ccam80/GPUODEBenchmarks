#!/bin/bash

# Script to run all GPU ODE benchmarks in sequence
# This allows for set-and-forget benchmarking while the GPU is available

# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1

echo "========================================="
echo "Starting All GPU ODE Benchmarks"
echo "========================================="
echo ""

# Optional: Parse command line arguments for custom nmax
nmax_arg=""
while getopts n: flag
do
    case "${flag}" in
        n) nmax_arg="-n ${OPTARG}";;
        \?) echo "Unknown option -$OPTARG"; echo "Usage: $0 [-n nmax]"; exit 1;;
    esac
done

# Array of languages to benchmark
languages=("julia" "cpp" "pytorch" "jax" "cubie" "cubie_mlir")

# Run benchmarks for each language
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
