#!/bin/bash

# Script to run all GPU ODE benchmarks in sequence
# This allows for set-and-forget benchmarking while the GPU is available

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
languages=("julia" "cpp" "pytorch" "jax" "cubie")

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
