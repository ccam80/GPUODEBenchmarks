#!/bin/bash
# Setup Julia environment for GPU ODE benchmarking
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up Julia environment..."

# Check if Julia is available
if ! command -v julia &> /dev/null; then
    echo "Error: julia is not installed"
    echo "Please install Julia from https://julialang.org/downloads/"
    exit 1
fi

echo "Julia version:"
julia --version

# Instantiate and precompile the main project
echo "Installing Julia packages for main project..."
julia --project=. -e '
    using Pkg
    Pkg.instantiate()
    Pkg.resolve()
    Pkg.precompile()
'

# Add CUDA for GPU support (NVIDIA)
echo "Adding CUDA package for GPU support..."
julia --project=. -e '
    using Pkg
    Pkg.add("CUDA")
'

echo "Julia environment setup complete!"
echo "To test the installation, run:"
echo "  julia --project=. -e 'using DiffEqGPU, CUDA'"
