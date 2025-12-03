#!/bin/bash
# Master script to setup all GPU ODE benchmark environments
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "Setting up all GPU ODE benchmark environments"
echo "========================================="
echo ""

# Track failures
FAILED_SETUPS=()

# Setup CUBIE
echo "========================================="
echo "1/4: Setting up CUBIE environment..."
echo "========================================="
if [ -f "$SCRIPT_DIR/GPU_ODE_CUBIE/setup_environment.sh" ]; then
    if bash "$SCRIPT_DIR/GPU_ODE_CUBIE/setup_environment.sh"; then
        echo "✓ CUBIE setup completed successfully"
    else
        echo "✗ CUBIE setup failed"
        FAILED_SETUPS+=("CUBIE")
    fi
else
    echo "✗ CUBIE setup script not found"
    FAILED_SETUPS+=("CUBIE")
fi
echo ""

# Setup JAX/Diffrax
echo "========================================="
echo "2/4: Setting up JAX/Diffrax environment..."
echo "========================================="
if [ -f "$SCRIPT_DIR/GPU_ODE_JAX/setup_environment.sh" ]; then
    if bash "$SCRIPT_DIR/GPU_ODE_JAX/setup_environment.sh"; then
        echo "✓ JAX/Diffrax setup completed successfully"
    else
        echo "✗ JAX/Diffrax setup failed"
        FAILED_SETUPS+=("JAX")
    fi
else
    echo "✗ JAX setup script not found"
    FAILED_SETUPS+=("JAX")
fi
echo ""

# Setup PyTorch/torchdiffeq
echo "========================================="
echo "3/4: Setting up PyTorch/torchdiffeq environment..."
echo "========================================="
if [ -f "$SCRIPT_DIR/GPU_ODE_PyTorch/setup_environment.sh" ]; then
    if bash "$SCRIPT_DIR/GPU_ODE_PyTorch/setup_environment.sh"; then
        echo "✓ PyTorch/torchdiffeq setup completed successfully"
    else
        echo "✗ PyTorch/torchdiffeq setup failed"
        FAILED_SETUPS+=("PyTorch")
    fi
else
    echo "✗ PyTorch setup script not found"
    FAILED_SETUPS+=("PyTorch")
fi
echo ""

# Setup Julia
echo "========================================="
echo "4/4: Setting up Julia environment..."
echo "========================================="
if [ -f "$SCRIPT_DIR/setup_julia.sh" ]; then
    if bash "$SCRIPT_DIR/setup_julia.sh"; then
        echo "✓ Julia setup completed successfully"
    else
        echo "✗ Julia setup failed"
        FAILED_SETUPS+=("Julia")
    fi
else
    echo "✗ Julia setup script not found"
    FAILED_SETUPS+=("Julia")
fi
echo ""

# Summary
echo "========================================="
echo "Setup Summary"
echo "========================================="
if [ ${#FAILED_SETUPS[@]} -eq 0 ]; then
    echo "✓ All environments setup successfully!"
    echo ""
    echo "To use the environments:"
    echo "  CUBIE:   source GPU_ODE_CUBIE/venv/bin/activate"
    echo "  JAX:     source GPU_ODE_JAX/venv/bin/activate"
    echo "  PyTorch: source GPU_ODE_PyTorch/venv/bin/activate"
    echo "  Julia:   julia --project=."
    exit 0
else
    echo "✗ Some environments failed to setup:"
    for env in "${FAILED_SETUPS[@]}"; do
        echo "  - $env"
    done
    exit 1
fi
