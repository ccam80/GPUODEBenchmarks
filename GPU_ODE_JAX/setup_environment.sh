#!/bin/bash
# Setup Python venv for JAX (Diffrax) ODE benchmarking
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up JAX/Diffrax environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

# Create venv
echo "Creating virtual environment..."
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade pip and install uv
echo "Installing uv package manager..."
pip install --upgrade pip
pip install uv

# Install JAX with CUDA support and other dependencies
echo "Installing JAX with CUDA support and dependencies..."
# Install JAX with CUDA 12 support (latest available)
uv pip install --upgrade "jax[cuda12]"
uv pip install diffrax
uv pip install equinox
uv pip install numpy
uv pip install scipy

# Verify installation
echo "Verifying installation..."
python -c "import jax; print('JAX version:', jax.__version__); print('JAX backend:', jax.default_backend())"
python -c "import diffrax; print('Diffrax installed successfully')"
python -c "import equinox; print('Equinox installed successfully')"

# Deactivate
deactivate

echo "JAX/Diffrax environment setup complete!"
echo "To activate: source $SCRIPT_DIR/venv/bin/activate"
