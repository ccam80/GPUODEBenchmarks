#!/bin/bash
# Setup Python venv for PyTorch (torchdiffeq) ODE benchmarking
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up PyTorch/torchdiffeq environment..."

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

# Install PyTorch with CUDA support and other dependencies
echo "Installing PyTorch with CUDA support and dependencies..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy
uv pip install scipy

# Install custom torchdiffeq fork with vmap support
echo "Installing torchdiffeq with vmap support..."
uv pip install git+https://github.com/utkarsh530/torchdiffeq.git@u/vmap

# Verify installation
echo "Verifying installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import torchdiffeq; print('torchdiffeq installed successfully')"

# Deactivate
deactivate

echo "PyTorch/torchdiffeq environment setup complete!"
echo "To activate: source $SCRIPT_DIR/venv/bin/activate"
