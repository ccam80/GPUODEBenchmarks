#!/bin/bash
# Setup Python venv for CUBIE ODE benchmarking
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up CUBIE environment..."

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

# Clone cubie source
echo "Cloning cubie repository..."
if [ -d "cubie" ]; then
    echo "Cubie directory already exists, removing..."
    rm -rf cubie
fi
if ! git clone https://github.com/ccam80/cubie.git; then
    echo "Error: Failed to clone cubie repository"
    exit 1
fi

# Install cubie from source using uv
echo "Installing cubie and dependencies..."
cd cubie
uv pip install -e .[dev]
cd ..

# Verify installation
echo "Verifying installation..."
python -c "import cubie; print('Cubie installed successfully')"
python -c "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"

# Deactivate
deactivate

echo "CUBIE environment setup complete!"
echo "To activate: source $SCRIPT_DIR/venv/bin/activate"
