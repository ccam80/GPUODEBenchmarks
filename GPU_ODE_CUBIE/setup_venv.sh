#!/bin/bash
# Setup Python venv for Cubie benchmarking

# Create venv
python -m venv venv_cubie

# Activate venv
source venv_cubie/bin/activate

# Upgrade pip
pip install --upgrade pip
pip install uv

# Clone cubie source into a subfolder in venv
cd venv_cubie
git clone https://github.com/ccam80/cubie.git
cd cubie

# Install cubie from source using uv
uv pip install .[dev]

# Return to original directory
cd ../..

# Verify installation
python -c "import cubie; print('Cubie installed successfully')"
python -c "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"

# Deactivate
deactivate

echo "Cubie environment setup complete. Activate with: source GPU_ODE_CUBIE/venv_cubie/bin/activate"
