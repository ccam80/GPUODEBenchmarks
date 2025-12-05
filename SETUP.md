# Environment Setup Guide

This guide explains how to set up the environments for all GPU ODE benchmark packages.

## Quick Start - Setup All Environments

To set up all environments at once, run:

**Cross-platform (Windows/Linux/macOS):**
```bash
python3 setup_all_environments.py
```

**Linux/macOS (bash):**
```bash
./setup_all_environments.sh
```

This master script will automatically:
1. Set up the CUBIE Python environment
2. Set up the JAX/Diffrax Python environment
3. Set up the PyTorch/torchdiffeq Python environment
4. Set up the Julia environment with all required packages

## Individual Package Setup

You can also set up each package environment individually:

### CUBIE

**Cross-platform (Windows/Linux/macOS):**
```bash
python3 GPU_ODE_CUBIE/setup_environment.py
```

**Linux/macOS (bash):**
```bash
./GPU_ODE_CUBIE/setup_environment.sh
```

This will:
- Create a Python virtual environment in `GPU_ODE_CUBIE/venv`
- Install `uv` package manager
- Clone and install the CUBIE library from source

To activate:
- Linux/macOS: `source GPU_ODE_CUBIE/venv/bin/activate`
- Windows (cmd): `GPU_ODE_CUBIE\venv\Scripts\activate.bat`
- Windows (PowerShell): `GPU_ODE_CUBIE\venv\Scripts\Activate.ps1`

### JAX (Diffrax)

**Cross-platform (Windows/Linux/macOS):**
```bash
python3 GPU_ODE_JAX/setup_environment.py
```

**Linux/macOS (bash):**
```bash
./GPU_ODE_JAX/setup_environment.sh
```

This will:
- Create a Python virtual environment in `GPU_ODE_JAX/venv`
- Install `uv` package manager
- Install JAX with CUDA support
- Install Diffrax, Equinox, and other dependencies

To activate:
- Linux/macOS: `source GPU_ODE_JAX/venv/bin/activate`
- Windows (cmd): `GPU_ODE_JAX\venv\Scripts\activate.bat`
- Windows (PowerShell): `GPU_ODE_JAX\venv\Scripts\Activate.ps1`

### PyTorch (torchdiffeq)

**Cross-platform (Windows/Linux/macOS):**
```bash
python3 GPU_ODE_PyTorch/setup_environment.py
```

**Linux/macOS (bash):**
```bash
./GPU_ODE_PyTorch/setup_environment.sh
```

This will:
- Create a Python virtual environment in `GPU_ODE_PyTorch/venv`
- Install `uv` package manager
- Install PyTorch with CUDA support
- Install the custom torchdiffeq fork with vmap support

To activate:
- Linux/macOS: `source GPU_ODE_PyTorch/venv/bin/activate`
- Windows (cmd): `GPU_ODE_PyTorch\venv\Scripts\activate.bat`
- Windows (PowerShell): `GPU_ODE_PyTorch\venv\Scripts\Activate.ps1`

### Julia

**Cross-platform (Windows/Linux/macOS):**
```bash
python3 setup_julia.py
```

**Linux/macOS (bash):**
```bash
./setup_julia.sh
```

This will:
- Install all Julia packages from `Project.toml`
- Add CUDA.jl for GPU support
- Resolve and precompile all packages

To use: `julia --project=.`

## Requirements

### Python Packages
- Python 3.8 or higher
- pip (included with Python)

### Julia
- Julia 1.8 or higher (Julia 1.9+ recommended for AMD GPU support)
- Download from: https://julialang.org/downloads/

### GPU Support
- For NVIDIA GPUs: CUDA Toolkit 11.x or 12.x
- For AMD GPUs: ROCm (see Julia 1.9+ requirements)
- For Intel GPUs: oneAPI
- For Apple M-series: Metal (macOS)

## Verification

After setup, verify each environment:

### CUBIE
```bash
source GPU_ODE_CUBIE/venv/bin/activate
python -c "import cubie; print('CUBIE OK')"
deactivate
```

### JAX
```bash
source GPU_ODE_JAX/venv/bin/activate
python -c "import jax, diffrax; print('JAX OK')"
deactivate
```

### PyTorch
```bash
source GPU_ODE_PyTorch/venv/bin/activate
python -c "import torch, torchdiffeq; print('PyTorch OK')"
deactivate
```

### Julia
```bash
julia --project=. -e 'using DiffEqGPU, CUDA; println("Julia OK")'
```

## Troubleshooting

### Python Virtual Environments

If a setup fails, you can clean up and retry:

**Linux/macOS:**
```bash
rm -rf GPU_ODE_CUBIE/venv GPU_ODE_CUBIE/cubie
rm -rf GPU_ODE_JAX/venv
rm -rf GPU_ODE_PyTorch/venv
```

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force GPU_ODE_CUBIE\venv, GPU_ODE_CUBIE\cubie
Remove-Item -Recurse -Force GPU_ODE_JAX\venv
Remove-Item -Recurse -Force GPU_ODE_PyTorch\venv
```

Then re-run the appropriate setup script.

### Julia Package Issues

If Julia package installation fails:
```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.update()'
```

### CUDA/GPU Issues

Ensure your GPU drivers and CUDA toolkit are properly installed:
```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA compiler (for C++ benchmarks)
nvcc --version
```

## Notes

- All Python virtual environments are created with the name `venv` in their respective package directories
- The CUBIE setup clones the repository into `GPU_ODE_CUBIE/cubie/`
- Virtual environment and cloned repository directories are excluded from git via `.gitignore`
- The `uv` package manager is used for faster Python package installation

## Platform Compatibility

The repository now includes two versions of setup scripts:

1. **Python scripts** (`.py` files) - **Recommended for all users**
   - Work on Linux, Windows, and macOS
   - Automatically detect the operating system and use appropriate paths
   - Can be run with `python3 script_name.py`
   
2. **Bash scripts** (`.sh` files) - **Linux/macOS only**
   - Original scripts that work on Unix-like systems
   - Can be run with `./script_name.sh` or `bash script_name.sh`

Both versions provide identical functionality. The Python scripts are recommended as they work across all platforms.
