# Environment Setup Guide

This guide explains how to set up the environments for all GPU ODE benchmark packages.

## Quick Start - Setup All Environments

To set up all environments at once, run:

```bash
python3 setup_all_environments.py
```

This master script will automatically:
1. Set up the CUBIE Python environment
2. Set up the CUBIE-MLIR Python environment (cubie `mlir` branch on numba-cuda-mlir)
3. Set up the JAX/Diffrax Python environment
4. Set up the PyTorch/torchdiffeq Python environment
5. Set up the Julia environment with all required packages

## Individual Package Setup

You can also set up each package environment individually:

### CUBIE

```bash
python3 GPU_ODE_CUBIE/setup_environment.py
```

This will:
- Create a Python virtual environment in `GPU_ODE_CUBIE/venv`
- Install `uv` package manager
- Clone and install the CUBIE library from source

To activate:
- Linux/macOS: `source GPU_ODE_CUBIE/venv/bin/activate`
- Windows (cmd): `GPU_ODE_CUBIE\venv\Scripts\activate.bat`
- Windows (PowerShell): `GPU_ODE_CUBIE\venv\Scripts\Activate.ps1`

### CUBIE-MLIR

```bash
python3 GPU_ODE_CUBIE_MLIR/setup_environment.py
```

This will:
- Create a Python virtual environment in `GPU_ODE_CUBIE_MLIR/venv`
- Install `uv` package manager
- Clone the `mlir` branch of the CUBIE repository and install it from source
  (pulls the `numba-cuda-mlir` compilation pipeline from PyPI instead of `numba-cuda`)

This environment is intentionally separate from the CUBIE one: both branches
install the same `cubie` package, so keeping them in separate venvs lets the
MLIR and non-MLIR pipelines be benchmarked side by side.

To activate:
- Linux/macOS: `source GPU_ODE_CUBIE_MLIR/venv/bin/activate`
- Windows (cmd): `GPU_ODE_CUBIE_MLIR\venv\Scripts\activate.bat`
- Windows (PowerShell): `GPU_ODE_CUBIE_MLIR\venv\Scripts\Activate.ps1`

### JAX (Diffrax)

```bash
python3 GPU_ODE_JAX/setup_environment.py
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

```bash
python3 GPU_ODE_PyTorch/setup_environment.py
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

```bash
python3 setup_julia.py
```

This will:
- Add CUDA.jl for GPU support first (to avoid compatibility issues)
- Manually add all required Julia packages without pinned versions
- Precompile all packages

To use: `julia --project=.`

## Requirements

### Linux prerequisites

On a fresh Linux machine, install these before running the setup scripts:

```bash
# Compiler toolchain and CUDA compiler (needed by MPGOS, and by
# numba-cuda/cubie unless the CUDA wheels extras are used)
sudo apt install build-essential nvidia-cuda-toolkit python3-venv python3-pip git
# Julia (puts `julia` on PATH for setup_julia.py)
curl -fsSL https://install.julialang.org | sh -s -- --yes
```

The NVIDIA driver must be installed on the host. Do NOT install Linux NVIDIA
driver packages inside WSL — the WSL driver is provided by Windows via
`/usr/lib/wsl/lib`, and a native `libcuda` in `/lib/x86_64-linux-gnu` will
shadow it and break CUDA context creation for native extensions.

JAX and PyTorch bundle their own CUDA runtime via pip wheels
(`jax[cuda12]`, torch cu121), so they need only the driver. JAX has no CUDA
wheels for native Windows; the JAX benchmark aborts on a CPU backend and
should be run on Linux or WSL2.

### Python Packages
- Python 3.10 or higher (3.12 recommended; the numba stack may lag the newest CPython)
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

### CUBIE-MLIR
```bash
source GPU_ODE_CUBIE_MLIR/venv/bin/activate
python -c "import cubie; print('CUBIE-MLIR OK')"
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
rm -rf GPU_ODE_CUBIE_MLIR/venv GPU_ODE_CUBIE_MLIR/cubie
rm -rf GPU_ODE_JAX/venv
rm -rf GPU_ODE_PyTorch/venv
```

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force GPU_ODE_CUBIE\venv, GPU_ODE_CUBIE\cubie
Remove-Item -Recurse -Force GPU_ODE_CUBIE_MLIR\venv, GPU_ODE_CUBIE_MLIR\cubie
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
- The CUBIE-MLIR setup clones the `mlir` branch into `GPU_ODE_CUBIE_MLIR/cubie/`
- Virtual environment and cloned repository directories are excluded from git via `.gitignore`
- The `uv` package manager is used for faster Python package installation

## Platform Compatibility

All setup scripts are written in Python and work cross-platform on Linux, Windows, and macOS. They automatically detect the operating system and use appropriate paths.
