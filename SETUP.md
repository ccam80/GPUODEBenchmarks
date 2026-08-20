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
5. Set up the Myokit-CUDA Python environment
6. Set up the Julia environment with all required packages

## Individual Package Setup

You can also set up each package environment individually:

### CUBIE

```bash
python3 GPU_ODE_CUBIE/setup_environment.py
```

This builds the **single shared cubie environment used by both cubie suites**.
It will:
- Create a Python virtual environment in `GPU_ODE_CUBIE/venv`
- Install `uv` package manager
- Install `cubie` from PyPI with *both* CUDA backends present in the one venv:
  `numba-cuda` (via the `cuda13` extra) and `cubie-numba-cuda-mlir` (via the
  `mlir-cuda13` extra), plus the test dependency set
- Verify that cubie imports and resolves correctly under each backend

Set `CUBIE_CUDA_MAJOR=12` to target CUDA 12 wheels instead of 13.

To activate:
- Linux/macOS: `source GPU_ODE_CUBIE/venv/bin/activate`
- Windows (cmd): `GPU_ODE_CUBIE\venv\Scripts\activate.bat`
- Windows (PowerShell): `GPU_ODE_CUBIE\venv\Scripts\Activate.ps1`

### CUBIE-MLIR

```bash
python3 GPU_ODE_CUBIE_MLIR/setup_environment.py
```

There is no separate MLIR environment. Since ccam80/cubie#617 the MLIR backend
ships alongside the default numba-cuda backend in a single cubie install, and
the active backend is chosen at *import time* by the `CUBIE_CUDA_BACKEND`
environment variable (`numba-cuda` | `mlir`). One install therefore serves both
pipelines, and they are benchmarked side by side by launching each with a
different value of that variable.

This script simply points `GPU_ODE_CUBIE_MLIR/venv` at the shared
`GPU_ODE_CUBIE/venv` (a symlink, or a directory junction on Windows) and
verifies that the MLIR backend activates. Run the CUBIE setup first; if the
shared venv is missing this script builds it on demand.

The benchmark launchers export the backend they need, so no manual switching is
required:

| Suite | Launcher sets |
|---|---|
| `GPU_ODE_CUBIE` | `CUBIE_CUDA_BACKEND=numba-cuda` |
| `GPU_ODE_CUBIE_MLIR` | `CUBIE_CUDA_BACKEND=mlir` |

To activate (either path reaches the same venv):
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
- Install a pinned `jax[cuda13]`, Diffrax, Equinox, and other dependencies

Linux only; elsewhere the script prints a skip and exits 0.

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
- Install a pinned PyTorch from the `cu132` wheel index
- Install the custom torchdiffeq fork at a pinned commit, and fail if a
  `torch.vmap` solve through it does not work

To activate:
- Linux/macOS: `source GPU_ODE_PyTorch/venv/bin/activate`
- Windows (cmd): `GPU_ODE_PyTorch\venv\Scripts\activate.bat`
- Windows (PowerShell): `GPU_ODE_PyTorch\venv\Scripts\Activate.ps1`

### Myokit-CUDA

```bash
python3 GPU_ODE_MYOKIT_CUDA/setup_environment.py
```

This will:
- Create a Python virtual environment in `GPU_ODE_MYOKIT_CUDA/venv`
- Install Myokit and the CUDA runtime dependencies used by the adapter
- Verify that the NVIDIA CUDA toolchain is available

The benchmark imports a CellML Lorenz model and uses Myokit's CUDA exporter.
The exporter generates float32 forward Euler device code only, so this
environment contributes fixed-step results and does not provide an adaptive
solver.

To activate:
- Linux/macOS: `source GPU_ODE_MYOKIT_CUDA/venv/bin/activate`
- Windows (cmd): `GPU_ODE_MYOKIT_CUDA\venv\Scripts\activate.bat`
- Windows (PowerShell): `GPU_ODE_MYOKIT_CUDA\venv\Scripts\Activate.ps1`

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

Every suite targets CUDA 13. The JAX, PyTorch, and Myokit setups read the
CUDA major from `nvcc`/`nvidia-smi` and refuse to build against anything
else. JAX and PyTorch bundle their own CUDA runtime via pip wheels, so they
need only the driver.

JAX's CUDA plugins are published for Linux only, and `bench_diffrax.py`
aborts on a CPU backend, so the JAX suite runs on Linux or WSL2.

Myokit-CUDA requires an NVIDIA GPU and CUDA toolchain. Its setup and
benchmark scripts expect the required NVIDIA tools and libraries to be
available on `PATH`.

### Pinned versions

Change these in `GPU_ODE_*/setup_environment.py`. Datasets are only
comparable across machines that share them.

| Package | Pin |
| --- | --- |
| torch | 2.13.0 (`cu132` index) |
| torchdiffeq | `utkarsh530/torchdiffeq` @ `4f4524f` (`u/vmap`) |
| jax / jaxlib | 0.11.1 (`cuda13` extra) |
| diffrax | 0.7.2 |
| equinox | 0.13.8 |
| myokit | 1.39.2 (`GPU_ODE_MYOKIT_CUDA/requirements.txt`) |
| cupy | 14.2.0 (`cupy-cuda13x`) |

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

### Myokit-CUDA
```bash
source GPU_ODE_MYOKIT_CUDA/venv/bin/activate
python -c "import cupy, myokit; print('Myokit-CUDA OK')"
deactivate
```

### Julia
```bash
julia --project=. -e 'using DiffEqGPU, CUDA; println("Julia OK")'
```

## Troubleshooting

### Python Virtual Environments

If a setup fails, you can clean up and retry:

Remove `GPU_ODE_CUBIE_MLIR/venv` (the link) before the shared venv it points at.

**Linux/macOS:**
```bash
rm -rf GPU_ODE_CUBIE_MLIR/venv   # link to the shared venv
rm -rf GPU_ODE_CUBIE/venv        # the shared venv itself
rm -rf GPU_ODE_JAX/venv
rm -rf GPU_ODE_PyTorch/venv
rm -rf GPU_ODE_MYOKIT_CUDA/venv
```

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force GPU_ODE_CUBIE_MLIR\venv   # junction to the shared venv
Remove-Item -Recurse -Force GPU_ODE_CUBIE\venv        # the shared venv itself
Remove-Item -Recurse -Force GPU_ODE_JAX\venv
Remove-Item -Recurse -Force GPU_ODE_PyTorch\venv
Remove-Item -Recurse -Force GPU_ODE_MYOKIT_CUDA\venv
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
- `cubie` is installed from PyPI; no git checkout is made
- The two cubie suites share one venv: `GPU_ODE_CUBIE_MLIR/venv` is a link to
  `GPU_ODE_CUBIE/venv`, which holds both the `numba-cuda` and
  `cubie-numba-cuda-mlir` backends. `CUBIE_CUDA_BACKEND` selects between them
  at import time
- Myokit-CUDA's generated CUDA export directory is excluded via `.gitignore`
- Virtual environment directories are excluded from git via `.gitignore`
- The `uv` package manager is used for faster Python package installation

## Platform Compatibility

All setup scripts are written in Python and work cross-platform on Linux, Windows, and macOS. They automatically detect the operating system and use appropriate paths.
