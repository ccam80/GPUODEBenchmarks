# Environment Setup Guide

This guide explains how to set up the environments for all GPU ODE benchmark packages.

## Quick Start - Setup All Environments

To set up all environments at once, run:

```bash
python3 setup_all_environments.py
```

This master script will automatically:
1. Set up the CUBIE Python environment
2. Set up the JAX/Diffrax Python environment
3. Set up the PyTorch/torchdiffeq Python environment
4. Set up the Julia environment with all required packages

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

### MPGOS (C++)

MPGOS is a C++ library for GPU-accelerated ODE solving that requires manual setup as it has external dependencies that need to be configured outside of this repository.

**Prerequisites:**
- CUDA Toolkit 11.x or 12.x (includes `nvcc` compiler)
- C++ compiler:
  - **Linux**: GCC (typically pre-installed or available via package manager)
  - **Windows**: Microsoft Visual C++ compiler (`cl.exe`) from Visual Studio or Build Tools
  - **macOS**: Clang (included with Xcode Command Line Tools)

#### Windows Setup

On Windows, `nvcc` requires the Microsoft Visual C++ compiler to be in your PATH. Follow these steps:

1. **Install Visual Studio Build Tools** (if you don't have Visual Studio):
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Under "All Downloads" → "Tools for Visual Studio" → Download "Build Tools for Visual Studio"
   - During installation, select "Desktop development with C++" workload
   - Ensure "MSVC" and "Windows SDK" components are selected

2. **Alternative**: Install full Visual Studio Community Edition (free) which includes all necessary build tools

3. **Setup environment for compilation**:
   
   Option A - Use Developer Command Prompt (Recommended):
   ```powershell
   # Open "Developer Command Prompt for VS" or "Developer PowerShell for VS" from Start Menu
   # This automatically sets up the PATH for cl.exe and other build tools
   ```
   
   Option B - Manually add to PATH:
   ```powershell
   # Add Visual Studio tools to PATH (adjust version/path as needed)
   # Typical locations:
   # C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64
   # C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\<version>\bin\Hostx64\x64
   ```

4. **Verify installation**:
   ```bash
   # Check CUDA compiler
   nvcc --version
   
   # Check C++ compiler
   cl
   ```
   
   If `cl` is found, you should see: "Microsoft (R) C/C++ Optimizing Compiler"

#### Linux Setup

On Linux, ensure you have GCC and CUDA Toolkit installed:

```bash
# Install GCC (if not already installed)
sudo apt-get update
sudo apt-get install build-essential

# Verify installation
nvcc --version
gcc --version
```

#### macOS Setup

On macOS, ensure you have Clang and CUDA Toolkit (if applicable):

```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# Verify installation
nvcc --version  # If using NVIDIA GPU
clang --version
```

#### Building MPGOS

Once the prerequisites are installed, MPGOS programs can be built using make:

```bash
# Linux/macOS
cd GPU_ODE_MPGOS
make clean
make

# Windows (in Developer Command Prompt or with cl.exe in PATH)
cd GPU_ODE_MPGOS
make clean
make
```

**Note**: The MPGOS source code is included in this repository under `GPU_ODE_MPGOS/SourceCodes/`, so no additional downloads are required. The setup only involves ensuring the build environment (compilers) is properly configured.

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

### MPGOS (C++)

**Linux/macOS:**
```bash
cd GPU_ODE_MPGOS
make clean
make
./Lorenz.exe 8
```

**Windows (Developer Command Prompt):**
```cmd
cd GPU_ODE_MPGOS
make clean
make
Lorenz.exe 8
```

If the compilation succeeds and the program runs, you should see output with timing information.

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

### MPGOS/C++ Compilation Issues

#### Windows: "Cannot find compiler 'cl.exe' in PATH"

This error occurs when NVCC cannot find the Microsoft Visual C++ compiler. Solutions:

1. **Use Developer Command Prompt** (Easiest):
   - Open "Developer Command Prompt for VS" or "Developer PowerShell for VS" from the Start Menu
   - Navigate to the repository directory
   - Run your build/benchmark commands

2. **Add Visual Studio to PATH permanently**:
   - Find the location of `cl.exe` (typically in `C:\Program Files\Microsoft Visual Studio\<version>\<edition>\VC\Tools\MSVC\<version>\bin\Hostx64\x64`)
   - Add this directory to your system PATH environment variable
   - Restart your terminal/PowerShell

3. **Run vcvars64.bat before compiling**:
   ```powershell
   # Adjust path based on your Visual Studio installation
   # Examples:
   # Visual Studio 2022 Community: "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
   # Visual Studio 2019 Professional: "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
   # Build Tools: "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
   cd path\to\GPUODEBenchmarks
   # Now run your build commands
   ```

4. **Verify the fix**:
   ```bash
   cl
   # Should output: Microsoft (R) C/C++ Optimizing Compiler...
   ```

#### Linux: Missing build-essential

If you get compilation errors on Linux:
```bash
sudo apt-get update
sudo apt-get install build-essential
```

#### GPU Architecture Warnings

If you see warnings like "Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed":
- This is a warning, not an error. The code will still compile and run.
- To suppress: Add `-Wno-deprecated-gpu-targets` to compiler flags in the makefile
- Or update the `--gpu-architecture` flag in `GPU_ODE_MPGOS/makefile` to match your GPU's compute capability (e.g., `sm_75`, `sm_80`, `sm_86`)

## Notes

- All Python virtual environments are created with the name `venv` in their respective package directories
- The CUBIE setup clones the repository into `GPU_ODE_CUBIE/cubie/`
- Virtual environment and cloned repository directories are excluded from git via `.gitignore`
- The `uv` package manager is used for faster Python package installation

## Platform Compatibility

All setup scripts are written in Python and work cross-platform on Linux, Windows, and macOS. They automatically detect the operating system and use appropriate paths.
