#!/usr/bin/env python3
"""
Cross-platform setup script for the CUBIE ODE benchmarking environment.

This builds GPU_ODE_CUBIE/venv: cubie from PyPI (not a git clone) with its
``mlir-cuda13`` extra (cubie-numba-cuda-mlir and CuPy) and its test extra, plus
the result-store and analysis dependencies. An older venv's numba-cuda is
uninstalled, so cubie compiles through cubie-numba-cuda-mlir only.

Works on Linux, Windows, and macOS.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

# CUDA major version to match. The extra pulls the matching
# cubie-numba-cuda-mlir / cupy builds; override with CUBIE_CUDA_MAJOR=12.
CUDA_MAJOR = os.environ.get("CUBIE_CUDA_MAJOR", "13")


def run_command(cmd, shell=False, check=True, cwd=None, env=None):
    """Run a command and handle errors, streaming output in real-time."""
    try:
        # Stream output directly to terminal for real-time feedback
        result = subprocess.run(
            cmd,
            shell=shell,
            check=check,
            cwd=cwd,
            env=env,
            text=True,
            encoding='utf-8',
            errors='replace'  # Replace encoding errors instead of failing
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        return False


def main():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    print("Setting up CUBIE environment...")

    # Check if Python is available
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        print(f"Using Python: {result.stdout.strip()}")
    except Exception as e:
        print(f"Error: python3 is not installed: {e}")
        return 1

    # Create or use existing venv
    venv_path = script_dir / "venv"
    if venv_path.exists():
        print("Virtual environment already exists, using existing one...")
    else:
        print("Creating virtual environment...")
        if not run_command([sys.executable, "-m", "venv", str(venv_path)]):
            print("Failed to create virtual environment")
            return 1

    # Determine the correct paths for the virtual environment
    is_windows = platform.system() == "Windows"
    if is_windows:
        venv_python = venv_path / "Scripts" / "python.exe"
        venv_pip = venv_path / "Scripts" / "pip.exe"
    else:
        venv_python = venv_path / "bin" / "python"
        venv_pip = venv_path / "bin" / "pip"

    # Upgrade pip using python -m pip (required for proper upgrade)
    print("Upgrading pip...")
    if not run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"]):
        print("Failed to upgrade pip")
        return 1

    # Install uv package manager
    print("Installing uv package manager...")
    if not run_command([str(venv_pip), "install", "uv"]):
        print("Failed to install uv")
        return 1

    # Determine uv executable path
    if is_windows:
        venv_uv = venv_path / "Scripts" / "uv.exe"
    else:
        venv_uv = venv_path / "bin" / "uv"

    # Install cubie from PyPI plus the test dependency set.
    # mlir-cuda<N> -> cubie-numba-cuda-mlir[cuN] + cupy-cudaNx
    # cubie_precompile.py calls Solver.compile(optimize_candidates=..., max_parallel=...), which needs 0.14.0.
    spec = f"cubie[mlir-cuda{CUDA_MAJOR},test]>=0.14.0"
    print(f"Installing {spec} from PyPI...")
    if not run_command([str(venv_uv), "pip", "install", "-p", str(venv_python),
                        "--upgrade", spec]):
        print("Failed to install cubie")
        return 1

    # A venv built before cubie went MLIR-only still holds numba-cuda; without it cubie has one compiler to pick.
    print("Removing numba-cuda if present...")
    if not run_command([str(venv_uv), "pip", "uninstall", "-p", str(venv_python), "numba-cuda"]):
        print("Failed to uninstall numba-cuda")
        return 1

    # The suite interpreter: the result store needs pyarrow and DuckDB, the analyses matplotlib.
    print("Installing the result-store and analysis dependencies (pyarrow, duckdb, matplotlib)...")
    if not run_command([str(venv_uv), "pip", "install", "-p", str(venv_python),
                        "pyarrow", "duckdb", "matplotlib"]):
        print("Failed to install pyarrow, duckdb and matplotlib")
        return 1

    print("Verifying installation...")
    if not run_command([str(venv_python), "-c",
                        "import cubie; print('Cubie', cubie.__version__, 'installed')"]):
        print("Failed to import cubie")
        return 1

    if not run_command([str(venv_python), "-c",
                        "from cubie.cuda_simsafe import cuda; print('CUDA available:', cuda.is_available())"]):
        print("Warning: CUDA verification failed")

    if not run_command([str(venv_python), "-c",
                        "import pyarrow, duckdb; print('pyarrow', pyarrow.__version__, "
                        "'duckdb', duckdb.__version__)"]):
        print("Failed to import pyarrow and duckdb")
        return 1

    print("\nCUBIE environment setup complete!")
    if is_windows:
        print(f"To activate: {venv_path / 'Scripts' / 'activate.bat'}")
        print(f"Or in PowerShell: {venv_path / 'Scripts' / 'Activate.ps1'}")
    else:
        print(f"To activate: source {venv_path / 'bin' / 'activate'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
