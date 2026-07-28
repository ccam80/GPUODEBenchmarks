#!/usr/bin/env python3
"""
Cross-platform setup script for the CUBIE ODE benchmarking environment.

This builds the single shared cubie venv used by *both* benchmark suites
(GPU_ODE_CUBIE and GPU_ODE_CUBIE_MLIR). cubie is installed from PyPI (not a
git checkout), with both CUDA backends present in the one environment:

  * numba-cuda              (via the ``cuda13`` extra)
  * cubie-numba-cuda-mlir   (via the ``mlir-cuda13`` extra)

Since ccam80/cubie#617 the active backend is chosen at *import time* by the
``CUBIE_CUDA_BACKEND`` environment variable (``numba-cuda`` | ``mlir``), so a
single install serves both suites; the benchmark launchers export the value
they need. GPU_ODE_CUBIE_MLIR/setup_environment.py links its ``venv`` path at
this one rather than building a second copy.

Works on Linux, Windows, and macOS.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

# CUDA major version to match. The extras pull the matching numba-cuda /
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

    print("Setting up CUBIE environment (shared by CUBIE and CUBIE-MLIR)...")

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

    # Install cubie from PyPI with BOTH backends plus the test dependency set.
    # cuda<N>      -> numba-cuda[cuN] + cupy-cudaNx
    # mlir-cuda<N> -> cubie-numba-cuda-mlir[cuN] + cupy-cudaNx
    # Both live side by side; CUBIE_CUDA_BACKEND picks between them at import.
    spec = f"cubie[cuda{CUDA_MAJOR},mlir-cuda{CUDA_MAJOR},test]"
    print(f"Installing {spec} from PyPI...")
    if not run_command([str(venv_uv), "pip", "install", "-p", str(venv_python),
                        "--upgrade", spec]):
        print("Failed to install cubie")
        return 1

    # Verify each backend resolves under its env var. The backend is read once
    # at import time, so each check runs in a fresh interpreter.
    for backend, expect_mlir in (("numba-cuda", False), ("mlir", True)):
        print(f"Verifying installation (backend: {backend})...")
        verify_env = os.environ.copy()
        verify_env["CUBIE_CUDA_BACKEND"] = backend
        verify_code = (
            "import cubie; "
            "from cubie.cuda_backend import CUDA_BACKEND, IS_MLIR; "
            f"assert IS_MLIR is {expect_mlir}, 'resolved backend: ' + CUDA_BACKEND; "
            "print('Cubie', cubie.__version__, 'installed; backend =', CUDA_BACKEND)"
        )
        if not run_command([str(venv_python), "-c", verify_code], env=verify_env):
            print(f"Failed to import cubie under CUBIE_CUDA_BACKEND={backend}")
            return 1

    if not run_command([str(venv_python), "-c",
                        "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"]):
        print("Warning: CUDA verification failed")

    print("\nCUBIE environment setup complete!")
    print("Both backends live in this one venv; select with "
          "CUBIE_CUDA_BACKEND=numba-cuda|mlir (the benchmark launchers set it).")
    if is_windows:
        print(f"To activate: {venv_path / 'Scripts' / 'activate.bat'}")
        print(f"Or in PowerShell: {venv_path / 'Scripts' / 'Activate.ps1'}")
    else:
        print(f"To activate: source {venv_path / 'bin' / 'activate'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
