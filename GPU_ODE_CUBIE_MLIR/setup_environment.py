#!/usr/bin/env python3
"""
Cross-platform setup script for the CUBIE (MLIR backend) ODE benchmarking
environment.

Installs cubie from its `main` branch. Since ccam80/cubie#617 the MLIR backend
lives on `main` alongside the default numba-cuda backend; the active backend is
chosen at import time by the ``CUBIE_CUDA_BACKEND`` environment variable
(``numba-cuda`` | ``mlir``). This suite runs the MLIR pipeline, so the
benchmark launchers export ``CUBIE_CUDA_BACKEND=mlir``.

The patched ``numba-cuda-mlir`` build is bundled as a local wheel under
``wheels/`` and installed first so the editable cubie install reuses it instead
of pulling the stock build from PyPI. When no matching wheel is present (e.g. a
different platform/Python), the ``dev-mlir13`` extra pulls numba-cuda-mlir from
PyPI as a fallback.

Works on Linux, Windows, and macOS.
"""
import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

CUBIE_BRANCH = "main"


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

    print("Setting up CUBIE-MLIR environment...")

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

    # Check if git is available
    if not shutil.which("git"):
        print("Error: git is not installed")
        print("Please install git from https://git-scm.com/downloads")
        return 1

    # Clone cubie source (main branch), or bring an existing clone onto main.
    cubie_dir = script_dir / "cubie"
    if cubie_dir.exists():
        print(f"Cubie directory already exists; ensuring it is on '{CUBIE_BRANCH}'...")
        if not run_command(["git", "fetch", "origin"], cwd=cubie_dir):
            print("Warning: git fetch failed; using the existing checkout as-is.")
        elif not (run_command(["git", "checkout", CUBIE_BRANCH], cwd=cubie_dir)
                  and run_command(["git", "pull", "--ff-only", "origin", CUBIE_BRANCH],
                                  cwd=cubie_dir)):
            print(f"Error: could not switch the existing clone to '{CUBIE_BRANCH}'.")
            print("(Delete the 'cubie' directory to force a fresh clone.)")
            return 1
    else:
        print(f"Cloning cubie repository (branch: {CUBIE_BRANCH})...")
        if not run_command(["git", "clone", "--branch", CUBIE_BRANCH,
                            "https://github.com/ccam80/cubie.git"]):
            print("Error: Failed to clone cubie repository")
            return 1

    # Install the patched numba-cuda-mlir from the bundled local wheel first so
    # the editable cubie install below reuses it rather than pulling the stock
    # build from PyPI. The wheel is platform/Python-specific; when none matches,
    # the dev-mlir13 extra installs numba-cuda-mlir from PyPI as a fallback.
    wheels_dir = script_dir / "wheels"
    patched_wheels = sorted(wheels_dir.glob("numba_cuda_mlir-*.whl")) if wheels_dir.exists() else []
    if patched_wheels:
        wheel = patched_wheels[-1]
        print(f"Installing patched numba-cuda-mlir wheel: {wheel.name}")
        if not run_command([str(venv_uv), "pip", "install", "-p", str(venv_python), str(wheel)]):
            print("Failed to install patched numba-cuda-mlir wheel")
            return 1
    else:
        print("No bundled numba-cuda-mlir wheel found; installing from PyPI via the extra.")

    # Install cubie (main) with the MLIR + test dependency set. numba-cuda-mlir
    # is already satisfied by the patched wheel above when present; the backend
    # is selected at runtime through CUBIE_CUDA_BACKEND, not at install time.
    print("Installing cubie (main) and dependencies...")
    if not run_command([str(venv_uv), "pip", "install", "-p", str(venv_python),
                        "-e", ".[dev-mlir13]"], cwd=cubie_dir):
        print("Failed to install cubie")
        return 1

    # Verify the install and that cubie resolves to the MLIR backend when
    # CUBIE_CUDA_BACKEND=mlir. The backend is read once at import time.
    print("Verifying installation (MLIR backend)...")
    verify_env = os.environ.copy()
    verify_env["CUBIE_CUDA_BACKEND"] = "mlir"
    verify_code = (
        "import cubie; "
        "from cubie.cuda_backend import CUDA_BACKEND, IS_MLIR; "
        "assert IS_MLIR, 'resolved backend: ' + CUDA_BACKEND; "
        "print('Cubie', cubie.__version__, 'installed; backend =', CUDA_BACKEND)"
    )
    if not run_command([str(venv_python), "-c", verify_code], env=verify_env):
        print("Failed to import cubie or the MLIR backend did not activate")
        return 1

    if not run_command([str(venv_python), "-c",
                        "from numba_cuda_mlir import cuda; print('CUDA available:', cuda.is_available())"]):
        print("Warning: CUDA verification failed")

    print("\nCUBIE-MLIR environment setup complete!")
    print("Backend is selected at runtime via CUBIE_CUDA_BACKEND=mlir "
          "(the benchmark launchers set this automatically).")
    if is_windows:
        print(f"To activate: {venv_path / 'Scripts' / 'activate.bat'}")
        print(f"Or in PowerShell: {venv_path / 'Scripts' / 'Activate.ps1'}")
    else:
        print(f"To activate: source {venv_path / 'bin' / 'activate'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
