#!/usr/bin/env python3
"""
Cross-platform setup script for JAX (Diffrax) ODE benchmarking environment.
Works on Linux, Windows, and macOS.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner_scripts"))
from cuda_toolkit import detect_cuda_major

JAX_VERSION = "0.11.1"
DIFFRAX_VERSION = "0.7.2"
EQUINOX_VERSION = "0.13.8"

# jax's CUDA plugins are manylinux-only, so the cuda extras work here alone.
CUDA_PLATFORM = "Linux"


def run_command(cmd, shell=False, check=True, cwd=None):
    """Run a command and handle errors, streaming output in real-time."""
    try:
        # Stream output directly to terminal for real-time feedback
        result = subprocess.run(
            cmd,
            shell=shell,
            check=check,
            cwd=cwd,
            text=True,
            encoding='utf-8',
            errors='replace'  # Replace encoding errors instead of failing
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        return False


def jax_requirement():
    """Return the jax requirement, with a CUDA extra where one exists."""
    if platform.system() != CUDA_PLATFORM:
        print(f"{platform.system()} has no CUDA jaxlib wheels; installing the "
              f"CPU build. Run the JAX benchmark on Linux or WSL2 - it aborts "
              f"on a CPU backend.")
        return f"jax=={JAX_VERSION}"
    cuda_major = detect_cuda_major()
    print(f"Detected CUDA {cuda_major}; using the jax[cuda{cuda_major}] extra.")
    return f"jax[cuda{cuda_major}]=={JAX_VERSION}"


def main():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    print("Setting up JAX/Diffrax environment...")

    # Check if Python is available
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        print(f"Using Python: {result.stdout.strip()}")
    except Exception as e:
        print(f"Error: python3 is not installed: {e}")
        return 1

    try:
        jax_spec = jax_requirement()
    except RuntimeError as error:
        print(f"Error: {error}")
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

    # One resolve for the whole stack, so diffrax cannot pull a different jax.
    print(f"Installing {jax_spec}, diffrax {DIFFRAX_VERSION}, "
          f"equinox {EQUINOX_VERSION}...")
    if not run_command([str(venv_uv), "pip", "install", "-p", str(venv_python),
                        jax_spec,
                        f"diffrax=={DIFFRAX_VERSION}",
                        f"equinox=={EQUINOX_VERSION}",
                        "numpy", "scipy"]):
        print("Failed to install the JAX stack")
        return 1

    # Verify installation
    print("Verifying installation...")
    if not run_command([str(venv_python), "-c", "import jax; print('JAX version:', jax.__version__); print('JAX backend:', jax.default_backend())"]):
        print("Warning: JAX verification failed")

    if not run_command([str(venv_python), "-c", "import diffrax; print('Diffrax', diffrax.__version__)"]):
        print("Warning: Diffrax verification failed")

    if not run_command([str(venv_python), "-c", "import equinox; print('Equinox', equinox.__version__)"]):
        print("Warning: Equinox verification failed")

    print("\nJAX/Diffrax environment setup complete!")
    if is_windows:
        print(f"To activate: {venv_path / 'Scripts' / 'activate.bat'}")
        print(f"Or in PowerShell: {venv_path / 'Scripts' / 'Activate.ps1'}")
    else:
        print(f"To activate: source {venv_path / 'bin' / 'activate'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
