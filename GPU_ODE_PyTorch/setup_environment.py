#!/usr/bin/env python3
"""
Cross-platform setup script for PyTorch (torchdiffeq) ODE benchmarking environment.
Works on Linux, Windows, and macOS.

PyTorch publishes no wheels for the newest CPython releases, so this venv is
built with the newest *supported* interpreter found on the system rather than
whichever one happens to be running this script (the other suites here are
routinely set up with a newer Python than torch supports).
"""
import os
import re
import shutil
import sys
import subprocess
import platform
from pathlib import Path

# Newest CPython minor with torch wheels on download.pytorch.org. Bump as
# upstream publishes new ABI tags.
MAX_TORCH_MINOR = 13


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


def _interpreter_minor(executable):
    """Return the CPython minor version of `executable`, or None if unknown."""
    try:
        out = subprocess.run(
            [executable, "-c", "import sys; print(sys.version_info[1])"],
            capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    match = re.match(r"^\s*(\d+)\s*$", out.stdout)
    return int(match.group(1)) if match else None


def find_torch_python():
    """Pick the newest interpreter that torch still publishes wheels for.

    Prefers the running interpreter when it qualifies, so an ordinary setup on
    a supported Python behaves exactly as before. Otherwise searches PATH for
    python3.<minor>, newest first.
    """
    if sys.version_info[1] <= MAX_TORCH_MINOR:
        return sys.executable

    print(f"Python 3.{sys.version_info[1]} has no torch wheels "
          f"(torch supports up to 3.{MAX_TORCH_MINOR}); looking for another interpreter...")
    for minor in range(MAX_TORCH_MINOR, 8, -1):
        for name in (f"python3.{minor}", f"python3.{minor}.exe"):
            found = shutil.which(name)
            if found and _interpreter_minor(found) == minor:
                return found
    return None


def main():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    print("Setting up PyTorch/torchdiffeq environment...")

    # Check if a torch-compatible Python is available
    python = find_torch_python()
    if python is None:
        print(f"Error: no CPython <= 3.{MAX_TORCH_MINOR} found on PATH, and torch "
              f"publishes no wheels for 3.{sys.version_info[1]}.")
        print(f"Install one (e.g. `apt install python3.{MAX_TORCH_MINOR}` or "
              f"`uv python install 3.{MAX_TORCH_MINOR}`) and re-run this script.")
        return 1
    try:
        result = subprocess.run([python, "--version"], capture_output=True, text=True)
        print(f"Using Python: {result.stdout.strip()}  ({python})")
    except Exception as e:
        print(f"Error: python3 is not installed: {e}")
        return 1

    # Create or use existing venv. A venv built by an incompatible interpreter
    # would fail the install again, so replace it rather than reusing it.
    venv_path = script_dir / "venv"
    if venv_path.exists():
        existing = venv_path / ("Scripts/python.exe" if platform.system() == "Windows"
                                else "bin/python")
        existing_minor = _interpreter_minor(str(existing)) if existing.exists() else None
        if existing_minor is not None and existing_minor <= MAX_TORCH_MINOR:
            print("Virtual environment already exists, using existing one...")
        else:
            print(f"Existing venv is Python 3.{existing_minor}, which torch does not "
                  f"support; recreating with {python}...")
            shutil.rmtree(venv_path)

    if not venv_path.exists():
        print("Creating virtual environment...")
        if not run_command([python, "-m", "venv", str(venv_path)]):
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
    
    # Install PyTorch with CUDA support and other dependencies
    print("Installing PyTorch with CUDA support and dependencies...")
    # Install PyTorch with CUDA 12.1 support (latest stable version)
    if not run_command([str(venv_uv), "pip", "install", "-p", str(venv_python), "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]):
        print("Failed to install PyTorch")
        return 1
    
    if not run_command([str(venv_uv), "pip", "install", "-p", str(venv_python), "numpy"]):
        print("Failed to install numpy")
        return 1
    
    if not run_command([str(venv_uv), "pip", "install", "-p", str(venv_python), "scipy"]):
        print("Failed to install scipy")
        return 1
    
    # Install custom torchdiffeq fork with vmap support
    print("Installing torchdiffeq with vmap support...")
    if not run_command([str(venv_uv), "pip", "install", "-p", str(venv_python), "git+https://github.com/utkarsh530/torchdiffeq.git@u/vmap"]):
        print("Failed to install torchdiffeq")
        return 1
    
    # Verify installation
    print("Verifying installation...")
    if not run_command([str(venv_python), "-c", "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"]):
        print("Warning: PyTorch verification failed")
    
    if not run_command([str(venv_python), "-c", "import torchdiffeq; print('torchdiffeq installed successfully')"]):
        print("Warning: torchdiffeq verification failed")
    
    print("\nPyTorch/torchdiffeq environment setup complete!")
    if is_windows:
        print(f"To activate: {venv_path / 'Scripts' / 'activate.bat'}")
        print(f"Or in PowerShell: {venv_path / 'Scripts' / 'Activate.ps1'}")
    else:
        print(f"To activate: source {venv_path / 'bin' / 'activate'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
