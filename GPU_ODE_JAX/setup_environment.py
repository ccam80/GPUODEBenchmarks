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


def run_command(cmd, shell=False, check=True, cwd=None):
    """Run a command and handle errors."""
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            check=check,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            print(result.stdout, end='')
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        if e.stderr:
            print(e.stderr)
        return False


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
    
    # Create venv
    print("Creating virtual environment...")
    venv_path = script_dir / "venv"
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
    
    # Upgrade pip and install uv
    print("Installing uv package manager...")
    if not run_command([str(venv_pip), "install", "--upgrade", "pip"]):
        print("Failed to upgrade pip")
        return 1
    
    if not run_command([str(venv_pip), "install", "uv"]):
        print("Failed to install uv")
        return 1
    
    # Determine uv executable path
    if is_windows:
        venv_uv = venv_path / "Scripts" / "uv.exe"
    else:
        venv_uv = venv_path / "bin" / "uv"
    
    # Install JAX with CUDA support and other dependencies
    print("Installing JAX with CUDA support and dependencies...")
    # Install JAX with CUDA 12 support (latest available)
    if not run_command([str(venv_uv), "pip", "install", "--upgrade", "jax[cuda12]"]):
        print("Failed to install JAX")
        return 1
    
    if not run_command([str(venv_uv), "pip", "install", "diffrax"]):
        print("Failed to install diffrax")
        return 1
    
    if not run_command([str(venv_uv), "pip", "install", "equinox"]):
        print("Failed to install equinox")
        return 1
    
    if not run_command([str(venv_uv), "pip", "install", "numpy"]):
        print("Failed to install numpy")
        return 1
    
    if not run_command([str(venv_uv), "pip", "install", "scipy"]):
        print("Failed to install scipy")
        return 1
    
    # Verify installation
    print("Verifying installation...")
    if not run_command([str(venv_python), "-c", "import jax; print('JAX version:', jax.__version__); print('JAX backend:', jax.default_backend())"]):
        print("Warning: JAX verification failed")
    
    if not run_command([str(venv_python), "-c", "import diffrax; print('Diffrax installed successfully')"]):
        print("Warning: Diffrax verification failed")
    
    if not run_command([str(venv_python), "-c", "import equinox; print('Equinox installed successfully')"]):
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
