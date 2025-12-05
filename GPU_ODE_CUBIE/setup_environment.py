#!/usr/bin/env python3
"""
Cross-platform setup script for CUBIE ODE benchmarking environment.
Works on both Linux and Windows.
"""
import os
import sys
import subprocess
import shutil
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
    
    print("Setting up CUBIE environment...")
    
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
    
    # Clone cubie source
    print("Cloning cubie repository...")
    cubie_dir = script_dir / "cubie"
    if cubie_dir.exists():
        print("Cubie directory already exists, removing...")
        shutil.rmtree(cubie_dir)
    
    if not run_command(["git", "clone", "https://github.com/ccam80/cubie.git"]):
        print("Error: Failed to clone cubie repository")
        return 1
    
    # Install cubie from source using uv
    print("Installing cubie and dependencies...")
    if not run_command([str(venv_uv), "pip", "install", "-e", ".[dev]"], cwd=cubie_dir):
        print("Failed to install cubie")
        return 1
    
    # Verify installation
    print("Verifying installation...")
    if not run_command([str(venv_python), "-c", "import cubie; print('Cubie installed successfully')"]):
        print("Failed to import cubie")
        return 1
    
    if not run_command([str(venv_python), "-c", "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"]):
        print("Warning: CUDA verification failed")
    
    print("\nCUBIE environment setup complete!")
    if is_windows:
        print(f"To activate: {venv_path / 'Scripts' / 'activate.bat'}")
        print(f"Or in PowerShell: {venv_path / 'Scripts' / 'Activate.ps1'}")
    else:
        print(f"To activate: source {venv_path / 'bin' / 'activate'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
