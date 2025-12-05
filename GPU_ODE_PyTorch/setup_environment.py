#!/usr/bin/env python3
"""
Cross-platform setup script for PyTorch (torchdiffeq) ODE benchmarking environment.
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
    
    print("Setting up PyTorch/torchdiffeq environment...")
    
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
    
    # Upgrade pip
    print("Upgrading pip...")
    if not run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"]):
        print("Failed to upgrade pip")
        return 1
    
    # Install PyTorch with CUDA support and other dependencies
    print("Installing PyTorch with CUDA support and dependencies...")
    # Install PyTorch with CUDA 12.1 support (latest stable version)
    if not run_command([str(venv_python), "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]):
        print("Failed to install PyTorch")
        return 1
    
    if not run_command([str(venv_python), "-m", "pip", "install", "numpy"]):
        print("Failed to install numpy")
        return 1
    
    if not run_command([str(venv_python), "-m", "pip", "install", "scipy"]):
        print("Failed to install scipy")
        return 1
    
    # Install custom torchdiffeq fork with vmap support
    print("Installing torchdiffeq with vmap support...")
    if not run_command([str(venv_python), "-m", "pip", "install", "git+https://github.com/utkarsh530/torchdiffeq.git@u/vmap"]):
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
